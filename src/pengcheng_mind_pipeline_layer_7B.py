from mindspore.nn import Cell

import mindspore
from mindspore import ops, set_dump
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore import nn
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator
from mindspore import log as logger
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode
from mindspore.log import _LogActionOnce
from mindspore.nn.transformer.layers import _LayerNorm, \
    _args_type_validator_check, _valid_type_checks, _valid_value_checks, \
    _check_past_none_input_none, _check_input_dtype, _args_type_validator_check, _Linear
#from mindspore.nn.transformer import FeedForward, TransformerOpParallelConfig
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.log import _LogActionOnce
from mindspore.nn.transformer.op_parallel_config import default_dpmp_config, OpParallelConfig, \
    _Config, _check_config, MoEParallelConfig
from mindspore.nn.transformer.moe import default_moe_config, MoE, _check_moe_config
import numpy as np
import math

from mindspore.nn.layer.activation import get_activation
# from flash_attention.flash_attention_primitive import FlashAttentionPrimitive
from flash_attention.flash_attention_ops import FlashAttentionPrimitive



class FeedForward(Cell):
    @_LogActionOnce(logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                dropout_rate=Validator.check_non_negative_float,
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"),
                                parallel_config=_valid_type_checks([OpParallelConfig, MoEParallelConfig],
                                                                   "FeedForward"))
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 expert_num=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(FeedForward, self).__init__()
        if hidden_act is None or not (isinstance(hidden_act, str) or issubclass(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward cell, the hidden_act should str type or nn.Cell type, "
                            f"but got {hidden_act}.")
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))
            self.mul = P.Mul().shard(((dp, mp,), (dp, mp,)))
            input_size = hidden_size
            hidden_act = 'fast_gelu'
            if hidden_act == "swiglu":
                def Swiglu(x):
                    x = ops.chunk(x, 2, axis=-1)
                    return self.mul(ops.silu(x[0]), x[1])
                # new pcmind go to this stage
                output_size = ffn_hidden_size * 2
                self.activation_func = Swiglu
                # Project to ffn_hidden_size
                self.mapping = _Linear(in_channels=input_size,
                                       out_channels=output_size,
                                       transpose_b=False,
                                       expert_num=expert_num,
                                       expert_group_size=expert_group_size,
                                       outer_batch=dp,
                                       param_init_type=param_init_type)

            else:
                output_size = ffn_hidden_size
                self.activation_func = None
                # Project to ffn_hidden_size
                self.mapping = _Linear(in_channels=input_size,
                                       out_channels=output_size,
                                       activation=hidden_act,
                                       transpose_b=False,
                                       expert_num=expert_num,
                                       expert_group_size=expert_group_size,
                                       outer_batch=dp,
                                       param_init_type=param_init_type)

            # Project back to hidden_size
            self.projection = _Linear(in_channels=output_size,
                                      out_channels=input_size,
                                      transpose_b=False,
                                      expert_num=expert_num,
                                      expert_group_size=expert_group_size,
                                      outer_batch=dp,
                                      param_init_type=param_init_type)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = nn.Dropout(1 - dropout_rate)
            self.dropout_3d = nn.Dropout(1 - dropout_rate)
            self.dropout_4d = nn.Dropout(1 - dropout_rate)
            self.cast = P.Cast()
        else:
            _check_config(parallel_config)
            mp = parallel_config.model_parallel
            if expert_num > 1:
                ep = parallel_config.expert_parallel
            else:
                ep = 1
            # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
            dp = parallel_config.data_parallel // ep
            if ffn_hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'ffn_hidden_size' must be a multiple of the"
                                 "num of model parallel, but got the ffn_hidden_size is {} and the num of model "
                                 "parallel is {}.".format(ffn_hidden_size, mp))
            if hidden_size % mp != 0:
                raise ValueError("For 'FeedForward', the class variable 'hidden_size' must be a multiple of the num of "
                                 "model parallel, but got the hidden_size is {} and the num of model parallel is {}."
                                 .format(hidden_size, mp))
            if dropout_rate < 0 or dropout_rate >= 1:
                raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                                 "but got the value : {}.".format(dropout_rate))

            self.mul = P.Mul()#.shard(((1, 1,), (1, 1,)))
            input_size = hidden_size
            hidden_act = "swiglu"       #'swiglu'
            if hidden_act == "swiglu":
                def Swiglu(x):
                    x = ops.chunk(x, 2, axis=-1)
                    return ops.silu(x[0]) * x[1]
                    # return self.mul(ops.silu(x[0]), x[1])

                # new pcmind go to this stage
                output_size = ffn_hidden_size
                self.activation_func = Swiglu
                # Project to ffn_hidden_size
                self.mapping = _Linear(in_channels=input_size,
                                       out_channels=int(output_size * 2),
                                       transpose_b=False,
                                       expert_num=expert_num,
                                       expert_group_size=expert_group_size,
                                       outer_batch=dp,
                                       param_init_type=param_init_type)

            else:
                output_size = ffn_hidden_size
                self.activation_func = None
                # Project to ffn_hidden_size
                self.mapping = _Linear(in_channels=input_size,
                                       out_channels=output_size,
                                       activation=hidden_act,
                                       transpose_b=False,
                                       expert_num=expert_num,
                                       expert_group_size=expert_group_size,
                                       outer_batch=dp,
                                       param_init_type=param_init_type)

            if expert_num > 1:
                self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                                   strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)),
                                   strategy_activation=((dp, ep, 1, mp),))
            else:
                self.mapping.shard(strategy_matmul=((dp, 1), (1, mp)),
                                   strategy_bias=((dp, mp), (mp,)),
                                   strategy_activation=((dp, mp),))
            # Project back to hidden_size
            self.projection = _Linear(in_channels=output_size,
                                      out_channels=input_size,
                                      transpose_b=False,
                                      expert_num=expert_num,
                                      expert_group_size=expert_group_size,
                                      outer_batch=dp,
                                      param_init_type=param_init_type)
            if expert_num > 1:
                self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                                      strategy_bias=((dp, ep, 1, 1), (1, ep, 1, 1)))
            else:
                self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)),
                                      strategy_bias=((dp, 1), (1,)))
            self.projection.bias.parallel_optimizer = False
            self.dropout = nn.Dropout(1 - dropout_rate)
            self.dropout.dropout.shard(((dp, 1),))
            self.dropout_3d = nn.Dropout(1 - dropout_rate)
            self.dropout_3d.dropout.shard(((dp, 1, 1),))
            self.dropout_4d = nn.Dropout(1 - dropout_rate)
            self.dropout_4d.dropout.shard(((dp, ep, 1, 1),))
            self.cast = P.Cast()

    def construct(self, x):
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, mstype.float16)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.mapping(x)
        if not self.activation_func is None:
            hidden = self.activation_func(hidden)
        output = self.projection(hidden)
        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        elif len(F.shape(output)) == 2:
            output = self.dropout(output)
        else:
            output = self.dropout_4d(output)
        return output


def _get_lambda_func(total_layer=None):
    r"""
    A wrapper function of specifying pipeline stage and gradient aggregation fusion. If the total layer
    is not None, for example, set in the transformer model, the pipeline stage setting function will be
    `(layer_id + 0) // (total_layers / parallel_config.pipeline_stage)` for the encoder and,
    `(layer_id + offset) //
    (total_layers / parallel_config.pipeline_stage)` for the decoder, where `offset` is the layers in the encoder.
    """

    def _set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
        r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs an offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
        """
        # override the layers
        if total_layer:
            layers = total_layer
        # Used for the pipeline's stages setting
        if layers < parallel_config.pipeline_stage:
            raise ValueError(f"layers {layers} must be larger than pipeline stage {parallel_config.pipeline_stage}")

        pp_dis = max(layers // parallel_config.pipeline_stage, 1)
        # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
        pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
        network.pipeline_stage = pp_id

        # Used for optimizer's fusion tag
        dis = max(layers // parallel_config.gradient_aggregation_group, 1)
        network.set_comm_fusion((layer_id + offset) // dis + 1)
        # Used for enabling recomputation of the block
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute:
                paralel_op_comm_compute = parallel_config.recompute.parallel_optimizer_comm_recompute
                network.recompute(parallel_optimizer_comm_recompute=paralel_op_comm_compute,
                                  mp_comm_recompute=parallel_config.recompute.mp_comm_recompute,
                                  recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    return _set_parallel_configure_for_layer


class RoPE(nn.Cell):
    """
    定义RoPE位置Embedding
    """
    sin_matrix = None
    cos_matrix = None

    def __init__(self, num_heads, hidden_size, seq_length, parallel_config, compute_type):
        super().__init__(auto_prefix=False)
        self.dtype = np.float64
        self.is_first_iteration = True
        self.max_seq_len = seq_length
        self.output_dim = hidden_size // num_heads
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.mul = P.Mul().shard(((dp, mp, 1, 1), (1, 1, 1, 1)))
        self.add = P.Add().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.split = P.Split(axis=-1, output_num=2).shard(((dp, mp, 1, 1),))
        self.concat = P.Concat(axis=-1).shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.neg = P.Neg().shard(((dp, mp, 1, 1),))
        self.gather = P.Gather()

        # self.split = P.Split(axis=-1, output_num=2).shard(((dp, mp, 1, 1, 1),))
        # self.concat = P.Concat(axis=-1).shard(((dp, mp, 1, 1, 1), (dp, mp, 1, 1, 1)))
        # self.neg = P.Neg().shard(((dp, mp, 1, 1, 1),))

        if self.cos_matrix is None:
            position_ids = np.arange(0, self.max_seq_len, dtype=self.dtype)[None]
            indices = np.arange(0, self.output_dim // 2., dtype=self.dtype)
            indices = 1. / np.power(10000.0, 2 * indices / self.output_dim, dtype=self.dtype)
            embeddings = np.einsum('bn,d->bnd', position_ids, indices, dtype=self.dtype)
            # embeddings = np.repeat(embeddings, 2, axis=-1).reshape(1, 1, self.max_seq_len, self.output_dim)
            embeddings = np.concatenate((embeddings, embeddings), axis=-1).reshape(1, 1, self.max_seq_len, self.output_dim)
            RoPE.sin_matrix = Tensor(np.sin(embeddings, dtype=self.dtype), dtype=compute_type)
            RoPE.cos_matrix = Tensor(np.cos(embeddings, dtype=self.dtype), dtype=compute_type)

    def rotate_half(self, x, input_shape):
        # shape_5d = (input_shape[0], input_shape[1], input_shape[2], -1, 2)
        # x = self.reshape(x, shape_5d)
        x0, x1 = self.split(x)
        rotate = self.concat([self.neg(x1), x0])
        # rotate = self.reshape(rotate, input_shape)
        return rotate

    def construct(self, inputs, batch_valid_length):
        input_shape = self.shape(inputs)

        if self.is_first_iteration:
            seq_len = input_shape[2]
            sin_matrix = self.slice(self.sin_matrix, (0, 0, 0, 0), (1, 1, seq_len, input_shape[3]), (1, 1, 1, 1))
            cos_matrix = self.slice(self.cos_matrix, (0, 0, 0, 0), (1, 1, seq_len, input_shape[3]), (1, 1, 1, 1))
        else:
            sin_matrix = self.gather(self.sin_matrix, batch_valid_length, 2)
            cos_matrix = self.gather(self.cos_matrix, batch_valid_length, 2)

        output1 = self.mul(inputs, cos_matrix)
        output2 = self.mul(self.rotate_half(inputs, input_shape), sin_matrix)
        output = self.add(output1, output2)
        return output


class FlashAttention(nn.Cell):
    def __init__(self,
                 head_dim,
                 dropout_rate=0.0,
                 prev_block_num=65536,
                 next_block_num=65536,
                 tiling_stgy_name="xunfei",
                 dp=1,
                 mp=1,
                 param_init_type=mstype.float16,
                 ):
        super(FlashAttention, self).__init__()
        self.flash_attention = FlashAttentionPrimitive(
            prev_block_num=prev_block_num,
            next_block_num=next_block_num,
            high_precision=False
        )
        self.flash_attention.add_prim_attr("primitive_target", "Ascend")
        print("> FA dtype: {}".format(param_init_type))
        self.scale_factor = Tensor([1. / math.sqrt(math.sqrt(head_dim))], dtype=param_init_type)
        self.dim_mask = Tensor([1 for _ in range(head_dim)], dtype=mstype.int8)
        self.scale_mul = ops.Mul().shard(((dp, mp, 1, 1), (1,)))

        self.keep_prob = Tensor(1 - dropout_rate, dtype=param_init_type)
        self.fill_v2 = ops.FillV2().shard(((dp, mp, 1, 1), ()))
        self.tensor_one = Tensor(1.0, param_init_type)
        self.drop_gen_mask = ops.DropoutGenMask()
        self.do_dropout = ops.DropoutDoMask().shard(((dp, mp, 1, 1),))
        self.depend = ops.Depend()

    def shard(self, in_strategy=None, out_strategy=None):
        """Distributed configuration of FlashAttention
        :param in_strategy: Describe the split strategy of operator input. Default: None.
        :param out_strategy: Describe the split strategy of operator output, it is only for certain operators,
                                  such as MatMul. Default: None.
        :return:
        """
        if in_strategy is not None:
            shard_stgy = list(in_strategy)
            shard_stgy.insert(3, (1,))  # dim_mask
            shard_stgy = tuple(shard_stgy)
        else:
            # default: dp=1, mp=1, construct inputs only contain q, k, v
            shard_stgy = (
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1,),  # dim_mask
            )
        self.flash_attention.shard(shard_stgy)

    def construct(self, q, k, v, attn_mask=None, alibi_mask=None):
        """FlashAttention forward
        :param q:           [bsz, head_num, seq_len, head_dim]
        :param k:           [bsz, head_num, seq_len, head_dim]
        :param v:           [bsz, head_num, seq_len, head_dim]
        :param attn_mask:   [1 or bsz, seq_len, seq_len], if not None
        :param alibi_mask: [bsz, head_num, 1, seq_len], if not None
        :return: o          [bsz, head_num, seq_len, head_dim]
        """
        q = self.scale_mul(q, self.scale_factor)
        k = self.scale_mul(k, self.scale_factor)
        bsz, head_num, seq_len, _ = q.shape

        drop_mask_bits = self.drop_gen_mask((bsz, head_num, seq_len, seq_len), self.keep_prob)
        ones = self.fill_v2((bsz, head_num, seq_len, seq_len), self.tensor_one)
        ones = self.depend(ones, q)
        # ones = self.depend(ones, k)
        drop_mask = self.do_dropout(ones, drop_mask_bits, self.keep_prob)

        o, _, _ = self.flash_attention(q, k, v, self.dim_mask, attn_mask, drop_mask, None)
        # o, _, _ = self.flash_attention(q, k, v, attn_mask, drop_mask, None)
        return o


class PipeMultiHeadAttention(Cell):
    @_LogActionOnce(logger=logger, key='MultiHeadAttention',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                src_seq_length=Validator.check_positive_int,
                                tgt_seq_length=Validator.check_positive_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "MultiHeadAttention"),
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "MultiHeadAttention"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "MultiHeadAttention"),
                                parallel_config=_valid_type_checks([OpParallelConfig],
                                                                   "MultiHeadAttention"),
                                use_past=Validator.check_bool)
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config,
                 use_rope=False,
                 use_flash_attention=False):

        super(PipeMultiHeadAttention, self).__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)

        _check_config(parallel_config)
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # for simplicity,
        assert not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation())

        if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
        if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
            raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                             "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
        if hidden_size % num_heads != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                             "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                             .format(hidden_size, num_heads))
        if num_heads % parallel_config.model_parallel != 0:
            raise ValueError("For 'MultiHeadAttention', the class variable 'num_heads' must be a multiple of "
                             "'parallel_config.model_parallel', but got the num_heads is {} "
                             "and the parallel_config.model_parallel  is {}."
                             .format(num_heads, parallel_config.model_parallel))
        self.is_first_iteration = True
        # Output layer
        self.projection = _Linear(in_channels=hidden_size,
                                  out_channels=hidden_size,
                                  transpose_b=False,
                                  compute_dtype=compute_dtype,
                                  param_init_type=param_init_type)
        self.projection.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                              strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                               (parallel_config.model_parallel, 1)))
        self.projection.bias.parallel_optimizer = False
        self.transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))
        self.transpose_back = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = num_heads
        # embedding size per head
        self.size_per_head = hidden_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=softmax_compute_type)
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.real_div = P.RealDiv().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
        self.sub = P.Sub().shard(
            ((1,), (parallel_config.data_parallel, 1, 1, 1)))
        self.mul = P.Mul().shard(
            ((parallel_config.data_parallel, 1, 1, 1), (1,)))
        self.add = P.Add().shard(
            ((parallel_config.data_parallel, 1, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        # Normalize factor for attention, sqrt(dk) as widely used
        self.scale_factor = Tensor(math.sqrt(math.sqrt(self.size_per_head)))
        self.use_past = use_past
        self.dropout = nn.Dropout(1 - hidden_dropout_rate)
        self.dropout.dropout.shard(((parallel_config.data_parallel, 1),))
        self.prob_dropout = nn.Dropout(1 - attention_dropout_rate)
        self.prob_dropout.dropout.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d.softmax.shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
        self.expand_dims = P.ExpandDims().shard(((parallel_config.data_parallel, 1, 1),))

        # Query
        self.dense1 = _Linear(hidden_size,
                              hidden_size,
                              compute_dtype=compute_dtype,
                              param_init_type=param_init_type)
        self.dense1.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        # Key
        self.dense2 = _Linear(hidden_size,
                              hidden_size,
                              compute_dtype=compute_dtype,
                              param_init_type=param_init_type)
        self.dense2.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))

        # Value
        self.dense3 = _Linear(hidden_size,
                              hidden_size,
                              compute_dtype=compute_dtype,
                              param_init_type=param_init_type)
        self.dense3.shard(strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
                          strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                                         (parallel_config.model_parallel,)))
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_type
        if self.use_past:
            # operators used for state reuse
            seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
            self.range = Tensor(np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
            self.seq_length = src_seq_length
            self.attention_mask = Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
            self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.sub1 = P.Sub().shard(((1,), ()))
            self.tile = P.Tile().shard(((1, 1, 1, 1),))
            self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
            self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

        self.use_flash_attention = use_flash_attention
        if self.use_flash_attention:
            print("@@ Using FA")
            self.head_dim = hidden_size // num_heads
            dp = parallel_config.data_parallel
            mp = parallel_config.model_parallel
            self.flash_attention = FlashAttention(head_dim=self.head_dim,
                                                  dropout_rate=attention_dropout_rate,
                                                  prev_block_num=65536,
                                                  next_block_num=0,
                                                  dp=parallel_config.data_parallel,
                                                  mp=parallel_config.model_parallel,
                                                  param_init_type=param_init_type)

            q_shard_stgy = (dp, mp, 1, 1)
            k_shard_stgy = (dp, mp, 1, 1)
            v_shard_stgy = (dp, mp, 1, 1)
            attn_mask_shard_stgy = (dp, 1, 1)
            dropout_shard_stgy = (dp, mp, 1, 1)
            in_stgy = (q_shard_stgy, k_shard_stgy, v_shard_stgy, attn_mask_shard_stgy, dropout_shard_stgy)
            self.flash_attention.shard(in_stgy)
            self.pad_tensor = Tensor(
                np.zeros((self.batch_size, self.n_head, self.src_seq_length, 16 - self.head_dim % 16)),
                dtype=compute_dtype)
            self.zeroslike = P.ZerosLike().shard(((dp, mp, 1, 1),))
            self.concat = P.Concat(axis=-1).shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            # self.padding = P.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 16 - self.head_dim % 16))).shard(((dp, mp, 1, 1),))
            self.slice = P.Slice().shard(((dp, mp, 1, 1),))
            self.output_shape = (self.batch_size, self.n_head, self.src_seq_length, self.head_dim)

        self.use_rope = use_rope
        self.rope = None
        if self.use_rope:
            self.rope = RoPE(num_heads, hidden_size, tgt_seq_length, parallel_config, compute_dtype)

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        self._check_inputs(query_tensor, key_tensor, value_tensor, attention_mask, key_past,
                           value_past, batch_valid_length)
        ori_shape = F.shape(query_tensor)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor,
                                                                            attention_mask)
        ori_dtype = F.dtype(query_tensor)
        query_tensor = F.cast(query_tensor, self.dtype)
        key_tensor = F.cast(key_tensor, self.dtype)
        value_tensor = F.cast(value_tensor, self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = self.transpose(
            F.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            (0, 2, 1, 3))

        if self.use_rope:
            query = self.rope(query, batch_valid_length)
            key = self.rope(key, batch_valid_length)

        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and len(F.shape(attention_mask)) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                current_index = F.reshape(batch_valid_length, (-1, 1, 1))
                valid_length_vector = F.cast(self.tensor_le(self.range, current_index), self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(key, self.expand_dims(valid_length_vector, 3))
                value_present = self.mul1(value, self.expand_dims(valid_length_vector, 3))
            else:
                current_index = F.reshape(batch_valid_length, (-1, 1, 1))
                current_mask = F.cast(self.equal(self.range, current_index), self.dtype)
                current_key = self.mul1(key, self.expand_dims(current_mask, 3))
                current_value = self.mul1(value, self.expand_dims(current_mask, 3))
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        if self.use_flash_attention:
            attention_mask = 1.0 - attention_mask.squeeze(1)
            attention = self.flash_attention(self.concat((query, self.zeroslike(self.pad_tensor))),
                                             self.concat((key, self.zeroslike(self.pad_tensor))),
                                             self.concat((value, self.zeroslike(self.pad_tensor))),
                                             attention_mask)
            # attention = self.flash_attention(self.padding(query),
            #                                  self.padding(key),
            #                                  self.padding(value),
            #                                  attention_mask)
            attention = self.slice(attention, (0, 0, 0, 0), self.output_shape)
            attention = self._merge_heads(attention)
        else:
            key = self.transpose_back(key, (0, 1, 3, 2))
            attention = self._attn(query, key, value, attention_mask)

        # Output
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        output = F.cast(output, ori_dtype)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        r"""Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if len(F.shape(query)) == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return F.shape(query)[0] // self.src_seq_length
        return F.shape(query)[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _check_inputs(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                      value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(query_tensor), "query_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(key_tensor), "key_tensor", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(F.dtype(value_tensor), "value_tensor", [mstype.float32, mstype.float16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(F.dtype(attention_mask), "attention_mask", [mstype.float32, mstype.float16],
                               self.cls_name)

        key_is_tensor = isinstance(key_past, Tensor)
        value_is_tensor = isinstance(value_past, Tensor)
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        key_is_default = key_past is None
        value_is_default = value_past is None
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "key_past", self.cls_name, None, key_is_tensor,
                                    key_is_default)
        _check_past_none_input_none(self.use_past, "value_past", self.cls_name, None, value_is_tensor,
                                    value_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(F.dtype(key_past), "key_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(value_past), "value_past", [mstype.float16], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor, attention_mask):
        """convert a nd tensor to a 2d tensor"""
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))

        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """

        if self._is_ascend and self.softmax_dtype == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        attention_scores = P.Cast()(score, self.softmax_dtype)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum(F.cast(self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                                (F.shape(query)[0], 1, 1,
                                                                                 self.seq_length),
                                                                                (1, 1, 1, 1)),
                                                                     0), mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(F.cast(current_index, mstype.int32), 1)
                index = F.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = F.cast(self.tensor_le(self.range, index), mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(
                P.Cast()(F.tuple_to_array((1.0,)), P.DType()(attention_scores)),
                P.Cast()(attention_mask, P.DType()(attention_scores)))

            adder = self.mul(multiplu_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = P.Cast()(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class PipeTransformerEncoderLayer(Cell):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config,
                 use_rope=False,
                 use_flash_attention=False):
        super(PipeTransformerEncoderLayer, self).__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            print("@@ ParallelMode.AUTO_PARALLEL and _is_sharding_propagation")
            _check_config(parallel_config)
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                        .format(ffn_hidden_size, parallel_config.model_parallel))
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.transpose = P.Transpose().shard(((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = PipeMultiHeadAttention(batch_size=batch_size,
                                                    src_seq_length=seq_length,
                                                    tgt_seq_length=seq_length,
                                                    hidden_size=hidden_size,
                                                    num_heads=num_heads,
                                                    hidden_dropout_rate=hidden_dropout_rate,
                                                    attention_dropout_rate=attention_dropout_rate,
                                                    softmax_compute_type=softmax_compute_type,
                                                    param_init_type=param_init_type,
                                                    use_past=use_past,
                                                    parallel_config=attention_parallel_config,
                                                    use_rope=use_rope,
                                                    use_flash_attention=use_flash_attention)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = param_init_type #mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            print("@@ Not ParallelMode.AUTO_PARALLEL or _is_sharding_propagation")
            if num_heads % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'num_heads' must be divisibled by the "
                    "'parallel_config.model_parallel', but got the num_heads is {} and "
                    "parallel_config.model_parallel is {}.".format(num_heads, parallel_config.model_parallel))
            if hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(hidden_size, parallel_config.model_parallel))
            if ffn_hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'ffn_hidden_size' must be divisibled "
                    "by the 'parallel_config.model_parallel', but got the ffn_hidden_size is {} "
                    "and parallel_config. model_parallel is {}."
                        .format(ffn_hidden_size, parallel_config.model_parallel))
            _check_moe_config(moe_config, parallel_config)
            self.use_moe = (moe_config.expert_num > 1)
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm1.shard(((parallel_config.data_parallel, 1),))
            self.layernorm2 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2.shard(((parallel_config.data_parallel, 1),))
            self.transpose = P.Transpose().shard(
                ((parallel_config.data_parallel, 1, parallel_config.model_parallel, 1),))

            attention_parallel_config = parallel_config.dpmp if self.use_moe else parallel_config
            self.attention = PipeMultiHeadAttention(batch_size=batch_size,
                                                    src_seq_length=seq_length,
                                                    tgt_seq_length=seq_length,
                                                    hidden_size=hidden_size,
                                                    num_heads=num_heads,
                                                    hidden_dropout_rate=hidden_dropout_rate,
                                                    attention_dropout_rate=attention_dropout_rate,
                                                    softmax_compute_type=softmax_compute_type,
                                                    param_init_type=param_init_type,
                                                    use_past=use_past,
                                                    parallel_config=attention_parallel_config,
                                                    use_rope=use_rope,
                                                    use_flash_attention=use_flash_attention)
            if self.use_moe:
                self.output = MoE(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act,
                                  moe_config=moe_config,
                                  parallel_config=parallel_config)
            else:
                # Feed Forward Network, FFN
                self.output = FeedForward(hidden_size=hidden_size,
                                          dropout_rate=hidden_dropout_rate,
                                          ffn_hidden_size=ffn_hidden_size,
                                          param_init_type=param_init_type,
                                          hidden_act=hidden_act,
                                          parallel_config=parallel_config)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add().shard(((parallel_config.data_parallel, 1), (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
            self.dtype = param_init_type #mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # key_present = self.transpose(key_present, (0, 1, 3, 2))
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = F.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = F.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)

        if self.use_moe:
            return output, layer_present, aux_loss
        return output, layer_present

    def _check_input(self, x, input_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        if input_mask is not None:
            _check_input_dtype(F.dtype(input_mask), "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(F.dtype(init_reset), "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(F.dtype(batch_valid_length), "batch_valid_length", [mstype.int32], self.cls_name)
        return True


default_transformer_config = TransformerOpParallelConfig()


class PipeTransformerEncoder(Cell):
    @_LogActionOnce(logger=logger, key='TransformerEncoder',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(batch_size=Validator.check_positive_int,
                                hidden_size=Validator.check_positive_int,
                                num_heads=Validator.check_positive_int,
                                ffn_hidden_size=Validator.check_positive_int,
                                seq_length=Validator.check_positive_int,
                                num_layers=Validator.check_positive_int,
                                offset=Validator.check_non_negative_int,
                                attention_dropout_rate=Validator.check_non_negative_float,
                                hidden_dropout_rate=Validator.check_non_negative_float,
                                post_layernorm_residual=Validator.check_bool,
                                layernorm_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                           "TransformerEncoder"),
                                softmax_compute_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                         "TransformerEncoder"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "TransformerEncoder"),
                                parallel_config=_valid_type_checks([TransformerOpParallelConfig],
                                                                   "TransformerEncoder"),
                                use_past=Validator.check_bool,
                                use_rope=Validator.check_bool)
    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0,
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_transformer_config,
                 use_rope=False,
                 use_flash_attention=False,
                 pipeline_config=None):
        super(PipeTransformerEncoder, self).__init__()
        _check_config(parallel_config)
        _check_moe_config(moe_config, parallel_config)
        self.use_moe = (moe_config.expert_num > 1)
        config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.add = P.Add()
            self.aux_loss = Tensor(0.0, mstype.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = PipeTransformerEncoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    seq_length=seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    num_heads=num_heads,
                    hidden_act=hidden_act,
                    post_layernorm_residual=post_layernorm_residual,
                    param_init_type=param_init_type,
                    use_past=use_past,
                    moe_config=moe_config,
                    parallel_config=config_to_layer,
                    use_rope=use_rope,
                    use_flash_attention=use_flash_attention)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            self.add = P.Add().shard(((), ()))
            self.aux_loss = Tensor(0.0, mstype.float32)
            logger.warning("For parallel mode, sharding propagation is recommended, you can use it by setting "
                           "'set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, "
                           "search_mode=\"sharding_propagation\")' and "
                           "'set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)'")
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = PipeTransformerEncoderLayer(
                    hidden_size=hidden_size,
                    batch_size=batch_size,
                    ffn_hidden_size=ffn_hidden_size,
                    seq_length=seq_length,
                    attention_dropout_rate=attention_dropout_rate,
                    hidden_dropout_rate=hidden_dropout_rate,
                    layernorm_compute_type=layernorm_compute_type,
                    softmax_compute_type=softmax_compute_type,
                    num_heads=num_heads,
                    hidden_act=hidden_act,
                    post_layernorm_residual=post_layernorm_residual,
                    param_init_type=param_init_type,
                    use_past=use_past,
                    moe_config=moe_config,
                    parallel_config=config_to_layer,
                    use_rope=use_rope,
                    use_flash_attention=use_flash_attention)
                # If the user doesn't pass the fusion function, use the default one
                if not lambda_func:
                    lambda_func = _get_lambda_func()

                lambda_func(block, layer_id=i, layers=num_layers,
                            offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None):
        present_layer = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, present, aux_loss = self.blocks[i](hidden_states,
                                                                  attention_mask,
                                                                  init_reset,
                                                                  batch_valid_length)
                present_layer = present_layer + (present,)
                accum_loss = self.add(accum_loss, aux_loss)
            return hidden_states, present_layer, accum_loss

        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer
