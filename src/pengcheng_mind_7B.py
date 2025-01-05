# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PengChengMind model"""
import os
import copy
import numpy as np
import mindspore.nn as nn
import mindspore.numpy as np2
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.nn.transformer.transformer import VocabEmbedding, TransformerEncoder, TransformerEncoderLayer, \
    AttentionMask
from mindspore.nn.transformer import MoEConfig
from mindspore.nn.transformer.layers import _LayerNorm, _Dropout

from mindspore._extends import cell_attr_register
from src.pengcheng_mind_config import OperatorType

# from src.pengcheng_mind_pipeline_layer import PipeTransformerEncoder, PipeTransformerEncoderLayer
from src.pengcheng_mind_pipeline_layer_7B import PipeTransformerEncoder, PipeTransformerEncoderLayer

class EmbeddingLayer(nn.Cell):
    r"""Embedding layer of the PengChengMind Model"""
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        # Only for the pipeline mode, the embedding needs to be row sliced.
        ###################taoht###############################
        # if config.run_type == 'predict':
        #     dp = 1
        #     mp = 1
        # else:
        dp = config.parallel_config.embedding_dp_mp_config.data_parallel
        mp = config.parallel_config.embedding_dp_mp_config.model_parallel
        self.word_embedding = VocabEmbedding(vocab_size=config.vocab_size,
                                             embedding_size=config.hidden_size,
                                             param_init=initializer("normal", [config.vocab_size, config.hidden_size],
                                                                    dtype=mstype.float32),
                                             parallel_config=config.parallel_config.embedding_dp_mp_config)
        self.word_embedding.gather.shard(((mp, 1), (dp, 1)))
        self.word_embedding.embedding_table.parallel_optimizer = True
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        copied_parallel_config.vocab_emb_dp = True
        # if config.run_type == 'predict':
        #     copied_parallel_config.vocab_emb_dp = False
        self.position_embedding = None
        self.use_rope = config.use_rope
        if not self.use_rope:
            self.position_embedding = VocabEmbedding(vocab_size=config.seq_length,
                                                     embedding_size=config.hidden_size,
                                                     param_init=initializer("normal",
                                                                            [config.seq_length, config.hidden_size],
                                                                            dtype=mstype.float32),
                                                     parallel_config=copied_parallel_config.embedding_dp_mp_config)
        # self.split = P.Split(1, 2).shard(
        #     ((config.parallel_config.data_parallel, 1, 1),)
        # )
        self.add = P.Add().shard(
            ((config.parallel_config.data_parallel, 1, 1), (config.parallel_config.data_parallel, 1, 1)))
        self.dropout = _Dropout(1 - config.dropout_rate)
        self.dropout.shard(
            ((config.parallel_config.data_parallel, 1, 1),)
        )
        self.is_first_iteration = True
        self.use_past = config.use_past
        self.batch_size = config.batch_size

    def construct(self, input_ids, input_position, init_reset, batch_valid_length):
        embed, word_table = self.word_embedding(input_ids)
        if not self.use_rope:
            if self.use_past and not self.is_first_iteration:
                _, seq_length = F.shape(input_ids)
                input_position = batch_valid_length.view(self.batch_size, seq_length)
            position_embedding, _ = self.position_embedding(input_position)
            embed = self.add(embed, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table

    def get_word_embedding_weight(self):
        return self.word_embedding.embedding_table

class PengChengMindHead(Cell):
    """
    Head to get the logits of each token in the vocab
    Args:
        config(): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16,
                 parallel_config=None):
        super(PengChengMindHead, self).__init__()
        if parallel_config.vocab_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((parallel_config.data_parallel, 1), (
                parallel_config.model_parallel, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        # output logits over vocabulary [bs*seq_length, vocab_size]
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


# def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
#     r"""
#         Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.
#
#
#         Args:
#             network(Cell) - Represents the transformer block
#             layer_id(int) - Means the layer index for the current module, counts from zero.
#             offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
#             layers(int) - The total layers used for the model.
#     """
#     # Used for the pipeline's stages setting
#     # As the final layer is not included here, so we need to manually add here.
#     # original:  if set two stages, layers on two stages will be [15, 16+1]
#     # with 1 added, the layers on two stages will be [16, 15 +1]
#     pp_dis = max(int((layers + 1)/ parallel_config.pipeline_stage), 1)
#     # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
#     pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
#     network.pipeline_stage = pp_id
#     print(f"pipeline stage id is {pp_id}", flush=True)
#
#     # Used for optimizer's fusion tag
#     dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
#     if parallel_config.pipeline_stage > 1:
#         # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
#         network.set_comm_fusion(2)
#     else:
#         network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
#     # Used for enabling recomputation of the block
#     if isinstance(parallel_config.recompute, bool):
#         if parallel_config.recompute:
#             network.recompute()
#     else:
#         if parallel_config.recompute.recompute:
#             network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

def set_recompute_block(block, layer_config, use_flash_attention):
    if OperatorType.LINEAR_INPUT in layer_config:
        if layer_config[OperatorType.LINEAR_INPUT] == 1:
            block.attention.projection.matmul.recompute(mode=False)
            block.attention.projection.matmul.add_prim_attr("recompute_comm_op", False)
            block.attention.projection.bias_add.recompute(mode=False)
            #block.attention.dropout.dropout.recompute(mode=False)

    if OperatorType.ATTENTION_QKV in layer_config:
        atten_recomp = layer_config[OperatorType.ATTENTION_QKV]
        if atten_recomp > 0:
            block.attention.dense1.matmul.recompute(mode=False)
            block.attention.dense1.bias_add.recompute(mode=False)
            block.attention.transpose_q.recompute(mode=False)
            #block.attention.real_div_q.recompute(mode=False)
            atten_recomp -= 1
        if atten_recomp > 0:
            block.attention.dense2.matmul.recompute(mode=False)
            block.attention.dense2.bias_add.recompute(mode=False)
            block.attention.transpose_k.recompute(mode=False)
            #block.attention.transpose_back.recompute(mode=False)
            #block.attention.real_div_k.recompute(mode=False)
            atten_recomp -= 1
        if atten_recomp > 0:
            block.attention.dense3.matmul.recompute(mode=False)
            block.attention.dense3.bias_add.recompute(mode=False)
            block.attention.transpose_v.recompute(mode=False)
            atten_recomp -= 1

    if OperatorType.LINEAR_MAPPING in layer_config:
        if layer_config[OperatorType.LINEAR_MAPPING] == 1:
            block.output.mapping.matmul.recompute(mode=False)
            block.output.mapping.bias_add.recompute(mode=False)

    if not use_flash_attention:
        if OperatorType.ATTENTION_SOFTMAX in layer_config:
            if layer_config[OperatorType.ATTENTION_SOFTMAX] == 1:
                block.attention.softmax_3d.softmax.recompute(mode=False)
                block.attention.softmax_cast.recompute(mode=False)
                block.attention.softmax_reshape.recompute(mode=False)
                block.attention.add_mask.recompute(mode=False)

        if OperatorType.ATTENTION_ATTEN in layer_config:
            if layer_config[OperatorType.ATTENTION_ATTEN] == 1:
                block.attention.batch_matmul1.recompute(mode=False)

        if OperatorType.ATTENTION_BMM in layer_config:
            if layer_config[OperatorType.ATTENTION_BMM] == 1:
                block.attention.batch_matmul.recompute(mode=False)

    #if OperatorType.ATTENTION_X in layer_config:
    #    if layer_config[OperatorType.ATTENTION_X] == 1:
    #        block.layernorm1.layer_norm.add_prim_attr("recompute_comm_op", False)

    #if OperatorType.LINEAR_X in layer_config:
    #    if layer_config[OperatorType.LINEAR_X] == 1:
    #        block.layernorm2.layer_norm.add_prim_attr("recompute_comm_op", False)

    if OperatorType.LINEAR_GELU in layer_config:
        if layer_config[OperatorType.LINEAR_GELU] == 1:
            block.output.mapping.activation.fast_gelu.recompute(mode=False)

def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.
    """
    # Used for the pipeline's stages setting
    # As the final layer is not included here, so we need to manually add here.
    # original:  if set two stages, layers on two stages will be [15, 16+1]
    # with 1 added, the layers on two stages will be [16, 15 +1]
    pp_dis = max((layers + 1) / parallel_config.pipeline_stage, 1)

    if layers == 32:
        # print(">>> 7B model pp setting! >>>")
        if parallel_config.pipeline_stage == 4:
            pp_id = [0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3]
            # pp_id = [0, 0, 0, 0, 0, 0, 0, 0,
            #          1, 1, 1, 1, 1, 1, 1, 1,
            #          2, 2, 2, 2, 2, 2, 2, 2,
            #          3, 3, 3, 3, 3, 3, 3, 3]
        elif parallel_config.pipeline_stage == 3:
            pp_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        elif parallel_config.pipeline_stage == 2:
            pp_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:  # for pp=1
            pp_id = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise ValueError(f"Unknown pipeline config for pp: {parallel_config.pipeline_stage} and layers: {layers}")

    network.pipeline_stage = pp_id[layer_id]
    print(f"Layer id is {layer_id}, pipeline stage id is {pp_id[layer_id]}", flush=True)
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        network.set_comm_fusion(layer_id + 1)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if parallel_config.recompute:
        network.recompute(recompute_slice_activation=True)
        # network.attention.projection.matmul.recompute(mode=False)
        # network.attention.projection.matmul.add_prim_attr("recompute_comm_op", False)


class PengChengMind_Model(Cell):
    r"""The base backbone of the PengChengMind model"""
    def __init__(self, config):
        super(PengChengMind_Model, self).__init__()
        self.is_pipeline = config.parallel_config.pipeline_stage > 1
        self.embedding = EmbeddingLayer(config)
        self.config = config
        self.layernorm = _LayerNorm((config.hidden_size,)).to_float(mstype.float32)
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(config.num_layers)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1
        # Configure the shard configure of the Embedding layer
        self.embedding.pipeline_stage = 0
        self.num_layers = config.num_layers
        if config.use_moe:
            moe_config = MoEConfig(expert_num=config.expert_num,
                                   num_experts_chosen=config.per_token_num_experts_chosen)
        else:
            moe_config = MoEConfig(expert_num=1)
        # The shard setting of Transformer is set within the class StackedTransformer
        self.blocks = PipeTransformerEncoder(
            num_layers=config.num_layers,
            batch_size=config.batch_size,
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_heads=config.num_heads,
            seq_length=config.seq_length,
            attention_dropout_rate=config.dropout_rate,
            hidden_dropout_rate=config.dropout_rate,
            lambda_func=set_parallel_configure_for_layer,
            hidden_act=config.hidden_act,
            param_init_type=config.param_init_type,
            use_past=config.use_past,
            parallel_config=config.parallel_config,
            moe_config=moe_config,
            softmax_compute_type=config.softmax_compute_fp32,
            use_rope=config.use_rope,
            use_flash_attention=config.use_flash_attention,
            pipeline_config=config.pipeline_config).blocks
        for block in self.blocks:
            block.attention.dense1.bias.parallel_optimizer = False
            block.attention.dense2.bias.parallel_optimizer = False
            block.attention.dense3.bias.parallel_optimizer = False
            block.output.mapping.bias.parallel_optimizer = False

        self.dtype = mstype.float16

        if config.load_ckpt_path:
            self.load_embedding_from_ckpt(config.load_ckpt_path)
        self.run_type = config.run_type

    def construct(self, input_ids,
                  input_position,
                  encoder_masks,
                  init_reset=True,
                  batch_valid_length=None):
        r"""forward pass of the model"""
        embed, word_table = self.embedding(input_ids, input_position, init_reset, batch_valid_length)
        hidden_state = P.Cast()(embed, self.dtype)
        # the input of the incremental prediction is 3d
        #########################taoht############################
        # if self.run_type != 'predict':
        #     hidden_state = self.reshape_to_2d(hidden_state)
        hidden_state = self.reshape_to_2d(hidden_state)  # ignored predict to speed up
        if self.blocks is not None:
            for i in range(self.num_layers):
                hidden_state, _ = self.blocks[i](hidden_state, encoder_masks, init_reset, batch_valid_length)
        encoder_output = self.layernorm(hidden_state)
        return encoder_output, word_table

    def reshape_to_2d(self, x):
        r"""reshape nd tensor to 2d, if n <= 2, keep original shape."""
        shape = F.shape(x)
        if len(shape) <= 2:
            return x
        x = F.reshape(x, (-1, shape[-1]))
        return x

    def load_embedding_from_ckpt(self, load_ckpt_path):
        r"""load the weights from the checkpoint"""
        def load_param(path):
            if os.path.exists(path):
                p_table = np.load(path)
                table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(f"{path} file not exits, "
                                 f"please check whether embedding file exit.")
            return table_param

        # three embedding needed to be loaded
        # Loading the embedding table from the ckpt path:
        position_embedding_path = os.path.join(load_ckpt_path, 'position_embedding.npy')
        word_embedding_path = os.path.join(load_ckpt_path, 'word_embedding.npy')
        self.embedding.word_embedding.embedding_table = Parameter(initializer(load_param(word_embedding_path),
                                                                              [self.config.vocab_size,
                                                                               self.config.hidden_size]),
                                                                  name='word_embedding_table', parallel_optimizer=False)
        self.embedding.position_embedding.embedding_table = Parameter(initializer(load_param(position_embedding_path),
                                                                                  [self.config.seq_length,
                                                                                   self.config.hidden_size]),
                                                                      name='position_embedding_table',
                                                                      parallel_optimizer=False)


class PengChengMindModel(nn.Cell):
    """
    The PengChengMind network consisting of two parts the backbone and the head
    Args:
        config(PengChengMindConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """

    def __init__(self, config):
        super(PengChengMindModel, self).__init__()
        # Network head to get logits over vocabulary
        copied_parallel_config = copy.deepcopy(config.parallel_config)
        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False
        self.head = PengChengMindHead(hidden_size=config.hidden_size,
                              parallel_config=copied_parallel_config)
        self.head.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.backbone = PengChengMind_Model(config)
        self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(self.head.pipeline_stage)
        # self.cast = P.Cast()

    def construct(self, input_ids, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        #######################taoht#############################NO
        # attention_mask = self.cast(attention_mask, mstype.float16)
        ##########################################################
        output_states, word_table = self.backbone(input_ids, input_position, attention_mask,
                                                  init_reset, batch_valid_length)
        logits = self.head(output_states, word_table)
        return logits


class PengChengMindWithLoss(Cell):
    """
    PengChengMind training loss for generation.
    Args:
        config(PengChengMindConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    @cell_attr_register
    def __init__(self, config, network, loss):
        super(PengChengMindWithLoss, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.pad_token = config.pad_token
        self.loss = loss

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num
        self.cast = P.Cast()

    def construct(self, input_ids, input_position=None, attention_mask=None):
        r"""Forward process of the pengcheng mind model"""
        # tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        # input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        # print("===================="*10)
        # P.Print()("enter input_ids", input_ids)
        # P.Print()("enter input_position", input_position)
        # print("====================" * 10)
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        #######################taoht##############################
        attention_mask = self.cast(attention_mask, mstype.float16)
        ##########################################################
        decoder_attention_masks = self.slice2(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                              (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.pad_token),
                            mstype.float32)

        logits = self.network(tokens,
                              input_position,
                              decoder_attention_masks)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1),
                            (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output


class PengChengMindLossWithPrompt(Cell):
    """
    PengChengMind training loss for generation.
    Args:
        config(PengChengMindConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss):
        super(PengChengMindLossWithPrompt, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = config.parallel_config.data_parallel
        self.network = network
        self.pad_token = config.pad_token
        self.loss = loss

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(config.seq_length)
        self.equal = P.Equal()
        self.expand = P.ExpandDims()

    def construct(self, input_ids, prompt_ids):
        r"""Forward process of the pengcheng mind model"""
        tokens = input_ids
        input_mask = F.cast(self.not_equal(tokens, self.pad_token), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.len))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))

        input_mask_a = F.cast(self.equal(prompt_ids, self.pad_token), mstype.float32)
        attention_mask = self.get_attention_mask(input_mask)

        logits = self.network(tokens, input_position, attention_mask)

        log_probs = self.log_softmax(logits)
        input_mask_b = input_mask * input_mask_a
        return log_probs, input_mask_b

class PengChengMindLossWith_notPrompt(Cell):
    """
    PengChengMind training loss for generation.
    Args:
        config(PengChengMindConfig)
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, pad_token=0, seq_length=4096):
        super(PengChengMindLossWith_notPrompt, self).__init__(auto_prefix=False)
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        dp = 1
        self.network = network
        self.pad_token = pad_token
        self.loss = loss
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.micro_batch_step = 1
        if config.parallel_config.pipeline_stage > 1:
            self.micro_batch_step = config.parallel_config.micro_batch_num
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.equal = P.Equal()
        self.expand = P.ExpandDims()

    def construct(self, input_ids, mask_ids_input=None):
        r"""Forward process of the pengcheng mind model"""
        # input_ids = np2.array(input_ids)
        # tokens = Tensor(input_ids, mstype.int32)
        tokens = input_ids
        input_mask = F.cast(self.not_equal(tokens, self.pad_token), mstype.float32)
        input_position = F.tuple_to_array(F.make_range(self.len))
        input_position = P.Tile()(self.expand(input_position, 0), (self.batch_size, 1))

        attention_mask = self.get_attention_mask(input_mask)
        logits = self.network(tokens, input_position, attention_mask)

        # log_probs = self.log_softmax(logits)
        # input_mask_b = input_mask

        # Get label corresponding to input tokens
        labels = F.cast(np2.concatenate((input_ids[:, 1:], np2.ones((input_ids.shape[0], 1)) * self.pad_token), axis=-1), mstype.int32)
        # labels = self.slice(input_labels, (0, 1), (self.batch_size, self.len + 1),
        #                     (1, 1))
        labels = P.Reshape()(labels, (-1,))
        # input_mask = P.Reshape()(input_mask, (-1,))
        # if mask_ids_input == None:
        #     # model.infer_predict_layout()
        #     output = self.loss(logits, labels, input_mask)
        # else:
        #     # inference
        #     # mask_ids_input = F.cast(mask_ids_input, mstype.float32)
        #     # mask_ids_input = P.Reshape()(mask_ids_input, (-1,))
        output = self.loss(logits, labels, mask_ids_input)
        return output #log_probs, input_mask_b

        # log_probs = self.log_softmax(logits)
        # return log_probs


class EvalNet(nn.Cell):
    """
    PengChengMind evaluation net
    Args:
        backbone: backbone network of PengChengMind
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=128297, seq_length=4096):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        # used for incremental prediction
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        # taoht, not use use_past
        # if self.is_first_iteration is False:
        #     attention_mask = P.Tile()(self.all_ones_attention_mask, (bs, 1, 1))
        # else:
        #     attention_mask = self.get_attention_mask(input_mask)
        attention_mask = self.get_attention_mask(input_mask)  # For speed, ignore not first_iteration cond.
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)

        # log_probs = self.log_softmax(logits)
        # index = current_index.view(1,)
        # logits = self.gather(log_probs, index, 0)
        # logits = logits.view(bs, 1, -1)
        # return logits

        log_probs = self.log_softmax(logits)
        index = current_index.view(1,)
        logits = self.gather(log_probs, index, 0)
        logits = logits.view(bs, 1, -1)

        return logits
        # # print(type(logits))
        # # print(logits)
        # # print("logits: ", logits.data.asnumpy().shape)
        # # print(logits)
        # index = current_index.view(1,)
        # logits = self.gather(logits, index, 0)
        # logits = logits.view(bs, 1, -1)
        # log_probs = self.log_softmax(logits)
        # return log_probs


class EvalNet_use_past(nn.Cell):
    """
    PengChengMind evaluation net
    Args:
        backbone: backbone network of PengChengMind
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=0, seq_length=4096):
        super(EvalNet_use_past, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        # used for incremental prediction
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        # taoht, not use use_past
        if self.is_first_iteration is False:
            attention_mask = P.Tile()(self.all_ones_attention_mask, (bs, 1, 1))
        else:
            attention_mask = self.get_attention_mask(input_mask)
        # attention_mask = self.get_attention_mask(input_mask)  # For speed, ignore not first_iteration cond.
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)

        log_probs = self.log_softmax(logits)
        index = current_index.view(1,)
        logits = self.gather(log_probs, index, 0)
        logits = logits.view(bs, 1, -1)

        return logits

class EvalNet_200B(nn.Cell):
    """
    PengChengMind evaluation net
    Args:
        backbone: backbone network of PengChengMind
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
        current_index: the index of current token
        init_reset: whether reset saved states
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False, pad_token=0, seq_length=4096):
        super(EvalNet_200B, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.pad_token = pad_token
        self.argmax = P.Argmax()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.gather = P.GatherV2().shard(((1, 1), (1,)))
        self.log_softmax = P.LogSoftmax().shard(((1, 1),))
        self.get_attention_mask = AttentionMask(seq_length)
        self.expand = P.ExpandDims().shard(((1, 1, 1),))
        # used for incremental prediction
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, seq_length)), mstype.float32)

    def construct(self, input_ids, current_index, init_reset=True, batch_valid_length=None):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, self.pad_token), mstype.float32)
        bs, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        attention_mask = self.get_attention_mask(input_mask)  # For speed, ignore not first_iteration cond.
        logits = self.backbone(input_ids, input_position, attention_mask,
                               init_reset, batch_valid_length)

        log_probs = self.log_softmax(logits)
        index = current_index.view(1, )
        logits = self.gather(log_probs, index, 0)
        logits = logits.view(bs, 1, -1)

        return logits
