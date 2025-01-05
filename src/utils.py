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
"""
network config setting, gradient clip function and dynamic learning rate function
"""
import argparse
import ast
import os, shutil
import time
import hashlib
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.communication.management as D
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size, create_group
from mindspore.nn import AdamWeightDecay
from mindspore.common import Parameter, ParameterTuple
from mindspore.common.initializer import initializer

from mindspore.train.callback import Callback
from multiprocessing import Process

try:
    import moxing as mox
except:
    print(">>> using Local-NPU machine ENV >>>\n")

from mindspore.nn.optim.optimizer import Optimizer
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common.api import jit

'''
_adam_opt = C.MultitypeFuncGraph("adam_opt")
_fused_adam_weight_decay = C.MultitypeFuncGraph("fused_adam_weight_decay")

def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
class AdamWeightDecay(Optimizer):
    r"""
    Implements the Adam algorithm with weight decay.

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\: gradients \: g, \: \text{learning rate} \: \gamma,
             \text {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t, \: \text{weight decay} \: \lambda \\
            &\textbf{Init}:  m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{repeat} \\
            &\hspace{5mm} t \leftarrow t+1 \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla f_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\left(\gamma \hat{\boldsymbol{m}}_{t}
             /\left(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon\right)+\lambda \boldsymbol{w}_{t-1}\right) \\
            &\textbf{until}\text { stopping criterion is met } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`\gamma` represents `learning_rate`,
    :math:`\beta_1, \beta_2` represent `beta1` and `beta2`, :math:`t` represents the current step,
    :math:`w` represents `params`, :math:`\gamma` represents `weight_decay`.

    Note:
        There is usually no connection between a optimizer and mixed precision. But when `FixedLossScaleManager` is used
        and `drop_overflow_update` in `FixedLossScaleManager` is set to False, optimizer needs to set the 'loss_scale'.
        As this optimizer has no argument of `loss_scale`, so `loss_scale` needs to be processed by other means, refer
        document `LossScale <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html>`_ to
        process `loss_scale` correctly.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`. If not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 1e-3.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
   """
    _support_parallel_optimizer = True

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self._parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self._parameters.clone(prefix="adam_v", init='zeros')
        self.fused_opt = P.AdamWeightDecay()
        if context.get_context("device_target") == "CPU":
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()

        if self.use_fused_opt:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps),
                        lr, weight_decay, self._parameters, self.moments1,
                        self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr),
                        weight_decay, self._parameters, self.moments1, self.moments2,
                        gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(
                    F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr,
                              weight_decay),
                    self._parameters, self.moments1, self.moments2,
                    gradients, self.decay_flags, self.optim_filter)
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                                  lr, weight_decay, self._parameters, self.moments1,
                                                  self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                                  weight_decay, self._parameters, self.moments1, self.moments2,
                                                  gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                                              self._parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result, lr

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
        if value == 'CPU':
            self.fused_opt.set_device("CPU")
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False
'''

class FP32StateAdamWeightDecay(AdamWeightDecay):
    r"""
        This class is almost same with the mindspore's AdamWeightDecay implements, the
        only difference is the optimizer's state will be always initialized with float32,
        where the original AdamWeightDecay will initialize the optimizer's state with float16,
        if the parameters are initialized with fp16.
        This setting will avoid overflow in training PengChengMind-Alpha model using fp16.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FP32StateAdamWeightDecay, self).__init__(params, learning_rate=learning_rate,
                                                       beta1=beta1,
                                                       beta2=beta2,
                                                       eps=eps,
                                                       weight_decay=weight_decay)

        self.moments1 = self.clone_state(self.parameters, prefix='adam_m', init='zeros')
        self.moments2 = self.clone_state(self.parameters, prefix='adam_v', init='zeros')

    def clone_state(self, parameter_tuple, prefix, init):
        r"""
            parameter_tuple: ParameterTuple. The parameters of the network
            prefix: str. The prefix name of the parameters
            init: str. The initialization method
        """
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.param_info.obj = new_state
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    #P.Print()("grad:", grad)
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    #P.Print()("grad norm:", norm)
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Bool", "Tensor", "Tensor", "Tensor")
def _apply_global_norm(enable_grad_fp16, clip_norm, global_norm, grad):
    if enable_grad_fp16:
        grad = P.Cast()(grad * clip_norm / global_norm, mstype.float16)
    else:
        grad = grad * clip_norm / global_norm
    return grad


def _get_model_parallel_group(mp):
    """

    Calculate the communication group of model parallel dim in one pipeline stage

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str


def _get_pipeline_group():
    """

    Calculate the communication group between all pipeline stages

    """
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.is_pipeline = context.get_auto_parallel_context("pipeline_stages") > 1
        # self.is_data_parallel = context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL
        # self.config = config
        # self.group_size = 1
        # if self.is_data_parallel:
        #     self.merge_op = P.identity()
        # else:
        #     self.merge_op = P.AllReduce()
        if self.is_pipeline:
            if context.get_auto_parallel_context("enable_parallel_optimizer"):
                group_size = get_group_size() // config.parallel_config.pipeline_stage
            else:
                group_size = config.parallel_config.model_parallel
            group_list, group_name = _get_model_parallel_group(group_size)
            # In avoid of the group name too long
            hashed = hashlib.md5(group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({group_name})={hashed}")
            group_name = str(hashed)
            create_group(group_name, group_list)
            self.allreduce = P.AllReduce(group=group_name)
            pipeline_group_list, pipeline_group_name = _get_pipeline_group()
            hashed = hashlib.md5(pipeline_group_name.encode()).hexdigest()[:48]
            print(f"Creating hash value for the group_name hash({pipeline_group_name})={hashed}")
            pipeline_group_name = str(hashed)
            create_group(pipeline_group_name, pipeline_group_list)
            self.allreduce2 = P.AllReduce(group=pipeline_group_name)
        else:
            group_size = get_group_size()
        zero_repeated_size = 1.0
        if context.get_auto_parallel_context("enable_parallel_optimizer") and \
            context.get_auto_parallel_context("optimizer_weight_shard_size") > 1:
            zero_shard_size = context.get_auto_parallel_context("optimizer_weight_shard_size")
            zero_repeated_size = config.parallel_config.data_parallel // zero_shard_size
        self.allreduce_group_size = ()
        # if self.is_data_parallel:
        #     self.allreduce_group_size = (1,) * len(params)
        # else:
        #     self.allreduce_group_size = self._get_scale_for_gradient_norm(params)
        for x in params:
            if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (zero_repeated_size,)
            elif "embedding_table" not in x.name:
                self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)
            else:
                if not config.parallel_config.vocab_emb_dp and "position_embedding.embedding_table" not in x.name and \
                        "top_query_embedding.embedding_table" not in x.name:
                    self.allreduce_group_size = self.allreduce_group_size + (config.parallel_config.data_parallel * 1.0,)
                else:
                    self.allreduce_group_size = self.allreduce_group_size + (group_size * 1.0,)

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        #P.Print()("square sum: ", square_sum)
        if self.is_pipeline:
            stage_square_reduce_sum = self.allreduce(square_reduce_sum)
            #P.Print()("stage square sum: ", stage_square_reduce_sum)
            global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
            #P.Print()("global square sum: ", global_square_reduce_sum)
            global_norms = F.sqrt(global_square_reduce_sum)
        else:
            global_norms = F.sqrt(P.AllReduce()(square_reduce_sum))
        return grads, global_norms

    # def _get_scale_for_gradient_norm(self, params):
    #     allreduce_group_size = ()
    #     for x in params:
    #         if "projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table" not in x.name:
    #             allreduce_group_size = allreduce_group_size + (1.0,)
    #         elif "embedding_table" not in x.name:
    #             allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
    #         else:
    #             if not self.config.parallel_config.vocab_emb_dp and "position_embedding.embedding_table" not in x.name \
    #                     and "top_query_embedding_table" not in x.name:
    #                 allreduce_group_size = allreduce_group_size + \
    #                                             (self.config.parallel_config.data_parallel * 1.0,)
    #             else:
    #                 allreduce_group_size = allreduce_group_size + (self.group_size * 1.0,)
    #     return allreduce_group_size


class ClipByGlobalNorm(nn.Cell):
    """

    Clip grads by global norm

    """

    def __init__(self, params, config, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        if config.param_init_type == mstype.float16 and config.enable_offload:
            self.enable_grad_fp16 = True
        else:
            self.enable_grad_fp16 = False

    def construct(self, grads):
        """Clip grads by global norm construct"""
        grads, global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.enable_grad_fp16, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PengChengMind network.
    """

    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def add_inference_params(opt):
    """Add inference params"""
    opt.add_argument("--frequency_penalty",
                     type=float,
                     default=1.5,
                     help="coefficient for frequency_penalty")
    opt.add_argument("--presence_penalty",
                     type=float,
                     default=0.3,
                     help="coefficient for presence_penalty")
    opt.add_argument("--max_generate_length",
                     type=int,
                     default=500,
                     help="the maximum number of generated token")
    opt.add_argument("--top_k_num",
                     type=int,
                     default=3,
                     help="the number for top_k sampling")
    opt.add_argument("--top_p",
                     type=float,
                     default=1.0,
                     help="top_p sampling threshold, enabled if less than 1.0")
    opt.add_argument("--end_token",
                     type=int,
                     default=2,
                     help="the token id for <end of document>")
    opt.add_argument("--use_pynative_op",
                     type=int,
                     default=1,
                     help="Whether use pynative op for postproecess")
    opt.add_argument("--use_past",
                     type=bool,
                     default=False,
                     help="Whether enable state reuse")


def add_training_params(opt):
    """Add training params"""
    opt.add_argument("--seq_length",
                     type=int, default=2048,
                     help="sequence length, default is 1024.")
    opt.add_argument("--vocab_size",
                     type=int, default=50048,
                     help="vocabulary size, default is 40000.")
    opt.add_argument("--embedding_size",
                     type=int, default=4096,
                     help="embedding table size, default is 16384.")
    opt.add_argument("--num_layers",
                     type=int, default=80,
                     help="total layers, default is 64.")
    opt.add_argument("--num_heads",
                     type=int, default=80,
                     help="head size, default is 128.")
    opt.add_argument("--stage_num",
                     type=int, default=1,
                     help="Pipeline stage num, default is 1.")
    opt.add_argument("--micro_size",
                     type=int, default=32,
                     help="Pipeline micro_size, default is 1.")
    opt.add_argument("--eod_reset",
                     type=int, default=1,
                     help="Enable eod mask, default is 1.")
    opt.add_argument("--warmup_step",
                     type=int, default=1000,
                     help="Warmup step, default is 2000.")
    opt.add_argument("--decay_steps",
                     type=int, default=150000,
                     help="Decay step, default is 200000.")
    opt.add_argument("--optimizer",
                     type=str, default="adam",
                     choices=["adam", "lamb"],
                     help="select which optimizer to be used, default adam")
    opt.add_argument("--opt_offload",
                     type=int, default=0,
                     help="Enable optimizer status offload to host CPU, default is 0")
    opt.add_argument("--use_moe",
                     type=int, default=0,
                     help="Use moe, default is 0")
    opt.add_argument("--expert_num",
                     type=int, default=1,
                     help="Expert number, only effective when applying moe, Default is 1")
    opt.add_argument("--per_token_num_experts_chosen",
                     type=int, default=1,
                     help="Expert nums chosen by each token, only effective when applying moe, default is 1")
    opt.add_argument("--eod_id",
                     type=int, default=2,
                     help="The id of end of document")
    opt.add_argument("--padding_id",
                     type=int, default=0,
                     help="The padding id of dataset")
    opt.add_argument("--epoch_size",
                     type=int, default=1,
                     help="The training epoch")
    opt.add_argument("--sink_size",
                     type=int, default=4,
                     help="The sink size of the training. default is 2")
    opt.add_argument("--full_batch",
                     default=0, type=int,
                     help="Import the full size of a batch for each card, default is 1")
    opt.add_argument("--optimizer_shard",
                     type=int, default=1,
                     help="Enable optimizer parallel, default is 1")
    opt.add_argument("--per_batch_size",
                     type=int, default=1,
                     help="The batch size for each data parallel way. default 0")
    opt.add_argument("--start_lr",
                     type=float, default=2e-5,
                     help="The start learning rate. default 5e-5")
    opt.add_argument("--end_lr",
                     type=float, default=1e-6,
                     help="The end learning rate. default 1e-6")
    opt.add_argument("--op_level_model_parallel_num",
                     type=int, default=8,
                     help="The model parallel way. default 8")
    opt.add_argument("--expert_parallel_num",
                     type=int, default=1,
                     help="The expert parallel way, only effective when applying moe. Default 1")
    opt.add_argument("--word_emb_dp",
                     type=int, default=1,
                     choices=[0, 1],
                     help="Whether do data parallel in word embedding. default 1")
    opt.add_argument("--gradient_aggregation_group",
                     type=int, default=4,
                     help="The gradient communication fusion group. default 4")
    opt.add_argument("--data_column_name",
                     type=str, default="input_ids",
                     help="Column name of datasets")
    opt.add_argument("--micro_batch_interleaved",
                     type=int, default=1,
                     help="Parallel split num of batch size. default 2")
    opt.add_argument("--recompute",
                     type=bool, default=False,
                     help="Enable recompute. default False")
    opt.add_argument("--softmax_compute_fp32",
                     type=str,
                     default='FP16',
                     choices=['FP32', 'FP16'],
                     help="enable softmax_compute_type fp32")
    opt.add_argument("--top_query_softmax_fp32",
                     type=str,
                     default='FP32',
                     choices=['FP32', 'FP16'],
                     help="enable top_query softmax_compute_type fp32")
    opt.add_argument("--load_compiler_cache",
                     help="Enable download and load compiler cache in obs",
                     type=bool,
                     default=False)
    opt.add_argument("--compiler_name",
                     type=str, default="obs://research-my/compiler_cache/",
                     help="compiler_name")


def add_context_args_mode(opt):
    """Add context args params"""
    opt.add_argument("--parallel_mode",
                     type=str,
                     default="semi_auto_parallel",
                     choices=['data_parallel', "semi_auto_parallel", "auto_parallel"], help="The parallel context mode")


def add_downstream_params(opt):
    opt.add_argument("--eval_task",
                     help="Enable evaluating the tasks. Currently supports WPLC, C3.",
                     default=None,
                     choices=['wplc', 'ocnli', 'c3', 'chid'],
                     type=str)
    opt.add_argument("--enable_client",
                     help="Enable evaluation as an client.",
                     action='store_true')
    opt.add_argument("--server_ip",
                     help="The server ip of the model. The evaluation will send the data to the server, so the option"
                          "should be enabled with enable_client together",
                     type=str,
                     default="127.0.0.1")
    opt.add_argument("--port",
                     help="The server port of the model. The evaluation will send the data to the server, so the option"
                          "should be enabled with enable_client together",
                     type=str,
                     default="1500")


def add_retrain_params(opt):
    """
    Add parameters about retrain.
    """
    opt.add_argument("--pre_trained",
                     type=str,
                     default=None,
                     help="Pretrained checkpoint path.")
    opt.add_argument("--restore_steps",
                     type=int,
                     default=600,
                     help="restore checkpoint steps.")
    opt.add_argument("--restore_ckpt_name_prefix",
                     type=str,
                     default='PengChengMind_delta',
                     help="restore checkpoint name.")
    opt.add_argument("--restore_checkpoint_bucket_dir",
                     type=str,
                     default="s3://research-my/taoht-100b/ckpt_100b/PengChengMind_delta_exp00/",
                     help="restore checkpoint obs path.")
    opt.add_argument("--save_checkpoint_path",
                     type=str,
                     default="/cache/ckpt_new",
                     help="Save checkpoint path.")
    opt.add_argument("--save_checkpoint_bucket_dir",
                     type=str,
                     default="s3://research-my/test",
                     help="Save checkpoint path.")
    opt.add_argument("--save_strategy_bucket_dir",
                     type=str,
                     default="obs://research-my/predict_strategy/test/",
                     help="Save checkpoint path.")
    opt.add_argument("--save_summary_bucket_dir",
                     type=str,
                     default="obs://research-my/summary/test/",
                     help="Save checkpoint path.")
    opt.add_argument("--keep_checkpoint_max",
                     type=int,
                     default=1,
                     help="Max checkpoint save number.")
    opt.add_argument("--save_checkpoint_steps",
                     type=int,
                     default=300,
                     help="Save checkpoint step number.")
    opt.add_argument("--save_checkpoint",
                     type=ast.literal_eval,
                     default=True,
                     help="Whether save checkpoint in local disk.")
    opt.add_argument("--ckpt_name_prefix",
                     type=str,
                     default="PengChengMind_2400rst-1",
                     help="Saving checkpoint name prefix.")
    opt.add_argument("--has_trained_epoches",
                     type=int,
                     default=0,
                     help="Epoches has been trained before.")
    opt.add_argument("--has_trained_steps",
                     type=int,
                     default=0,
                     help="Steps has been trained before.")
    opt.add_argument("--data_start_index",
                     type=int,
                     default=0,
                     help="datasets has been trained before. 0")

def get_args(inference=False):
    """train function for PengChengMind"""
    parser = argparse.ArgumentParser(description="PengChengMind training")
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num",
                        type=int,
                        default=256,
                        help="Use device nums, default is 128.")
    parser.add_argument("--distribute",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Run distribute, default is true.")
    parser.add_argument("--load_ckpt_name",
                        type=str,
                        default='PengChengMind3.ckpt',
                        help="checkpint file name.")
    parser.add_argument("--load_ckpt_path",
                        type=str,
                        default=None,
                        help="predict file path.")
    parser.add_argument('--data_url',
                        default=None,
                        help='Location of data.')
    parser.add_argument('--eval_data_url',
                        default=None,
                        help='Location of eval data.')
    parser.add_argument('--train_url',
                        default="/cache/Graph/",
                        help='Location of training outputs.')
    parser.add_argument("--run_type",
                        type=str,
                        default="train",
                        choices=["train", "predict"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="7B",
                        choices=["200B", "13B", "7B", "2.6B", "1.3B", "100B", "350M"],
                        help="The scale of the model parameters")
    parser.add_argument("--device_target",
                        type=str,
                        default="Ascend",
                        choices=["Ascend", "GPU"],
                        help="The running device")
    parser.add_argument("--strategy_load_ckpt_path",
                        type=str,
                        default="",
                        help="The training prallel strategy for the model.")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        default="./tokenizer_path",
                        help="The path where stores vocab and vocab model file")
    parser.add_argument("--param_init_type",
                        type=str,
                        default="fp16",
                        help="The initialization type for parameters. Default fp16.")
    parser.add_argument("--offline",
                        type=int,
                        default=1,
                        help="Running on cloud of not. Default 1.")
    parser.add_argument("--export",
                        type=int,
                        default=0,
                        help="Whether export mindir for serving.")
    parser.add_argument("--incremental_training",
                        type=int,
                        default=0,
                        help="Enable incremental training. Default 0.")
    parser.add_argument("--train_and_eval_mode",
                        type=int,
                        default=0,
                        help="Enable evaling while training. Default 0.")
    parser.add_argument("--save_strategy_name",
                        type=str,
                        default="strategy_inference_d16_mp16_dp1_rank-*.ckpt",
                        help="Save checkpoint path.")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=10,
                        help="The eval step in train and eval mode. Default 10.")
    parser.add_argument("--enable_alltoall",
                        type=int,
                        default=0,
                        help="Enable alltoall communication, only effective when applying moe. Default 0")
    parser.add_argument("--hccl_connect_time",
                        type=int,
                        default=7200,
                        help="Set the hccl build time out, only effective on Ascend. Default 6000")
    # add RoPE
    parser.add_argument('--use_rope',
                        type=ast.literal_eval,
                        default=False,
                        help="Use rotary position embedding")
    # use flash attention
    parser.add_argument('--use_flash_attention',
                        type=ast.literal_eval,
                        default=False,
                        help="Use rotary position embedding")
    # add sequence parallel
    parser.add_argument('--sequence_parallel',
                        type=ast.literal_eval,
                        default=False,
                        help="Use sequence parallel")
    # add select recompute
    parser.add_argument('--select_recompute',
                        type=ast.literal_eval,
                        default=False,
                        help="Use select recompute")
    parser.add_argument('--duRepeate',
                        type=bool,
                        default=True,
                        help="Use select recompute")

    # enable profiler
    parser.add_argument('--enable_profiler',
                        type=ast.literal_eval,
                        default=False,
                        help="Profile timeline")

    # save ir
    parser.add_argument('--save_graph',
                        type=ast.literal_eval,
                        default=False,
                        help="save ir graph")
    
    # translate id
    parser.add_argument('--lang_idx',
                        type=int,
                        default=0,
                        help="")

    # add pipeline config
    parser.add_argument('--pipeline_config_filename',
                        type=str,
                        default=None,
                        help="configuration for pipeline")

    add_context_args_mode(parser)
    add_training_params(parser)
    add_retrain_params(parser)
    if inference:
        add_inference_params(parser)
        add_downstream_params(parser)
    args_opt = parser.parse_args()

    return args_opt

def ckpt_copy_tar_new(obs_path, target_path="/cache/ckpt"):
    """
        requires the obs_path to be a complete name
        Copy tar file from the obs to the /cache/
    """
    import moxing as mox
    sub_name_list = ['part_0.tar', 'part_1.tar', 'part_2.tar', 'part_3.tar']
    for item in sub_name_list:
        sub_name = obs_path + item
        tmp_name = 'model.tar'

        mox.file.copy(sub_name, os.path.join(target_path, tmp_name))
        os.system('cd {}; tar -xvf {}'.format(target_path, tmp_name))


def get_ckpt_file_list(ckpt_path, device_num):
    # path = os.listdir(ckpt_path)
    # print('Get path list:', path, flush=True)

    returned_list = []
    for i in range(0, device_num):
        returned_list.append('filerted_{}.ckpt'.format(i))
    # filtered = [item for item in path if 'embedding' not in item]save_checkpoint_bucket_dir

    # filtered.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))
    returned_list = [os.path.join(ckpt_path, item) for item in returned_list if 'embedding' not in item]
    print("Sorted list", returned_list)
    for item in returned_list:
        fsize = os.path.getsize(item)
        f_gb = fsize / float(1024) / 1024
        print(item, " :{:.2f} MB".format(f_gb))
    return returned_list

class CheckpointSaveCallback(Callback):
    def __init__(self, local_ckpt_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                 bucket='obs://ckeckpoint_file/path/', syn_obs_steps=100):
        self.local_ckpt_dir = local_ckpt_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = bucket
        self.syn_obs_steps = syn_obs_steps

        self.obs_save_rank_dir = os.path.join(bucket, "rank_" + str(local_rank))
        if not mox.file.exists(self.obs_save_rank_dir):
            print("Creating ckeckpoint bueckt dir {}".format(self.obs_save_rank_dir))
            mox.file.make_dirs(self.obs_save_rank_dir)

    def step_end(self, run_context):
        #         try:
        #             os.system('ls /cache/ckpt_new/rank_0/')
        #         except:
        #             pass
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step
        if cur_step % self.syn_obs_steps == 0:

            try:
                print("Copying ckpt to the buckets start", flush=True)
                self.syn_files()
                # time.sleep(self.local_rank % 16)
                # mox.file.copy_parallel(self.local_ckpt_dir, self.obs_save_rank_dir)
                print("Copying ckpt to the buckets ends", flush=True)
            except Exception as error:
                print(error)
                print('\n###########Upload ckpt Error#############\n')
                print("Copying ckpt to the buckets again", flush=True)
                self.syn_files()
                # time.sleep((self.local_rank+15) % 16)
                # mox.file.copy_parallel(self.local_ckpt_dir, self.obs_save_rank_dir)
                print("Copying ckpt to the buckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.local_ckpt_dir, self.obs_save_rank_dir), name="file_sync")
        process.start()

class CheckpointSaveCallback_Exception():
    def __init__(self, local_ckpt_dir, local_rank=0,
                 bucket='obs://ckeckpoint_file/path/'):
        self.local_ckpt_dir = local_ckpt_dir
        self.local_rank = local_rank
        self.bucket = bucket

        self.obs_save_rank_dir = os.path.join(bucket, "rank_" + str(local_rank))
        if not mox.file.exists(self.obs_save_rank_dir):
            print("Creating ckeckpoint bueckt dir {}".format(self.obs_save_rank_dir))
            mox.file.make_dirs(self.obs_save_rank_dir)

        self.obs_save_rank_dir = os.path.join(self.obs_save_rank_dir, local_ckpt_dir.split("/")[-1])

    def step_end(self):
        try:
            print("Copying ckpt to the buckets start", flush=True)
            self.syn_files()
            print("Copying ckpt to the buckets ends", flush=True)
        except Exception as error:
            print(error)
            print('\n###########Upload ckpt Error#############\n')
            print("Copying ckpt to the buckets again", flush=True)
            self.syn_files()
            print("Copying ckpt to the buckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.local_ckpt_dir, self.obs_save_rank_dir), name="file_sync")
        process.start()


class CheckpointSaveCallback_7B(Callback):
    def __init__(self, args, local_ckpt_dir, local_rank=0, has_trained_epoch=0, has_trained_step=0,
                 bucket='obs://ckeckpoint_file/path/', syn_obs_steps=100):
        self.local_ckpt_dir = local_ckpt_dir
        self.local_rank = local_rank
        self.has_trained_epoch = has_trained_epoch
        self.has_trained_step = has_trained_step

        self.bucket = bucket
        self.syn_obs_steps = syn_obs_steps
        self.stage_num = args.stage_num

        D.init()
        self.device_num = D.get_group_size()
        rank = D.get_rank()

        if self.local_rank % (self.device_num // self.stage_num) == 0:
            self.obs_save_rank_dir = os.path.join(bucket, "rank_" + str(local_rank))
            if not mox.file.exists(self.obs_save_rank_dir):
                print("Creating ckeckpoint bueckt dir {}".format(self.obs_save_rank_dir))
                mox.file.make_dirs(self.obs_save_rank_dir)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.has_trained_step

        if cur_step % self.syn_obs_steps == 0:
            if self.local_rank % (self.device_num // self.stage_num) == 0:
                try:
                    print("Copying ckpt to the buckets start", flush=True)
                    self.syn_files()
                    # time.sleep(self.local_rank % 16)
                    print("Copying ckpt to the buckets ends", flush=True)
                except Exception as error:
                    print(error)
                    print('\n###########Upload ckpt Error#############\n')
                    print("Copying ckpt to the buckets again", flush=True)
                    self.syn_files()
                    # time.sleep((self.local_rank+15) % 16)
                    print("Copying ckpt to the buckets ends", flush=True)

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.local_ckpt_dir, self.obs_save_rank_dir),
                          name="file_sync")
        process.start()


class Strategy_and_compilerSaveCallback(Callback):
    def __init__(self, strategy_path, local_rank=0, has_trained_step=0,
                 bucket='obs://mindspore-file/strategy_ckpt/', compiler_name='2520_pp21_mp2dp60ms96', sym_step=100):
        self.strategy_path = strategy_path
        self.local_rank = local_rank
        self.sym_step = sym_step
        self.bucket = bucket
        self.has_trained_step = has_trained_step
        self.has_send = False
        self.file_name = strategy_path.split('/')[-1]
        self.compiler_cache_local_path = f"/cache/compiler_cache/rank_{local_rank}/"
        self.compiler_cache_obs_path = f"{compiler_name}/rank_{local_rank}/"

        if not mox.file.exists(self.bucket):
            print("Creating summary bueckt dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)
        if not mox.file.exists(self.compiler_cache_obs_path):
            mox.file.make_dirs(self.compiler_cache_obs_path)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if self.has_send is False:
            print("Send strategy file to the obs")
            self.syn_files()
            print("Send compiler cache file to the obs")
            time.sleep((self.local_rank % 8)*30)
            self.syn_files2()
            self.has_send = True

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.strategy_path, self.bucket + self.file_name),
                          name="file_sync")
        process.start()

    def syn_files2(self):
        process = Process(target=mox.file.copy_parallel, args=(self.compiler_cache_local_path, self.compiler_cache_obs_path),
                          name="file_sync")
        process.start()

class StrategySaveCallback(Callback):
    def __init__(self, strategy_path, local_rank=0, has_trained_step=0,
                 bucket='obs://mindspore-file/strategy_ckpt/', sym_step=100):
        self.strategy_path = strategy_path
        self.local_rank = local_rank
        self.sym_step = sym_step
        self.bucket = bucket
        self.has_trained_step = has_trained_step
        self.has_send = False
        self.file_name = strategy_path.split('/')[-1]

        if not mox.file.exists(self.bucket):
            print("Creating summary bueckt dir {}".format(self.bucket))
            mox.file.make_dirs(self.bucket)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        if self.has_send is False:
            print("Send strategy file to the obs")
            self.syn_files()

            self.has_send = True

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel, args=(self.strategy_path, self.bucket + self.file_name),
                          name="file_sync")
        process.start()


class GraphsSaveCallback(Callback):
    def __init__(self, Graph_path, local_rank=0, has_trained_step=0,
                 bucket='obs://research-my/taoht-100b/graphs_100b/', sym_step=100):
        self.Graph_path = Graph_path
        self.local_rank = local_rank
        self.sym_step = sym_step
        self.bucket_obs = bucket + f"rank_{local_rank}"
        self.has_trained_step = has_trained_step
        self.has_send = False
        self.file_name = Graph_path.split('/')[-1]

        if not mox.file.exists(self.bucket_obs):
            print("Creating graph bueckt dir {}".format(self.bucket_obs))
            mox.file.make_dirs(self.bucket_obs)

    def step_end(self, run_context):
        if self.has_send is False:
            print("Send Graph file to the obs")
            self.syn_files()
            self.has_send = True

    def syn_files(self):
        process = Process(target=mox.file.copy_parallel,
                          args=(self.Graph_path, self.bucket_obs),
                          name="file_sync")
        process.start()


def download_data(src_data_url, tgt_data_path, rank, flag="0"):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    cache_url = tgt_data_path
    EXEC_PATH = '/tmp'
    if rank % 8 == 0:
        import moxing as mox
        print("Modify the time out from 300 to 30000")

        print(os.listdir(cache_url))
        for files in os.listdir(cache_url):
            files_path = os.path.join(cache_url, files)
            try:
                shutil.rmtree(files_path)
            except OSError:
                os.remove(files_path)
        print(f"Delete dataset in path {cache_url}...")
        print("begin download dataset", flush=True)
        if not os.path.exists(cache_url):
            os.makedirs(cache_url, exist_ok=True)
        mox.file.copy_parallel(src_url=src_data_url,
                               dst_url=cache_url)
        print("Dataset download succeed!", flush=True)

        f = open(f"/tmp/install_{flag}.txt", 'w')
        f.close()
    # stop
    while not os.path.exists(f"/tmp/install_{flag}.txt"):
        time.sleep(1)

def download_compiler_cache_from_obs(args_opt, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    compiler_cache_obs_path = f"{args_opt.compiler_name}/rank_{rank}/"
    compiler_cache_local_path = f"/cache/compiler_cache/rank_{rank}/"

    time.sleep(rank%16)
    mox.file.copy_parallel(src_url=compiler_cache_obs_path,
                           dst_url=compiler_cache_local_path)
    print(f"Rank {rank} download compiler cache {compiler_cache_obs_path} succeed!", flush=True)

    f = open(f"/tmp/restore_compiler_{rank}.txt", 'w')
    f.close()
    while not os.path.exists(f"/tmp/restore_compiler_{rank}.txt"):
        time.sleep(1)

def download_ckpt_from_obs(args_opt, cache_url, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """

    local_ckpt_url = os.path.join(cache_url, f"rank_{rank}.ckpt")
    #
    # if args_opt.restore_steps == 0:
    #     obs_ckpt_url = os.path.join(args_opt.restore_checkpoint_bucket_dir,
    #                                 f"rank_{rank}",
    #                                 args_opt.restore_ckpt_name_prefix.replace('*', str(rank)))
    # else:
    obs_ckpt_url = os.path.join(args_opt.restore_checkpoint_bucket_dir,
                                        f"rank_{rank}",
                                        f"{args_opt.restore_ckpt_name_prefix.replace('*', str(rank))}")

    import moxing as mox
    print("Modify the time out from 300 to 30000")
    # print(f"begin download ckpt from {obs_ckpt_url}")

    # if not os.path.exists(os.path.join(cache_url, f"rank_{rank}")):
    #     os.makedirs(os.path.join(cache_url, f"rank_{rank}"), exist_ok=True)
    # if not os.path.exists(cache_url):
    #     os.makedirs(cache_url, exist_ok=True)
    time.sleep(rank % 4)
    try:
        mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
    except Exception as error:
        print(f"Error: {error}, download again...")
        time.sleep(10)
        mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
    print(f"Rank {rank} download ckpt succeed!", flush=True)
    # print(f"Rank {rank} download ckpt {obs_ckpt_url} succeed!", flush=True)

    f = open(f"/tmp/restore_{rank}.txt", 'w')
    f.close()
    # stop
    while not os.path.exists(f"/tmp/restore_{rank}.txt"):
        time.sleep(1)
    print(os.listdir(cache_url))

def download_ckpt_from_obs_7B(args_opt, cache_url, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """

    # print(f"begin download ckpt from {obs_ckpt_url}")

    # if not os.path.exists(os.path.join(cache_url, f"rank_{rank}")):
    #     os.makedirs(os.path.join(cache_url, f"rank_{rank}"), exist_ok=True)
    # if not os.path.exists(cache_url):
    #     os.makedirs(cache_url, exist_ok=True)
    # time.sleep(rank % 4)
    if rank%8 ==0:
        local_ckpt_url = os.path.join(cache_url, f"rank.ckpt")

        stage_num = args_opt.stage_num

        D.init()
        device_num = D.get_group_size()
        stage_device_num = device_num // stage_num

        rank_download = (rank // stage_device_num) * stage_device_num

        obs_ckpt_url = os.path.join(args_opt.restore_checkpoint_bucket_dir,
                                    f"rank_{rank_download}",
                                    f"{args_opt.restore_ckpt_name_prefix}_{rank_download}-{args_opt.restore_steps}_4.ckpt")

        import moxing as mox
        print("Modify the time out from 300 to 30000")
        try:
            mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
        except Exception as error:
            print(f"Error: {error}, download again...")
            time.sleep(3)
            mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
        print(f"Rank {rank} download ckpt succeed!", flush=True)
        # print(f"Rank {rank} download ckpt {obs_ckpt_url} succeed!", flush=True)

        f = open(f"/tmp/restore.txt", 'w')
        f.close()
    # stop
    while not os.path.exists(f"/tmp/restore.txt"):
        time.sleep(1)
    print(os.listdir(cache_url))

def download_merged_ckpt_from_obs(args_opt, cache_url, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """

    local_ckpt_url = os.path.join(cache_url, f"merged.ckpt")
    obs_ckpt_url = "obs://research-my/taoht-100b/ckpt_100b/merged_100b/PengChengMind_delta_rst11_36900-3980-5548_layers_fp16.ckpt"

    import moxing as mox
    print("Modify the time out from 300 to 30000")
    print("begin download ckpt", flush=True)

    if rank%8 == 0:
        mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
        f = open(f"/tmp/restore.txt", 'w')
        f.close()
    while not os.path.exists(f"/tmp/restore.txt"):
        time.sleep(1)
    print(f"Rank {rank} download ckpt succeed!", flush=True)

def download_OneCKPT_from_obs(obs_ckpt_url, local_ckpt_url, rank):
    """
        Download the dataset from the obs.
        src_data_url (Str): should be the dataset path in the obs
        tgt_data_path (Str): the local dataset path
        rank (Int): the current rank id

    """
    import time
    import moxing as mox
    print("Modify the time out from 300 to 30000")
    print(f"begin download ckpt from {obs_ckpt_url}")

    if rank % 8 == 0:

        time.sleep(rank % 4)
        mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
        os.system(f"chmod 777 {local_ckpt_url}")
        print(f"Rank {rank} download ckpt {obs_ckpt_url} succeed!", flush=True)

        f = open(f"/tmp/restore.txt", 'w')
        f.close()
        # stop
    while not os.path.exists(f"/tmp/restore.txt"):
        time.sleep(1)


