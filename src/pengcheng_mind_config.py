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
network config setting
"""
from enum import Enum
import mindspore.common.dtype as mstype
import json

class OperatorType(str, Enum):
    ATTENTION_X = "ATTENTION_X"
    ATTENTION_QKV = "ATTENTION_QKV"
    ATTENTION_ATTEN = "ATTENTION_ATTEN"
    ATTENTION_SOFTMAX = "ATTENTION_SOFTMAX"
    ATTENTION_BMM = "ATTENTION_BMM"
    LINEAR_INPUT = "LINEAR_INPUT"
    LINEAR_X = "LINEAR_X"
    LINEAR_MAPPING = "LINEAR_MAPPING"
    LINEAR_GELU = "LINEAR_GELU"

class PengChengMindConfig:
    """
    PengChengMindConfig config class which defines the model size
    """

    def __init__(self,
                 batch_size=32,
                 seq_length=4096,
                 vocab_size=49984,
                 hidden_size=768,
                 ffn_hidden_size=768,
                 num_layers=12,
                 num_heads=12,
                 load_ckpt_path=None,
                 param_init_type=mstype.float16,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 eod_token=2,
                 pad_token=0,
                 use_past=False,
                 hidden_act='fast_gelu',
                 eod_reset=True,
                 enable_offload=False,
                 use_moe=False,
                 expert_num=1,
                 per_token_num_experts_chosen=1,
                 parallel_config=None,
                 run_type='train',
                 use_rope=False,
                 use_flash_attention=False,
                 pipeline_config_filename=None):
                 # softmax_compute_fp32=mstype.float16,

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.eod_token = eod_token
        # Use post-layernorm or pre-layernrom, default:pre-layernorm
        self.post_layernorm_residual = post_layernorm_residual
        self.load_ckpt_path = load_ckpt_path
        self.param_init_type = param_init_type
        self.dropout_rate = dropout_rate
        self.compute_dtype = mstype.float16
        self.parallel_config = parallel_config
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.use_past = use_past
        self.eod_reset = eod_reset
        self.enable_offload = enable_offload
        self.use_moe = bool(use_moe)
        self.expert_num = expert_num
        self.per_token_num_experts_chosen = per_token_num_experts_chosen
        self.run_type = run_type
        self.pad_token = pad_token
        self.use_rope = use_rope
        self.use_flash_attention = use_flash_attention
        # self.softmax_compute_fp32 = softmax_compute_fp32
        # if softmax_compute_fp32:
        #     self.softmax_compute_type = mstype.float32
        # else:
        #     self.softmax_compute_type = mstype.float16

        # if self.run_type == "predict":
        #     self.softmax_compute_type = mstype.float32

        pipeline_strategy = None
        self.pipeline_config = None

    def __str__(self):
        info = '===' * 10 + "[PengChengMindConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "--{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info

def set_parse_200B(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 12672
    args_opt.num_layers = 104
    args_opt.num_heads = 96
    if args_opt.run_type == "train":
        # args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        # args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1

def set_parse_100B(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 10240
    # args_opt.num_layers = 80
    # args_opt.num_heads = 80
    # args_opt.word_emb_dp = 4# 1
    # args_opt.op_level_model_parallel_num = 4 # 8
    if args_opt.run_type == "train":
        # args_opt.start_lr = 4e-5# 5e-5
        # args_opt.end_lr = 1e-6

        # args_opt.stage_num = 16
        # args_opt.micro_size = 32
        # args_opt.op_level_model_parallel_num = 16
        # if args_opt.optimizer_shard == 1:
        #     args_opt.op_level_model_parallel_num = 8
        #
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        # if args_opt.per_batch_size == 0:
        #     args_opt.per_batch_size = 2 # 8
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        # args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1

def set_parse_13B(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 5120
    args_opt.num_layers = 40
    args_opt.num_heads = 40
    args_opt.word_emb_dp = 1
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 5e-5
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 8
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_7B(args_opt):
    r"""
        Set config for 7B mode
    """
    args_opt.embedding_size = 4096
    args_opt.num_layers = 32
    args_opt.num_heads = 32
    # args_opt.word_emb_dp = 4# 1
    # args_opt.op_level_model_parallel_num = 4 # 8
    if args_opt.run_type == "train":
        # args_opt.start_lr = 4e-5# 5e-5
        # args_opt.end_lr = 1e-6

        # args_opt.stage_num = 16
        # args_opt.micro_size = 32
        # args_opt.op_level_model_parallel_num = 16
        # if args_opt.optimizer_shard == 1:
        #     args_opt.op_level_model_parallel_num = 8
        #
        # args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        # if args_opt.per_batch_size == 0:
        #     args_opt.per_batch_size = 2 # 8
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        # args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_2_6B(args_opt):
    r"""
        Set config for 2.6B mode
    """
    args_opt.embedding_size = 2560
    args_opt.num_layers = 32
    args_opt.num_heads = 32
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 1e-4
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 16
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1


def set_parse_1_3B(args_opt):
    r"""
        Set config for 1.3B mode
    """
    args_opt.embedding_size = 1024
    args_opt.num_layers = 16
    args_opt.num_heads = 32
    args_opt.op_level_model_parallel_num = 8
    if args_opt.run_type == "train":
        args_opt.start_lr = 1e-4
        args_opt.end_lr = 1e-6
        args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 16
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1

def set_parse_350M(args_opt):
    r"""
        Set config for 13B mode
    """
    args_opt.embedding_size = 1024
    args_opt.num_layers = 24
    args_opt.num_heads = 16
    if args_opt.run_type == "train":
        # args_opt.optimizer_shard = 1
        args_opt.full_batch = args_opt.opt_offload
        args_opt.micro_batch_interleaved = 1
        if args_opt.stage_num > 1:
            args_opt.word_emb_dp = 0
    elif args_opt.run_type == "predict":
        # args_opt.stage_num = 1
        args_opt.micro_size = 1
        if args_opt.per_batch_size == 0:
            args_opt.per_batch_size = 1

def set_parse(args_opt):
    r"""
        Set config according to the mode
    """
    parse_fn_dict = {"200B": set_parse_200B,
                     "100B": set_parse_100B,
                     "13B": set_parse_13B,
                     "7B": set_parse_7B,
                     "2.6B": set_parse_2_6B,
                     "1.3B": set_parse_1_3B,
                     "350M": set_parse_350M}
    if args_opt.mode not in parse_fn_dict.keys():
        raise ValueError("Invalid mode: {}. Optional mode: 200B, 13B, 2.6B and 1.3B".format(args_opt.mode))
    parse_fn_dict[args_opt.mode](args_opt)
