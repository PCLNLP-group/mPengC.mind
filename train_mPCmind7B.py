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
PengChengMind train script
"""

import datetime
import json
import glob
import os
import math
import time

from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.nn.transformer import TransformerOpParallelConfig, CrossEntropyLoss, TransformerRecomputeConfig
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pengcheng_mind_7B import PengChengMindWithLoss, PengChengMindModel
from src.pengcheng_mind_wrapcell import PengChengMindTrainOneStepWithLossScaleCell, PengChengMindTrainPipelineWithLossScaleCell
from src.pengcheng_mind_config import set_parse, PengChengMindConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.callbacks import EvalCallBack, LossCallBack, LossSummaryCallback
from src.metrics import PPLMetric

from src.utils import download_data, download_OneCKPT_from_obs, download_ckpt_from_obs
from src.utils import StrategySaveCallback, CheckpointSaveCallback
from mindspore.common import Parameter
from mindspore.common.tensor import Tensor
import numpy as np
from mindspore import Profiler
from mindspore.train import Callback
import mindspore

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)

class ProfileStep(Callback):
    def __init__(self, start_step, stop_step, output_path):
        super(ProfileStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(start_profile=False, output_path=output_path)

    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()

def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0

    }, {
        'order_params': params
    }]
    return group_params


def add_checkpoint_callback_policy(args_param, callback, rank_id):
    r"""
    Add checkpoint policy to callback.
    """

    if args_param.save_checkpoint:
        if not os.path.exists(args_param.save_checkpoint_path):
            os.makedirs(args_param.save_checkpoint_path, exist_ok=True)
        # checkpoint store epoch_num and step_num info
        ckpt_append_info = [{"epoch_num": args_param.has_trained_epoches, "step_num": args_param.has_trained_steps}]
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args_param.save_checkpoint_steps,
                                       keep_checkpoint_max=1,
                                       integrated_save=False,
                                       append_info=ckpt_append_info,
                                       exception_save=True)# 临终遗言

        save_dir_rank = os.path.join(args_param.save_checkpoint_path, f"rank_{rank_id}")
        save_ckptfile_name = args_param.ckpt_name_prefix + '_' + str(rank_id)
        if not os.path.exists(save_dir_rank):
            os.makedirs(save_dir_rank, exist_ok=True)
        ckpoint_cb = ModelCheckpoint(prefix=save_ckptfile_name,
                                     directory=save_dir_rank,
                                     config=ckpt_config)
        callback.append(ckpoint_cb)

        if not args_param.offline:
            ckpt_save_obs_cb = CheckpointSaveCallback(local_ckpt_dir=save_dir_rank,
                                                      local_rank=rank_id,
                                                      has_trained_epoch=args_param.has_trained_epoches,
                                                      has_trained_step=args_param.has_trained_steps,
                                                      bucket=args_param.save_checkpoint_bucket_dir,
                                                      syn_obs_steps=args_param.save_checkpoint_steps)
            callback.append(ckpt_save_obs_cb)

    if rank_id == (args_param.device_num-1):
        if not args_param.offline:
            callback.append(LossSummaryCallback(summary_dir="summary",
                                                local_rank=rank_id,
                                                has_trained_epoch=args_param.has_trained_epoches,
                                                has_trained_step=args_param.has_trained_steps,
                                                micro_size=args_param.micro_size,
                                                bucket=args_param.save_summary_bucket_dir,
                                                syn_times=10))

    if not args_param.offline:
        callback.append(StrategySaveCallback(strategy_path=f'/cache/strategy_{rank_id}.ckpt',
                                             local_rank=rank_id,
                                             has_trained_step=args_param.has_trained_steps,
                                             bucket=args_param.save_strategy_bucket_dir))



def restore_checkpoint(args_param, sink_size, dataset, model, network, epoch, cache_url='/cache/Ckpt/'):
    r"""
    Load checkpoint process.
    """
    print("======start single checkpoint", flush=True)
    ckpt_name = args_param.restore_ckpt_name_prefix
    restore_ranks = D.get_rank()

    ckpt_files = os.path.join(cache_url, f"rank_{restore_ranks}.ckpt")

                                # f"{ckpt_name}_{restore_ranks}-{args_param.restore_steps}_4.ckpt")
                                # f"rank_{restore_ranks}",

    # ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        print(f"There is no ckpt file in {cache_url}, in rank {restore_ranks}"
              f"current ckpt_files found is {os.listdir(cache_url)} "
              f"with pattern {ckpt_files}, so skip the loading.")

    time_stamp = datetime.datetime.now()
    print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} pre trained ckpt model {ckpt_files} loading",
          flush=True)
    # Load checkpoint files latest file
    print(f'Start to load from {ckpt_files}')
    param_dict = load_checkpoint(ckpt_files)
    # print(f"Load rank {restore_ranks} param_dict: \n", param_dict.keys())
    # print(f"===global_step: {param_dict['global_step'].data.asnumpy()}")
    # print(f"===current_iterator_step: {param_dict['current_iterator_step'].data.asnumpy()}")
    # print(f"===step_num: {param_dict['step_num'].data.asnumpy()}")
    # print(f"===loss_scale: {param_dict['loss_scale'].data.asnumpy()}")

    # param_dict["loss_scale"] = Parameter(Tensor(268435456, dtype=mstype.float32), name="loss_scale")
    #print(param_dict.keys())
    
    '''
    param_dict["loss_scale"] = Parameter(Tensor(param_dict['loss_scale'].data.asnumpy(), dtype=mstype.float32), name="loss_scale")
    param_dict["global_step"] = Parameter(Tensor([int(param_dict['global_step'].data.asnumpy())],
                                                 dtype=mstype.int32), name="global_step")
    param_dict["current_iterator_step"] = Parameter(Tensor(int(param_dict['current_iterator_step'].data.asnumpy()),
                                                           dtype=mstype.int32),
                                                           name="current_iterator_step")
    param_dict["step_num"] = Parameter(Tensor(int(param_dict['step_num'].data.asnumpy()),
                                              dtype=mstype.int32),
                                              name="step_num")
    param_dict["epoch_num"] = Parameter(Tensor(int(param_dict['epoch_num'].data.asnumpy()),
                                                 dtype=mstype.int32), name="epoch_num")
    param_dict["last_overflow_iterator_step"] = Parameter(Tensor(int(param_dict['last_overflow_iterator_step'].data.asnumpy()),
                                                 dtype=mstype.int32), name="last_overflow_iterator_step")
    if param_dict.get("epoch_num") and param_dict.get("step_num"):
        # args_param.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        args_param.has_trained_step = int(param_dict["step_num"].data.asnumpy())
    '''
    param_dict["loss_scale"] = Parameter(Tensor(32768, dtype=mstype.float32), name="loss_scale")
    param_dict["current_iterator_step"] = Parameter(Tensor([0],
                                                           dtype=mstype.int32),
                                                           name="current_iterator_step")
    param_dict["global_step"] = Parameter(Tensor([0],
                                                 dtype=mstype.int32), name="global_step")
    param_dict["current_iterator_step"] = Parameter(Tensor(0,
                                                           dtype=mstype.int32),
                                                           name="current_iterator_step")
    param_dict["step_num"] = Parameter(Tensor(0,
                                              dtype=mstype.int32),
                                              name="step_num")
    #param_dict["epoch_num"] = Parameter(Tensor(int(param_dict['epoch_num'].data.asnumpy()),
                                                 #dtype=mstype.int32), name="epoch_num")
    param_dict["last_overflow_iterator_step"] = Parameter(Tensor(0,
                                                 dtype=mstype.int32), name="last_overflow_iterator_step")
    #if param_dict.get("epoch_num") and param_dict.get("step_num"):
        # args_param.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        #args_param.has_trained_step = int(param_dict["step_num"].data.asnumpy())
    print(f"======================================================\n"*3)
    print(f"===global_step: {param_dict['global_step'].data.asnumpy()}")
    print(f"===current_iterator_step: {param_dict['current_iterator_step'].data.asnumpy()}")
    print(f"===step_num: {param_dict['step_num'].data.asnumpy()}")
    print(f"===loss_scale: {param_dict['loss_scale'].data.asnumpy()}")
    model.build(train_dataset=dataset, sink_size=sink_size, epoch=epoch)
    load_param_into_net(network, param_dict, strict_load=False)

def set_pipeline_parallel_context(args_opt):
    r"""Set pipeline parallel context."""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank_id, device_num))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=args_opt.parallel_mode,
        gradients_mean=False,
        full_batch=bool(args_opt.full_batch),
        loss_repeated_mean=True,
        device_num=device_num,
        enable_parallel_optimizer=bool(args_opt.optimizer_shard),
        optimizer_weight_shard_size=args_opt.optimizer_shard,
        parallel_optimizer_config={"gradient_accumulation_shard": False},
        pipeline_stages=args_opt.stage_num,
        strategy_ckpt_save_file=f'/cache/strategy_{rank_id}.ckpt',
        enable_alltoall=bool(args_opt.enable_alltoall))
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank_id, device_num


def run_train_pipeline(args_opt):
    r"""The main training process in pipeline."""
    if args_opt.save_graph:
        graphs_url = '/home/ma-user/modelarts/outputs/train_url_0'
        print(f"graphs_url: {graphs_url}\n")
        context.set_context(save_graphs=2,
                            save_graphs_path=graphs_url,
                            mode=context.GRAPH_MODE,
                            device_target=args_opt.device_target,
                            op_timeout=0)
    else:
        context.set_context(save_graphs=False,
                            mode=context.GRAPH_MODE,
                            device_target=args_opt.device_target,
                            op_timeout=0)# task优化

    context.set_context(max_device_memory="31GB")
    mindspore.set_seed(1234)
    rank_id = 0
    device_num = 1
    if args_opt.distribute == "true":
        rank_id, device_num = set_pipeline_parallel_context(args_opt)
    # copy data from the cloud to the /cache/Data
    cache_url = '/cache/Data/'
    if args_opt.offline:
        cache_url = args_opt.data_url
        print(os.listdir(cache_url))
    else:
        download_data(src_data_url=args_opt.data_url, tgt_data_path=cache_url, rank=rank_id)
        print(os.listdir(cache_url))
        # download_data(src_data_url=args_opt.eval_data_url, tgt_data_path=eval_cache_url, rank=rank_id)

    model_parallel_num = args_opt.op_level_model_parallel_num
    print(f'device_num:{device_num}')
    print(f'stage_num:{args_opt.stage_num}')
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    print(f'stage_device_num:{stage_device_num}')
    is_last_stage = (rank_id // stage_device_num) == args_opt.stage_num - 1
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size
    micro_batch_interleaved = args_opt.micro_batch_interleaved
    # recompute_config = TransformerRecomputeConfig(recompute=True,
    #                                               recompute_slice_activation=bool(args_opt.recompute_slice_activation))
    recompute_config = args_opt.recompute
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=bool(args_opt.optimizer_shard),
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=recompute_config)
    # add sequence_parallel
    parallel_config.sequence_parallel = args_opt.sequence_parallel
    # add select_recompute
    parallel_config.select_recompute = args_opt.select_recompute

    if args_opt.softmax_compute_fp32 == "FP32":
        softmax_compute_type = mstype.float32
    elif args_opt.softmax_compute_fp32 == "FP16":
        softmax_compute_type = mstype.float16
    else:
        raise ValueError(f"Unknown softmax_compute_fp32 {args_opt.softmax_compute_fp32}")

    if args_opt.top_query_softmax_fp32 == "FP32":
        top_query_softmax = mstype.float32
    elif args_opt.top_query_softmax_fp32 == "FP16":
        top_query_softmax = mstype.float16
    else:
        raise ValueError(f"Unknown top_query_softmax_fp32 {args_opt.top_query_softmax_fp32}")

    print(f"softmax_compute_type: {softmax_compute_type}")
    print(f"top_query_softmax: {top_query_softmax}")

    ################################################### pc-mind v1 ####################################################
    # config = PengChengMindConfig(batch_size=batch_size // parallel_config.micro_batch_num // micro_batch_interleaved,
    #                           num_heads=args_opt.num_heads,
    #                           hidden_size=args_opt.embedding_size,
    #                           seq_length=args_opt.seq_length,
    #                           vocab_size=args_opt.vocab_size,
    #                           use_moe=bool(args_opt.use_moe),
    #                           eod_token=args_opt.eod_id,
    #                           num_layers=args_opt.num_layers,
    #                           ffn_hidden_size=args_opt.embedding_size * 4,
    #                           eod_reset=bool(args_opt.eod_reset),
    #                           load_ckpt_path=args_opt.load_ckpt_path,
    #                           param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
    #                           enable_offload=bool(args_opt.opt_offload),
    #                           parallel_config=parallel_config,
    #                           use_rope=args_opt.use_rope,
    #                           use_flash_attention=args_opt.use_flash_attention,
    #                           pipeline_config_filename=args_opt.pipeline_config_filename)
    #                           # softmax_compute_fp32=softmax_compute_type)
    #
    # config.softmax_compute_fp32 = softmax_compute_type
    # config.top_query_softmax_fp32 = top_query_softmax
    # config.hidden_act = "gelu"
    #####################################################################################################################

    ################################################## pc-mind v2 #######################################################
    config = PengChengMindConfig(batch_size=batch_size // parallel_config.micro_batch_num // micro_batch_interleaved,
                                 num_heads=args_opt.num_heads,
                                 hidden_size=args_opt.embedding_size,
                                 seq_length=args_opt.seq_length,
                                 vocab_size=args_opt.vocab_size,
                                 use_moe=bool(args_opt.use_moe),
                                 eod_token=args_opt.eod_id,
                                 num_layers=args_opt.num_layers,
                                 ffn_hidden_size=int((2 * args_opt.embedding_size * 4 / 3) / 64) * 64,
                                 eod_reset=bool(args_opt.eod_reset),
                                 load_ckpt_path=args_opt.load_ckpt_path,
                                 param_init_type=mstype.float32 if args_opt.param_init_type == 'fp32' else mstype.float16,
                                 enable_offload=bool(args_opt.opt_offload),
                                 parallel_config=parallel_config,
                                 use_rope=args_opt.use_rope,
                                 use_flash_attention=args_opt.use_flash_attention,
                                 pipeline_config_filename=args_opt.pipeline_config_filename)
    # softmax_compute_fp32=softmax_compute_type)
    print(">>> FFN_Hidden_Size for PengChengMind-new7B is: {} >>>\n".format(config.ffn_hidden_size))
    config.softmax_compute_fp32 = softmax_compute_type
    config.top_query_softmax_fp32 = top_query_softmax
    # config.hidden_act = "swiglu"
    #####################################################################################################################

    print("[Configure] is: ", config, flush=True)
    pengcheng_mind = PengChengMindModel(config=config)
    loss = CrossEntropyLoss(config.parallel_config.dp_mp_config)
    pengcheng_mind_with_loss_net = PipelineCell(MicroBatchInterleaved(PengChengMindWithLoss(config, pengcheng_mind, loss),
                                                                   micro_batch_interleaved),
                                             config.parallel_config.micro_batch_num)
    pengcheng_mind_with_loss = _VirtualDatasetCell(pengcheng_mind_with_loss_net)
    print("[args_opt] is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr, end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step, decay_steps=args_opt.decay_steps)
    params = pengcheng_mind_with_loss.trainable_params()
    group_params = set_weight_decay(params)
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif args_opt.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.98,
                                      param_init_type=config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.98, eps=1e-8)

    ds = create_dataset(config.batch_size * parallel_config.micro_batch_num * micro_batch_interleaved,
                        data_path=cache_url,
                        device_num=stage_device_num,
                        rank=rank_id % stage_device_num,
                        eod_reset=True,
                        eod_id=args_opt.eod_id,
						data_start_index=args_opt.data_start_index,
                        full_batch=context.get_auto_parallel_context("full_batch"),
                        column_name=args_opt.data_column_name)
    # print(f"train full batch: {context.get_auto_parallel_context('full_batch')}")
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    loss_callback = LossCallBack(step_per_epoch, rank_id,
                                 is_last_stage=is_last_stage,
                                 micro_size=parallel_config.micro_batch_num * micro_batch_interleaved)

    callback = [TimeMonitor(callback_size), loss_callback]
    if args_opt.enable_profiler:
        callback.append(ProfileStep(12, 20, f"/home/ma-user/modelarts/outputs/train_url_0/profile_{rank_id}"))
    loss_scale_value = math.pow(2, 21)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=200)
    pengcheng_mind_with_grads = PengChengMindTrainPipelineWithLossScaleCell(
        pengcheng_mind_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)
    if args_opt.train_and_eval_mode:
        test_cache_url = '/cache/EvalData_test/'
        download_data(src_data_url="obs://research-my/lama_4096_6G_test",
                      tgt_data_path=test_cache_url, rank=rank_id, flag="test")

        ds_eval_test = create_dataset(config.batch_size * parallel_config.micro_batch_num * micro_batch_interleaved,
                                 data_path=test_cache_url,
                                 device_num=stage_device_num,
                                 rank=rank_id % stage_device_num,
                                 eod_reset=True,
                                 eod_id=args_opt.eod_id,
                                 data_start_index=0,
                                 full_batch=context.get_auto_parallel_context("full_batch"),
                                 column_name=args_opt.data_column_name)
        print(f"Eval dataset size: {ds_eval_test.get_dataset_size()}")

        ppl_metric = PPLMetric(config.seq_length)
        pengcheng_mind_with_loss_eval_net = _VirtualDatasetCell(PengChengMindWithLoss(config, pengcheng_mind, loss))
        model = Model(pengcheng_mind_with_grads, eval_network=pengcheng_mind_with_loss_eval_net, metrics={"ppl": ppl_metric})

        model.build(ds, ds_eval_test, sink_size=callback_size)
        eval_callback_test = EvalCallBack(model, ds_eval_test, ppl_metric)
        callback.append(eval_callback_test)
    else:
        model = Model(pengcheng_mind_with_grads)

    #local_ckpt_path = '/cache/pretrained_7b_fp16.ckpt'
    if args_opt.pre_trained:
        '''
        download_OneCKPT_from_obs(
            # obs_ckpt_url=f'obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/new-fp16-iter47200-pcmind_new7B_227Btks.ckpt',
            #obs_ckpt_url=f'obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/new-fp16-v660-pcmind_new7B_660Btks.ckpt',
            args_opt,
            local_ckpt_url=local_ckpt_path,
            rank=rank_id)
        '''
        cache_url = '/cache/Ckpt/'
        download_ckpt_from_obs(args_opt, cache_url, rank=rank_id)
        #param_dict = load_checkpoint(local_ckpt_path)
        #load_param_into_net(pengcheng_mind_with_loss, param_dict)
        #print("================load param ok=================", flush=True)
        restore_checkpoint(args_opt, callback_size, ds, model, pengcheng_mind_with_grads,
                                 epoch=actual_epoch_num,
                                 cache_url=cache_url)

    # #######################################################################
    # if not args_opt.offline:
    #     import moxing as mox
    #     gpu_model_npy_obspath = "obs://test-zy/merged_ckpt_pt.npy"
    #     local_npy_path = "/cache/ckpt.npy"
    #     if rank_id % 8 == 0:
    #         mox.file.copy(gpu_model_npy_obspath, local_npy_path)
    #         f = open(f"/tmp/download_npy.txt", 'w')
    #         f.close()
    #     while not os.path.exists(f"/tmp/download_npy.txt"):
    #         time.sleep(1)
    #     gpu_param_dict_array = np.load(local_npy_path, allow_pickle=True)
    #     #######################################################################
    #
    #     for key, value in pengcheng_mind.parameters_dict().items():
    #         print(key, value)
    #
    #     for param in pengcheng_mind.get_parameters():
    #         for gpu_param_dict in gpu_param_dict_array:
    #             key = gpu_param_dict['name']
    #             value = gpu_param_dict['data']
    #             if key == param.name or ('position' in key and 'position' in param.name):
    #                 # param.data.set_data(Tensor(value))
    #                 param.data.set_data(Tensor(value, config.param_init_type))
    #                 print('> setting: {}'.format(param.name))
    #     print("================load param ok=================", flush=True)
    #     #######################################################################
    #     # saving npu-ckpt for yunnao2-obs
    #     tokens_flag = "227B"
    #     save_local_ckpt = "./pcmind_new7B_{}tks.ckpt".format(tokens_flag)
    #     # mindspore.save_checkpoint(pengcheng_mind, save_local_ckpt)
    #     # if not args_opt.offline:
    #     #     if config.param_init_type == mstype.float32:
    #     #         mox.file.copy(save_local_ckpt, "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/"
    #     #                                            "fp32-{}".format(save_local_ckpt))
    #     #     else:
    #     #         mox.file.copy(save_local_ckpt, "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/"
    #     #                                            "fp16-{}".format(save_local_ckpt))
    #     #     print("================save param to OBS ok=================", flush=True)
    #######################################################################################################

        loss_callback.has_trained_step = args_opt.has_trained_steps
    add_checkpoint_callback_policy(args_opt, callback, rank_id)

    if rank_id%8 == 0:
        count_parameters = 0
        params_all = pengcheng_mind_with_loss.get_parameters()
        for param in params_all:
            # print(param)
            param_size = param.shape
            if len(param_size) == 1:
                count_parameters += param_size[0]
            else:
                sum_ceil = 1
                for i in param_size:
                    sum_ceil *= i
                count_parameters += sum_ceil
        print("====="*10)
        print("PC Mind parameters :", count_parameters)
        print("====="*10)

    print(">>>>>" * 10)
    print("Actual Train Steps:", actual_epoch_num)
    print(">>>>>" * 10)
    model.train(actual_epoch_num, ds, callbacks=callback,
                sink_size=callback_size, dataset_sink_mode=True)


if __name__ == "__main__":
    opt = get_args()
    set_parse(opt)
    if opt.per_batch_size == 0:
        raise ValueError("The per_batch_size has not been configured.")
    if bool(opt.enable_alltoall) is True and bool(opt.use_moe) is False:
        raise ValueError("The alltoall communication is only effective when applying moe")
    os.environ['HCCL_CONNECT_TIMEOUT'] = str(opt.hccl_connect_time)
    if opt.stage_num > 1:
        run_train_pipeline(opt)
    else:
        print(f"Set pipline stage number for {opt.stage_num}")
        run_train_pipeline(opt)
