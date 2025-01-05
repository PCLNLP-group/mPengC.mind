import moxing as mox
import os, time
import mindspore
import mindspore as ms
from multiprocessing import Pool
import numpy as np
from mindspore.common import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

import argparse

WORKROOT = None
### Defines whether the task is a training environment or a debugging environment ###
def WorkEnvironment(environment):
    global WORKROOT
    if environment == 'train':
        workroot = '/home/work/user-job-dir'
    elif environment == 'debug':
        workroot = '/home/ma-user/work'
    print('current work mode:' + environment + ', workroot:' + workroot)
    WORKROOT = workroot
    return workroot

rank_id = int(os.getenv('RANK_ID', '0'))
#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_local_path', default='/cache/ckpt_tmp',
                    type=str,
                    help="/cache/ckpt_tmp")
parser.add_argument('--multi_data_url',
                     help='path to multi dataset',
                     default=WorkEnvironment('train'))
parser.add_argument('--pretrain_url',
                    default="/cache/Pretrain/",
                    help='Location of pretrain_url.')
parser.add_argument('--model_url',
                    default="",
                    help='Location of pretrain_url.')
parser.add_argument('--grampus_code_file_name',
                    default="",
                    help='Location of pretrain_url.')

parser.add_argument('--obs_save_path', default='obs://pcl-2023/merged_7B_ckpts/right_noFA_RoPE_noTopQ_Mind7b_npy/',
                    type=str,
                    help="OBS save PATH")
parser.add_argument('--ckpt_obs_path', default='obs://pcl-2023/ckpts/stage1_oldenv_pp2',
                    type=str,
                    help="obs://pcl-2023/ckpts/stage1_oldenv_pp2")

parser.add_argument('--ckpt_prefix', default='merged_7b',
                    type=str,
                    help="oldProj_mind_7b_pp2")
parser.add_argument('--ckpt_step',
                    default="3500",
                    type=str,
                    help='ckpt sink_step number',)
args = parser.parse_args()

if rank_id % 8 == 0:
    # # download 7B checkpoint for rank 400
    # ckpt_obs_path = f"obs://pcl-2023/merged_7B_ckpts/right_noFA_RoPE_noTopQ_Mind7b/"
    # ckpt_local_path = "/cache/ckpt_tmp"
    # ckpt_prefix = "merged_7b"
    # ckpt_step = "3500"

    ckpt_local_path = args.ckpt_local_path
    ckpt_obs_path = args.ckpt_obs_path
    ckpt_prefix = args.ckpt_prefix
    ckpt_step = args.ckpt_step

    ckpt_full_name = f"{ckpt_prefix}_{ckpt_step}-4.ckpt"

    if not os.path.exists(ckpt_local_path):
        os.makedirs(ckpt_local_path, exist_ok=True)

    def download_ckpt(ckpt_name_prefix):
        if not os.path.exists(f"{ckpt_local_path}/"):
            os.makedirs(f"{ckpt_local_path}/", exist_ok=True)

        mox.file.copy(src_url=f"{ckpt_obs_path}/{ckpt_name_prefix}",
                               dst_url=f"{ckpt_local_path}/{ckpt_name_prefix}")

    time0 = time.time()
    download_ckpt(ckpt_full_name)


    if rank_id % 8 == 0:
        model_prefix = f"{ckpt_local_path}/{ckpt_full_name}"
        param_dict = ms.load_checkpoint(model_prefix)

        # adam_names = [item for item in param_dict.keys() if 'adam' in item]
        # for item in adam_names:
        #     param_dict.pop(item)
        # param_dict.pop("scale_sense")
        # param_dict.pop("global_step")
        # param_dict.pop("current_iterator_step")
        # param_dict.pop("last_overflow_iterator_step")
        # param_dict.pop("epoch_num")
        # param_dict.pop("step_num")

        param_list = []
        for (key, value) in param_dict.items():
            print(key, value)

            each_param = {}
            each_param["name"] = key
            param_data = ms.Tensor(value.data, ms.float16).asnumpy()
            each_param["data"] = param_data
            param_list.append(each_param)

        local_CKPT_npy_path = "/cache/chatMind7B_merged_ckpt.npy"
        ckpt_obs_path = f"obs://pcl-2023/merged_7B_ckpts/right_noFA_RoPE_noTopQ_Mind7b_npy/"
        if not mox.file.exists(ckpt_obs_path):
            mox.file.make_dirs(ckpt_obs_path)
        # ckpt_npy_obs_path = os.path.join(ckpt_obs_path, f"{ckpt_prefix}_{ckpt_step}-4_ckpt.npy")
        ckpt_npy_obs_path = os.path.join(ckpt_obs_path, f"merged_ckpt")
        np.save(local_CKPT_npy_path, param_list)
        mox.file.copy(local_CKPT_npy_path, ckpt_npy_obs_path)
