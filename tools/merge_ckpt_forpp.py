import moxing as mox
import os, time
import mindspore
import mindspore as ms
from multiprocessing import Pool
import numpy as np
from mindspore.common import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

from mindspore.parallel._parallel_serialization import _rank_list_for_transform_parallel_checkpoint,\
    _transform_parallel_checkpoint,_get_device_num_from_strategy,\
    _make_dir,_extract_layout_map,_extract_src_dst_layout_map,\
    _parameter_not_in_local_stage,_extract_pipeline_stage_num

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

# # right_noTopQ_noFA_RoPE_7B
# ckpt_obs_path = "obs://pcl-2023/ckpts/stage1_oldenv_pp2"
# ckpt_local_path = "/cache/ckpt_tmp"
# ckpt_name = 'oldProj_mind_7b_pp2'
# ckpt_step = '3500'
# train_nodes = 208

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

parser.add_argument('--obs_save_path', default='obs://pcl-2023/merged_7B_ckpts/right_noFA_RoPE_noTopQ_Mind7b/',
                    type=str,
                    help="OBS save PATH")
parser.add_argument('--ckpt_obs_path', default='obs://pcl-2023/ckpts/stage1_oldenv_pp2',
                    type=str,
                    help="obs://pcl-2023/ckpts/stage1_oldenv_pp2")

parser.add_argument('--ckpt_name', default='oldProj_mind_7b_pp2',
                    type=str,
                    help="oldProj_mind_7b_pp2")
parser.add_argument('--ckpt_step',
                    default="3500",
                    type=str,
                    help='ckpt sink_step number',)
args = parser.parse_args()


if rank_id % 8 == 0:
    # download and load 100B strategy for rank 512
    strategy_local_path = "/cache/strategy/"
    strategy_name_prefix = f"obs://pcl-2023/strategy/stage1_oldenv_pp2/"
    src_obs_strategy_inference = "obs://pcl-2023/strategy/7B_inference_onecard/mind_7b_node1.ckpt"

    if not os.path.exists(strategy_local_path):
        os.makedirs(strategy_local_path, exist_ok=True)

    train_strategy_dir = "/cache/7b_oldProj_pp2_mp8_dp13_strategy_restored.ckpt"

    ############# 6ff commit save_strategy_file error ############################################################
    # download and merge strategy for rank 512
    mox.file.copy_parallel(src_url=strategy_name_prefix, dst_url=strategy_local_path)
    ckpt_list = [os.path.join(strategy_local_path, i) for i in os.listdir(strategy_local_path) if '.ckpt' in i]
    print(len(ckpt_list))
    mindspore.merge_pipeline_strategys(strategy_local_path, train_strategy_dir)

    mox.file.copy(train_strategy_dir, "obs://pcl-2023/7b_oldProj_pp2_mp8_opt1_strategy.ckpt")
    ##############################################################################################################

    mox.file.copy("obs://pcl-2023/7b_oldProj_pp2_mp8_opt1_strategy_restored.ckpt", train_strategy_dir)
    strategy_dict = _extract_layout_map(train_strategy_dir)
    count = len(strategy_dict.keys())
    print(count)
    for k, v in strategy_dict.items():
        if 'adam' not in k:
            print(k,v)

    inference_strategy_local_dir = "/cache/7b_inference_strategy_mp1_dp1.ckpt"
    mox.file.copy(src_url=src_obs_strategy_inference, dst_url=inference_strategy_local_dir)


    # # download 100B ckeckpoint for rank 512  ########### FA_RoPE_Wrong_ENV
    # ckpt_obs_path = "obs://pcl-2023/ckpts/stage1_rope_fa_new"
    # ckpt_local_path = "/cache/ckpt_tmp"
    # ckpt_name = 'mind_yizx_7b'
    # ckpt_step = '6000'
    # train_nodes = 208

    # # right_noTopQ_noFA_RoPE_7B
    # ckpt_obs_path = "obs://pcl-2023/ckpts/stage1_oldenv_pp2"
    # ckpt_local_path = "/cache/ckpt_tmp"
    # ckpt_name = 'oldProj_mind_7b_pp2'
    # ckpt_step = '3500'
    # train_nodes = 208

    ckpt_local_path = args.ckpt_local_path
    ckpt_obs_path = args.ckpt_obs_path
    ckpt_name = args.ckpt_name
    ckpt_step = args.ckpt_step

    if not os.path.exists(ckpt_local_path):
        os.makedirs(ckpt_local_path, exist_ok=True)

    def download_ckpt(ckpt_name_prefix, rank_id):
        if not os.path.exists(f"{ckpt_local_path}/rank_{rank_id}/"):
            os.makedirs(f"{ckpt_local_path}/rank_{rank_id}/", exist_ok=True)

        mox.file.copy(src_url=f"{ckpt_obs_path}/rank_{rank_id}/{ckpt_name_prefix}",
                               dst_url=f"{ckpt_local_path}/rank_{rank_id}/{ckpt_name}_{rank_id}-{ckpt_step}_4.ckpt")

        if rank_id % 8 == 0:
            print(f"Download ckpt for rank {rank_id} done...")


    transform_rank = 0
    print(f"rank {rank_id}, transform_rank {transform_rank}...")
    rank_list = ms.rank_list_for_transform(transform_rank,
                                           src_strategy_file=train_strategy_dir,
                                           dst_strategy_file=None)
    # rank_list = rank_list[::-1]
    print(rank_list)

    # download 100B ckeckpoint of rank 0-512
    time0 = time.time()
    print("主进程开始执行>>> pid={}".format(os.getpid()))
    ps = Pool(processes=8)
    for i in rank_list:
        ckpt_name_prefix = f"{ckpt_name}_{i}-{ckpt_step}_4.ckpt"
        res = ps.apply_async(download_ckpt, args=(ckpt_name_prefix, i))
        time.sleep(i % 8)
    ps.close()
    ps.join()
    print("主进程终止")
    ckpt_name_dir = f"/cache/ckpt_tmp/rank_207/{ckpt_name}_{207}-{ckpt_step}_4.ckpt"
    if not os.path.isfile(ckpt_name_dir):
        time.sleep(20)
    print(os.listdir("/cache/ckpt_tmp"))

    # for i in range(train_nodes):
    for i in rank_list:
        ckpt_name_dir = f"/cache/ckpt_tmp/rank_{i}/{ckpt_name}_{i}-{ckpt_step}_4.ckpt"
        if not os.path.isfile(ckpt_name_dir):
            print(f"{ckpt_name_dir} Not found !")
            download_ckpt(f"{ckpt_name}_{i}-{ckpt_step}_4.ckpt", i)
            print(f"{ckpt_name_dir} download again...")
    print(f"copy ckpt cost : {time.time()-time0} s")

    # transformer ckpt
    checkpoint_file_map = {}
    src_checkpoints_dir = ckpt_local_path
    dst_checkpoints_dir = "/cache/ckpt_inference/"
    if not os.path.exists(dst_checkpoints_dir):
        os.makedirs(dst_checkpoints_dir, exist_ok=True)

    # rank_list = [0, 1, 2, 3, 4, 5, 6, 7, 104, 105, 106, 107, 108, 109, 110, 111]
    for rank_id2 in rank_list:
        checkpoint_file_map[rank_id2] = f"{src_checkpoints_dir}/rank_{rank_id2}/{ckpt_name}_{rank_id2}-{ckpt_step}_4.ckpt"
    print(checkpoint_file_map)

    save_checkpoint_path = f"{dst_checkpoints_dir}/rank_{transform_rank}.ckpt"
    ms.transform_checkpoint_by_rank(transform_rank, checkpoint_file_map, save_checkpoint_path,
                                    src_strategy_file=train_strategy_dir, dst_strategy_file=None)

    # load and save ckpt
    param_dict = ms.load_checkpoint(save_checkpoint_path)
    parame_list = []
    for k, v in param_dict.items():
        if 'accu_grads' not in k and 'adam_' not in k:
            print(k, v)
            #param = Parameter(Tensor(v.data.asnumpy().astype(np.float16), dtype=mstype.float16), name=k)

            param = Tensor(v.data, dtype=mstype.float16)
            parame_list.append({"name": k, "data": param})

    print(f"Save {save_checkpoint_path.split('/')[-1]} for params {len(parame_list)}")
    save_checkpoint_path_2 = dst_checkpoints_dir + "mind_" + save_checkpoint_path.split('/')[-1]
    ms.save_checkpoint(parame_list, save_checkpoint_path_2)

    # upload ckpt
    obs_save_path = args.obs_save_path      #"obs://pcl-2023/merged_7B_ckpts/right_noFA_RoPE_noTopQ_Mind7b/"
    if not mox.file.exists(obs_save_path):
        mox.file.make_dirs(obs_save_path)
    try:
        mox.file.copy(src_url=save_checkpoint_path_2, dst_url=obs_save_path+f"merged_7b_{ckpt_step}-4_new.ckpt")
    except:
        time.sleep(3)
        mox.file.copy(src_url=save_checkpoint_path_2, dst_url=obs_save_path+f"merged_7b_{ckpt_step}-4_new.ckpt")










