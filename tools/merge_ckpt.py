import mindspore as ms
from mindspore.common import Parameter
from mindspore.common.tensor import Tensor
import moxing as mox
import numpy as np
import mindspore.common.dtype as mstype
import argparse

def transform_ckpt(args):

    
    print('..........Downloading ckpt form obs.........')
    for rank in range(0,args.rank):
        ckpt_obs_path = args.restore_checkpoint_bucket_dir
        ckpt_name = args.restore_ckpt_name_prefix.replace('*', str(rank))
        obs_path = ckpt_obs_path + f'rank_{rank}/' + ckpt_name
        local_path = './sft_model/' + f'rank_{rank}/' + f'checkpoint_{rank}.ckpt'
        print(f'.....Copying {obs_path} to {local_path}.....')
        mox.file.copy(obs_path, local_path)
        if args.remove_obs_ckpt:
            mox.file.remove(obs_path)
    
    print('..........Abandoning optimizers.........')
    for rank in range(0,args.rank):
        src_ckpt_local_path = './sft_model/' + f'rank_{rank}/' + f'checkpoint_{rank}.ckpt'
        para_dict = ms.load_checkpoint(src_ckpt_local_path)
        parame_list = []
        for k, v in para_dict.items():
            if 'accu_grads' not in k and 'adam' not in k:
                param = Parameter(Tensor(v.data.asnumpy().astype(np.float16), dtype=mstype.float16), name=k)
                parame_list.append({"name": k, "data": param})
        dst_ckpt_local_path = './sft_model/' + f'rank_{rank}/' + f'checkpoint_a_{rank}.ckpt'
        ms.save_checkpoint(parame_list, dst_ckpt_local_path)
    
    
    print('..........Transforming ckpt.........')
    #ms.transform_checkpoints('./sft_model/', './sft_model_128rank/', "checkpoint_a_", args.stragety, 'mpc_mind_1080b_train_128rank_mp2.ckpt')
    #ms.transform_checkpoints('./sft_model_128rank/', './sft_model_1rank/', "checkpoint_a_", 'mpc_mind_1080b_train_128rank_mp2.ckpt', None)
    ms.transform_checkpoints('./sft_model/', './sft_model_1rank/', "checkpoint_a_", args.stragety, None)

    # para_dict = ms.load_checkpoint('./sft_model_1rank/rank_0/checkpoint_0.ckpt')

    
    
    
    print('..........Copying new ckpt to obs.........')
    mox.file.copy('./sft_model_1rank/rank_0/checkpoint_a_0.ckpt', args.obs_ckpt_save_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--restore_checkpoint_bucket_dir', type=str)
    parser.add_argument('--restore_ckpt_name_prefix', type=str)
    parser.add_argument('--local_ckpt_save_name', type=str)
    parser.add_argument('--obs_ckpt_save_name', type=str)
    parser.add_argument('--remove_obs_ckpt', action='store_true')
    parser.add_argument('--stragety', type=str, required=True)
    parser.add_argument('--rank', type=int, default=16, required=True)
    
    args = parser.parse_args()
    print(args)
    transform_ckpt(args)