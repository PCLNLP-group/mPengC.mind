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
PengChengMind predict run
"""
import json
import os
import requests
import datetime
import glob

import numpy as np
from tqdm import tqdm

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore as ms
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.train.model import Model
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint
from mindspore.nn.transformer.transformer import TransformerOpParallelConfig

from src.generate import get_scores
from src.pengcheng_mind_7B import EvalNet, PengChengMindModel, EvalNet_200B
from src.pengcheng_mind_config import set_parse, PengChengMindConfig
from src.utils import get_args

from mindspore.common import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.utils import download_ckpt_from_obs


project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
hf_project = lambda *x: os.path.join(_PROJECT_ROOT, *x)

os.system("cat /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/mindspore/.commit_id")

def restore_checkpoint(args_param, network, cache_url='/cache/Ckpt/'):
    r"""
    Load checkpoint process.
    """
    restore_ranks = D.get_rank()
    print("======start single checkpoint", flush=True)
    ckpt_name = os.path.join(cache_url, f"rank_{restore_ranks}.ckpt")

    if not ckpt_name:
        print(f"There is no ckpt file in {ckpt_name}, "
              f"current ckpt_files found is {ckpt_name} "
              f"with pattern {ckpt_name}, so skip the loading.")

    time_stamp = datetime.datetime.now()
    print(f"time stamp {time_stamp.strftime('%Y.%m.%d-%H:%M:%S')} pre trained ckpt model {ckpt_name} loading",
          flush=True)
    # Load checkpoint files latest file
    print(f'Start to load from {ckpt_name}')
    param_dict = load_checkpoint(ckpt_name)
    # for k, v in param_dict.items():
    #     print("rank: ", restore_ranks, k)
    load_param_into_net(network, param_dict, strict_load=False)

def set_auto_parallel_context(args_opt):
    """Set the auto parallel context"""
    rank = 0
    device_num = 1
    context.reset_auto_parallel_context()
    # context.set_auto_parallel_context(
    #     strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_save_file=f'/cache/strategy_{rank}.ckpt',
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    return rank, device_num

def load_model(args_opt):
    r"""
     The main function for load model
    """
    context.set_context(mode=context.GRAPH_MODE)
    # Set parallel context
    rank, device_num = set_auto_parallel_context(args_opt)

    context.set_context(variable_memory_max_size="30GB")
    context.set_context(save_graphs=2,
                        save_graphs_path="/cache/graphs_of_device_id_" + str(rank),
                        device_target=args_opt.device_target)

    strategy_local_file = f"/cache/inference_strategy_100b_d8_mp8_dp1-{rank}.ckpt"
    ms.set_auto_parallel_context(strategy_ckpt_save_file=strategy_local_file)

    if args_opt.eval_task:
        use_past = False
    else:
        use_past = args_opt.use_past
    print('local_rank:{}, start to run...'.format(rank), flush=True)

    # Set model property, rewrite the model parallel
    if device_num < args_opt.op_level_model_parallel_num:
        print(f"The op_level_model_parallel_num {args_opt.op_level_model_parallel_num} is smaller than the device num，"
              f"so change it to the {device_num}", flush=True)
        args_opt.op_level_model_parallel_num = device_num
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / (model_parallel_num*args_opt.stage_num))
    micro_batch_interleaved = args_opt.micro_batch_interleaved

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  vocab_emb_dp=False,
                                                  recompute=False)
    # add sequence_parallel
    parallel_config.sequence_parallel = args_opt.sequence_parallel
    # add select_recompute
    parallel_config.select_recompute = args_opt.select_recompute

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    # Now only support single batch_size for predict
    if args_opt.run_type == "predict":
        batch_size = 1

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
                                 use_past=False,
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

    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    # Define network
    pengcheng_mind = PengChengMindModel(config)
    eval_net = EvalNet_200B(pengcheng_mind, pad_token=args_opt.padding_id, seq_length=args_opt.seq_length)
    eval_net.set_train(False)

    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif args_opt.eval_task:
        # Compiling only needs the shape
        predict_layout = model_predict.infer_predict_layout(inputs_np, inputs_np)
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_true, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    #local_ckpt_path = '/cache/pretrained_7b_fp16.ckpt'
    cache_url = '/cache/Ckpt/'
    local_ckpt_url = '/cache/Ckpt/tmp.ckpt'
    if args_opt.pre_trained:
        # STEP = 500
        # download_OneCKPT_from_obs(
        #     obs_ckpt_url=f'obs://pcmind7b/ckpts/PengCheng-Mind-7B-1000_tokens_fp16.ckpt',
        #     local_ckpt_url=local_ckpt_path,
        #     rank=rank)
        # param_dict = load_checkpoint(local_ckpt_path)
        # load_param_into_net(eval_net, param_dict)
        # print("================load param ok=================", flush=True)

        #######################################################################

        # #### method1: loading [.npy file]
        # if not args_opt.offline:
        #     import moxing as mox
        #     gpu_model_npy_obspath = "obs://test-zy/merged_ckpt_pt.npy"
        #     local_npy_path = "/cache/ckpt.npy"
        #     mox.file.copy(gpu_model_npy_obspath, local_npy_path)
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
        #     # #######################################################################
        #     # # saving npu-ckpt for yunnao2-obs
        #     # tokens_flag = "227B"
        #     # save_local_ckpt = "pcmind_new7B_{}tks.ckpt".format(tokens_flag)
        #     # ms.save_checkpoint(pengcheng_mind, save_local_ckpt)
        #     # if not args_opt.offline:
        #     #     if config.param_init_type == mstype.float32:
        #     #         mox.file.copy(save_local_ckpt,
        #     #                       "obs://"
        #     #                       "fp32-{}".format(save_local_ckpt))
        #     #     else:
        #     #         mox.file.copy(save_local_ckpt,
        #     #                       "obs://"
        #     #                       "fp16-{}".format(save_local_ckpt))
        #     #     print("================save param to OBS ok=================", flush=True)

        #### method2: loading [.ckpt file]
        if not args_opt.offline:
            import moxing as mox
            # gpu_model_npy_obspath = "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/" \
            #                         "fp16-pcmind_new7B_227Btks.ckpt"
            # gpu_model_npy_obspath = "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/" \
            #                         "new-fp16-iter47200-pcmind_new7B_227Btks.ckpt"
            # gpu_model_npy_obspath = "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/" \
            #                         "new-fp32-iter47200-pcmind_new7B_227Btks.ckpt"
            '''
            gpu_model_npy_obspath = "obs://pcl-verify/yizx/2023_7b_ckpts/mind7b_to_ckpts/new7b_gpu_npy2msckpt/" \
                                    "new-fp16-iter168000-pcmind_new7B_660Btks.ckpt"
                                    # "new-fp16-v660sft-pcmind_new7B_660Btks.ckpt"
            '''
            obs_ckpt_url = args_opt.restore_checkpoint_bucket_dir
            mox.file.copy(src_url=obs_ckpt_url, dst_url=local_ckpt_url)
            print(f"Rank {rank} download ckpt succeed!", flush=True)
            param_dict = load_checkpoint(local_ckpt_url)
            load_param_into_net(eval_net, param_dict)
            print("================load param ok=================", flush=True)
            # #######################################################################

    return model_predict, config


def run_predict(model_predict, config, args_opt):
    """run predict"""
    from src.generate import generate, generate_increment, generate_100b, generate_100b_task
    import time

    D.init()
    rank_id = D.get_rank()
    device_num = D.get_group_size()

    #generate_func = generate_increment if config.use_past else generate
    generate_func = generate_100b_task
    # Define tokenizer
    from transformers import AutoTokenizer
    pcmind7b_tokenizer_dir = hf_project("tokenizer/baichuan2")
    print(pcmind7b_tokenizer_dir)
    print(os.listdir(pcmind7b_tokenizer_dir))
    vocab_path = pcmind7b_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(vocab_path, trust_remote_code=True)
    EOT = tokenizer.eos_token_id
    PAD = tokenizer.unk_token_id

    test_sample = ["中国的四大发明有哪些？\n 请简要回答",
                   "请详细的介绍中国的四大发明：\n 请简要回答",
                   "请从来源、时期、发明人、演变过程、对世界格局的影响等方面详细的介绍中国的”四大发明“。\n 请简要回答",
                   "四大发明来源于哪个国家？请用专业的知识给大家介绍。\n 请简要回答",
                   "假如你是知名大学研究汉语言文学的教授，请您给大家讲解’中国四大发明‘，\n 请简要回答",
                   "请把‘We introduce Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. Preliminary evaluation using GPT-4 as a judge shows Vicuna-13B achieves more than 90% quality of OpenAI ChatGPT and Google Bard while outperforming other models like LLaMA and Stanford Alpaca in more than 90%* of cases. The cost of training Vicuna-13B is around $300. ‘翻译为中文：",
                   "以’风、雪‘为基调，写一首七言绝句。",
                   # "以’风、雪‘为关键词，根据你自己此时的心境，写一首四句话的七言绝句：",
                   # "假设你是一个唐代的诗人，以’风、雪‘为关键词，根据你自己此时的心境，写一首七言绝句：",
                   # "你读了’《咏雪》唐·白居易\n已讶衾枕冷，\n复见窗户明。\n夜深知雪重，\n时闻折竹声。‘，即兴而发，以’风、雪‘为主题，写了一首七言绝句：",
                   # "北极熊和灰熊的科学名称分别是什么？",
                   # "北极熊和灰熊的科学名称分别是什么？请从科学研究的角度详细介绍一下这两个物种。",
                   # "我要做蛋炒饭，请按步骤写一个详细的制作教程：",
                   # "中国有哪些好玩的城市？请选取其中一个城市详细介绍当地景点和游玩攻略。",
                   ]

    time_all = []
    time_word = []
    for idx, sample in enumerate(test_sample):

        # Tokenize input sentence to ids
        start_sentence = tokenizer.encode(sample, add_special_tokens=False)
        # start_sentence = tokenizer.encode(sample)
        input_ids = np.array(start_sentence).reshape(1, -1)
        time_start = time.time()
        output_ids = generate_func(model_predict,
                                   input_ids,
                                   args_opt)
        time_use = time.time() - time_start
        output_list = output_ids[input_ids.shape[-1]:].tolist()
        output_samples = tokenizer.decode(output_list, skip_special_tokens=True)
        time_all.append(time_use/max(1, len(output_ids[len(input_ids):].tolist())))
        time_word.append(time_use/max(1, len(output_samples)))
        print(f"----------------------{idx}------------------------")
        print(f"Input is: {sample}\n")
        print(f'Output is: {output_samples}\n')
        print(f"Average time: {time_use / len(output_list)} s/token")

def run_predict_translate(model_predict, config, args_opt):
    """run translate_predict"""
    from src.generate import generate, generate_increment, generate_100b, generate_100b_task
    import time

    D.init()
    rank_id = D.get_rank()
    device_num = D.get_group_size()

    #generate_func = generate_increment if config.use_past else generate
    generate_func = generate_100b_task
    # Define tokenizer
    from transformers import AutoTokenizer
    pcmind7b_tokenizer_dir = hf_project("tokenizer/baichuan2")
    print(pcmind7b_tokenizer_dir)
    print(os.listdir(pcmind7b_tokenizer_dir))
    vocab_path = pcmind7b_tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(vocab_path, trust_remote_code=True)
    EOT = tokenizer.eos_token_id
    PAD = tokenizer.unk_token_id
    print(EOT, PAD)



    test_text = 'What are the four great inventions of China?'
    #input_text = f'en\t{test_text} translate to \tzh'
    input_ids = np.array(tokenizer.encode(test_text, add_special_tokens=False)).reshape(1, -1)
    output_ids = generate_func(model_predict,
                               input_ids,
                               args_opt,
                               top_p=args_opt.top_p,
                               top_k_num=args_opt.top_k_num,
                               max_generate_length=args_opt.max_generate_length,
                               duRepeate=args_opt.duRepeate)  # args_opt.duRepeate
    output_txt = tokenizer.decode(output_ids[input_ids.shape[-1]:].tolist(), skip_special_tokens=True)
    print(f"Input is: {test_text}\n")
    print(f'Output is: {output_txt}\n')
    print("####################"*5)
    
    '''
    langs_54_101 = {'en': 'eng', 'ru': 'rus', 'es': 'spa', 'ar': 'ara', 'ja': 'jpn',
            'kk': 'kaz', 'mn': 'mon', 'cs': 'ces', 'fr': 'fra', 'de': 'deu',
            'pt': 'por', 'sr': 'srp', 'it': 'ita', 'tr': 'tur', 'bg': 'bul',
            'pl': 'pol', 'ro': 'ron', 'hu': 'hun', 'nl': 'nld', 'he': 'heb',
            'hr': 'hrv', 'id': 'ind', 'fa': 'fas', 'da': 'dan', 'no': 'nob',
            'fi': 'fin', 'el': 'ell', 'ko': 'kor', 'uk': 'ukr', 'th': 'tha',
            'bn': 'ben', 'ms': 'zsm', 'ta': 'tam', 'vi': 'vie', 'hi': 'hin',
            'sk': 'slk', 'ur': 'urd'}
    '''
    
    langs_54_101 = {'en': '英语', 'ru': '俄语', 'es': '西班牙语', 'ar': '阿拉伯语', 'ja': '日语',
            'kk': '哈萨克语', 'mn': '蒙语', 'cs': '捷克语', 'fr': '法语', 'de': '德语',
            'pt': '葡萄牙语', 'sr': '塞尔维亚语', 'it': '意大利语', 'tr': '土耳其语', 'bg': '保加利亚语',
            'pl': '波兰语', 'ro': '罗马尼亚语', 'hu': '匈牙利语', 'nl': '荷兰语', 'he': '希伯来语',
            'hr': '克罗地亚语', 'id': '印尼语', 'fa': '波斯语', 'da': '丹麦语', 'no': '挪威语',
            'fi': '芬兰语', 'el': '希腊语', 'ko': '韩语', 'uk': '乌克兰语', 'th': '泰语',
            'bn': '孟加拉语', 'ms': '马来语', 'ta': '泰米尔语', 'vi': '越南语', 'hi': '印地语',
            'sk': '斯洛伐克语', 'ur': '乌尔都语'}
    prompt_templates = {'en':'Please translate {input} to {tgt_lang}', 
                        'de':'Bitte übersetzen Sie {input} in {tgt_lang}',
                        'fr':'Veuillez traduire {input} en {tgt_lang}',
                        'es':'Traduzca {input} al {tgt_lang}',
                        'ar':'يرجى ترجمة {input} إلى {tgt_lang}',
                        'ru':'Пожалуйста, переведите {input} на {tgt_lang}',
                        'ja':'{input} を {tgt_lang} に翻訳してください',
                        'zh':'请把{input}翻译为{tgt_lang}'}
    lang_name = {'en':{'zh':'Chinese'},
                 'de':{'zh':'chinesisch'},
                 'fr':{'zh':'Chinois'},
                 'es':{'zh':'Chino'},
                 'ar':{'zh':'الصينية'},
                 'ru':{'zh':'китайский'},
                 'ja':{'zh':'中国語'},
                 'zh':{'en':'英语', 'de':'德语', 'fr':'法语', 'es':'西班牙语', 'ar':'阿拉伯语', 'ru':'俄语', 'ja':'日语'}}
    
    flores_101 = "/home/ma-user/modelarts/user-job-dir/pengchengmind-7b-ms20/inference/mPC_flores101-51_devtest.json"
    with open(flores_101, 'r', encoding="utf-8") as fl:
        flores = json.loads(fl.read())
    
    lang = list(langs_54_101.keys())[args_opt.lang_idx]
    print(f'process for language: {lang}')

    time_word = []
    time_token = []

    result = {}
    result[f"{lang}-zh"] = {}
    result[f"zh-{lang}"] = {}
    result[f"{lang}-zh"] = {}
    result[f"zh-{lang}"] = {}
    local_result = "/home/ma-user/modelarts/user-job-dir/pengchengmind-7b-ms20/result.json"
    dir_result = args_opt.restore_checkpoint_bucket_dir.split('/')[-1].split('-7B_')[-1].split('.ckpt')[0]
    upload_obs_path = f"obs://mpcmind/majy/PC_Sailboat.Translation-100/inference/mPCM-7B_result//{dir_result}_result-2en2lang/{lang}-zh_output.json"
    import moxing as mx
    for idx in range(1012):
        src_text = flores[str(idx)][lang]
        dst_text = flores[str(idx)]['zh']

        #input_text1 = f'{lang}\t{src_text} translate to \ten'
        input_text1 = prompt_templates[lang].format(input=src_text, tgt_lang=lang_name[lang]['zh'])
        input_ids1 = np.array(tokenizer.encode(input_text1)).reshape(1, -1)
        time1 = time.time()
        output_ids1 = generate_func(model_predict,
                                   input_ids1,
                                   args_opt,
                                   top_p=args_opt.top_p,
                                   top_k_num=args_opt.top_k_num,
                                   max_generate_length=args_opt.max_generate_length,
                                   duRepeate=args_opt.duRepeate)  # args_opt.duRepeate
        output_txt1 = tokenizer.decode(output_ids1[input_ids1.shape[-1]:].tolist(), skip_special_tokens=True)
        #----------------------------------------------------------------------------------------------------
        #input_text11 = f'en\t{output_txt1} translate to \tzh'
        #input_text11 = f'请把“{output_txt1} ”翻译成中文。'
        #input_ids11 = np.array(tokenizer.encode(input_text11)).reshape(1, -1)
        #output_ids11 = generate_func(model_predict,
        #                           input_ids11,
        #                           args_opt,
        #                           top_p=args_opt.top_p,
        #                           top_k_num=args_opt.top_k_num,
        #                           max_generate_length=args_opt.max_generate_length,
        #                           duRepeate=args_opt.duRepeate)  # args_opt.duRepeate
        #output_txt11 = tokenizer.decode(output_ids11[input_ids11.shape[-1]:].tolist(), skip_special_tokens=True)

        result[f"{lang}-zh"][str(idx)] = [src_text, output_txt1, dst_text]
        print(f"=========================================--{idx}--===================================================")
        print(f"Input is: {input_text1}")
        #print(f"Output en is: {output_txt1}")
        print(f'Output is: {output_txt1}')

        #input_text2 = f'zh\t{dst_text} translate to \ten'
        input_text2 = prompt_templates['zh'].format(input=dst_text, tgt_lang=lang_name['zh'][lang])
        input_ids2 = np.array(tokenizer.encode(input_text2)).reshape(1, -1)
        output_ids2 = generate_func(model_predict,
                                   input_ids2,
                                   args_opt,
                                   top_p=args_opt.top_p,
                                   top_k_num=args_opt.top_k_num,
                                   max_generate_length=args_opt.max_generate_length,
                                   duRepeate=args_opt.duRepeate)  # args_opt.duRepeate
        output_txt2 = tokenizer.decode(output_ids2[input_ids2.shape[-1]:].tolist(), skip_special_tokens=True)
        #----------------------------------------------------------------------------------------------
        #input_text22 = f'en\t{output_txt2} translate to \t{lang}'
        #input_text22 = f'请把“{output_txt2} ”翻译成{langs_54_101[lang]}。'
        #input_ids22 = np.array(tokenizer.encode(input_text22)).reshape(1, -1)
        #output_ids22 = generate_func(model_predict,
        #                           input_ids22,
        #                           args_opt,
        #                           top_p=args_opt.top_p,
        #                           top_k_num=args_opt.top_k_num,
        #                           max_generate_length=args_opt.max_generate_length,
        #                           duRepeate=args_opt.duRepeate)  # args_opt.duRepeate
        #output_txt22 = tokenizer.decode(output_ids22[input_ids22.shape[-1]:].tolist(), skip_special_tokens=True)
        result[f"zh-{lang}"][str(idx)] = [dst_text, output_txt2, src_text]
        print(f"-----------------------------------------------------------------------------------------------------")
        print(f"Input is: {input_text2}")
        #print(f"Output en is: {output_txt2}")
        print(f'Output is: {output_txt2}')

        if idx %23 ==0 and idx !=0:
            result['langs_average-token'] = np.average(time_token[2:])
            result['en_average-word'] = np.average(time_word[2:])
            with open(local_result, 'w', encoding="utf-8") as fl2:
                fl2.write(json.dumps(result, ensure_ascii=False))
            mx.file.copy(local_result, upload_obs_path)
            print(f"Index {idx} result upload success...")
    result['langs_average-token'] = np.average(time_token[2:])
    result['en_average-word'] = np.average(time_word[2:])
    with open(local_result, 'w', encoding="utf-8") as fl2:
        fl2.write(json.dumps(result, ensure_ascii=False))
    mx.file.copy(local_result, upload_obs_path)
    print("result upload success...")


def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config = load_model(opt)
    run_predict_translate(model_predict, config, opt)

if __name__ == "__main__":
    main()






































