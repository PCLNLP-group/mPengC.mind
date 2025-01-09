# Copyright 2024 PCLNLP_GROUP
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import glob
import json
import os
import re
import random
import multiprocessing
import time
from multiprocessing import Pool, current_process, Process
import numpy as np

import sys
sys.setrecursionlimit(10000)

try:
    from transformers import GPT2Tokenizer
except ModuleNotFoundError:
    print("module 'transformers' not installed.")

from mindspore.mindrecord import FileWriter

#python src/pre_process_chinese.py --input_glob= --output_file= --tokenizer= --SEQ_LEN= --file_batch_size=

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='openwebtext')
###########################
parser.add_argument('--input_glob', type=str, default='/home/pretrain/*.txt')
#############################
parser.add_argument('--output_file', type=str,
                     default='/home/mindrecord/')

parser.add_argument('--file_partition', type=int, default=50)
parser.add_argument('--file_batch_size', type=int, default=512)
parser.add_argument('--num_process', type=int, default=96)
parser.add_argument('--worker_size', type=int, default=6)
parser.add_argument('--tokenizer', type=str, default='baichuan2')
parser.add_argument('--SEQ_LEN', type=int, default=4097)
parser.add_argument('--rankOfCluster', type=str, default='0of1')

args = parser.parse_args()
SEQ_LEN = args.SEQ_LEN  # the length of sample
working_dir = os.getcwd()

if args.tokenizer == 'baichuan2':
    from transformers import AutoTokenizer
    vocab_path = working_dir + '/tokenizer/baichuan2'
    print(vocab_path)
    tokenizer = AutoTokenizer.from_pretrained(vocab_path, trust_remote_code=True)

    EOT = tokenizer.eos_token_id
    PAD = tokenizer.unk_token_id
############################

def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i,e in enumerate(listTemp):
        twoList[i%n].append(e)
    return twoList

def package_file(it, n):
    """ package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch

def padding_eot(chunk):
    pad = [PAD] * (SEQ_LEN - len(chunk))
    chunk.extend(pad)
    return chunk

def get_input_data_batch_slice_map(input_ids, eod_id):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_id: the id for <EOD>
    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((len(input_ids), seq_length))
    # batch_attention_mask = np.ones((len(input_ids), seq_length, seq_length))

    batch_eod_index = []
    # Loop through batches
    for bs_i, _ in enumerate(range(len(input_ids))):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        # batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find eod_of_document
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            index = eod_index[i]
            # Reset position_ids and attention_mask considering EOD
            # batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_position_ids

def tokenize_text(iterator):
    """ tokenize text dataset"""
    content = []
    for file_path in iterator:
        print('fuck it ',file_path)
        if os.path.getsize(file_path) == 0:
            continue

        if '.txt' not in file_path:
            print(file_path, "jump...")
            continue
        else:
            print('file_open is ',file_path)
            with open(file_path, 'r', encoding='utf-8') as fl:
                fi = fl.read().split('\n\n')
                for idx, info_one in enumerate(fi):
                    try:
                        para = eval(info_one)['text']
                        flag = eval(info_one)['flag']
                        if flag == 'parallel':
                           src, tag = eval(info_one)["src_lang"], eval(info_one)["tgt_lang"]
                           src_id = tokenizer.encode(src+" content :", add_special_tokens=False)
                           tag_id = tokenizer.encode(tag+" content :", add_special_tokens=False)
                           translate_id = tokenizer.encode('can be translated to', add_special_tokens=False)
                           src_data = para["src"]
                           tag_data = para["tag"].strip()
                           tokenized_id1 = tokenizer.encode(''+src_data, add_special_tokens=False)
                           tokenized_id2 = tokenizer.encode(''+tag_data, add_special_tokens=False)
                           ratio = len(tokenized_id1) / len(tokenized_id2)

                           if 6>=ratio>=0.167:
                              assembly_1 = src_id + tokenized_id1 + translate_id + tag_id + tokenized_id2
                              assembly_2 = tag_id + tokenized_id2 + translate_id + src_id + tokenized_id1
                              
                              if len(assembly_1) < (SEQ_LEN-1):
                                 content.append(assembly_1 + [EOT]) 
                              
                              if len(assembly_2) < (SEQ_LEN-1):
                                 content.append(assembly_2 + [EOT])

                        elif flag == 'mono':
                            lang = eval(info_one)['lang']
                            lang_id = tokenizer.encode(lang+" content :", add_special_tokens=False)
                            tokenized_id = tokenizer.encode(''+para, add_special_tokens=False)
                            assembly = lang_id + tokenized_id
                            if len(assembly) <= (SEQ_LEN-1):
                               content.append(assembly + [EOT])

                        elif len(para)>=6:
                            tokenized_id = tokenizer.encode(''+para, add_special_tokens=False)
                            if len(tokenized_id) < (SEQ_LEN-1):
                                content.append(tokenized_id + [EOT])
                            else:
                                ratio = int(len(tokenized_id)/(SEQ_LEN-2))
                                len_r = SEQ_LEN-2
                                for mn in range(ratio):
                                    content.append(tokenized_id[mn*len_r:(mn+1)*len_r] + [EOT])
                                if len(tokenized_id[ratio*len_r:]) >= 80:
                                    content.append(tokenized_id[ratio*len_r:] + [EOT])
                                    
                    except Exception as error:
                        print("eval info Error, jump...", error)

    random.shuffle(content)
    content_new = []
    for i in content:
        content_new += i

    for chunk in chunks(content_new, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            batch_position_ids = get_input_data_batch_slice_map(
                np.array([sample['input_ids']], dtype=np.int32),
                eod_id=EOT)
            sample['position_id'] = np.array(batch_position_ids[0], dtype=np.int32)
            yield sample
        else:
            sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
            batch_position_ids = get_input_data_batch_slice_map(
                np.array([sample['input_ids']], dtype=np.int32),
                eod_id=EOT)
            sample['position_id'] = np.array(batch_position_ids[0], dtype=np.int32)
            yield sample

def task_unit(iterator, mindrecord_filename):
    batch_size = 1024
    schema = {"input_ids": {"type": "int32", "shape": [-1]},
              "position_id": {"type": "int32", "shape": [-1]},}
    writer = FileWriter(file_name = mindrecord_filename, shard_num=1)
    writer.add_schema(schema)
    writer.open_and_set_header()
    
    time0 = time.time()
    item_iter = tokenize_text(iterator)
    count = 0
    while 1:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            random.shuffle(data_batch)
        except Exception as error:
             print(error)
        if len(data_batch) > 0 and isinstance(data_batch, list):
           writer.write_raw_data(data_batch)
        else:
           time1 = time.time()
           print("==>> End to create mindrecord file: {}".format(mindrecord_filename), flush=True) 
           print("==>> create mindrecord file {} time use: {}".format(mindrecord_filename, time1-time0), flush=True)
           break
    writer.commit()    

def get_total_size(input_files):
    t_size = 0
    for f in input_files:
        f_size = os.path.getsize(f)
        t_size += f_size
        if 'corpus' in f:
           t_size += f_size
    return t_size

if __name__ == '__main__':

    if args.dataset_type == 'openwebtext':
        input_files = list(glob.iglob(args.input_glob))
        input_files.sort()
        random.seed(100)
        random.shuffle(input_files)

        file_size_list = []
        for f in input_files:
            size = os.path.getsize(f)
            file_size_list.append((f, size))
        sorted_file_list = sorted(file_size_list, key=lambda x:x[1], reverse=True)
        avg_file_size = get_total_size(input_files) / args.num_process
        
        process_file_list = [[] for _ in range(args.num_process)]
        process_size_list = [0] * args.num_process
        for i in range(args.num_process):
            item = sorted_file_list[args.num_process - i - 1]
            process_file_list[i].append(item[0])
            process_size_list[i] += item[1]
        pid = 0
        for item in sorted_file_list[args.num_process:]:
            if process_size_list[pid] > avg_file_size:
               pid +=1
            process_file_list[pid].append(item[0])
            process_size_list[pid] += item[1]
        
        whole_size = 0
        for pid in range(args.num_process):
            print(f"pid: {pid}, file number: {len(process_file_list[pid])}, size: {process_size_list[pid] / 1024 / 1024} MB")
            whole_size += process_size_list[pid]
        print('input_files into Process: ',len(input_files), len(process_file_list))
        print(f"whole size: {whole_size/1024/1024} MB")
        
        mindrecord_filenames = [f'{args.output_file}_{str(i)}.mindrecord' for i in range(args.num_process)]
        begin = time.time()
        process_list = []
        for i in range(args.num_process):
            p = Process(target=task_unit, args=(process_file_list[args.num_process-i-1], mindrecord_filenames[i]))
            process_list.append(p)
            if len(process_list) == args.worker_size:
                for pi in process_list:
                    pi.start()
                for pi in process_list:
                    pi.join()
                process_list.clear()
        if len(process_list) !=0:
           for p in process_list:
               p.start()
           for p in process_list:
               p.join()
        end = time.time()
        print(f"read time: {end - begin} s", flush=True)
        
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type))
