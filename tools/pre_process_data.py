# Copyright 2020 Huawei Technologies Co., Ltd
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
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import glob
import json
import os
import re
import random
from multiprocessing import Pool, current_process
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
# parser.add_argument('--input_glob1', type=str, default='/userhome/dataset/chinese_txt/**/*.txt')
parser.add_argument('--input_glob', type=str, default='/userhome/dataset/chinese_txt/webtext2019zh/*.txt')
#############################
parser.add_argument('--output_file', type=str,
                    default='/userhome/dataset/chinese_txt/webtext2019zh_mindrecord/webtext2019zh_EOT30000_mindrecord')
parser.add_argument('--file_partition', type=int, default=3)
parser.add_argument('--file_batch_size', type=int, default=300) #不影响结果, 影响打乱程度
parser.add_argument('--num_process', type=int, default=100)#96
parser.add_argument('--tokenizer', type=str, default='baichuan2')
parser.add_argument('--SEQ_LEN', type=int, default=4097)
parser.add_argument('--rankOfCluster', type=str, default='0of1')

args = parser.parse_args()

SEQ_LEN = args.SEQ_LEN  # the length of sample

# EOT = 50256  # id of endoftext
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

working_dir = os.getcwd()

if args.tokenizer == 'llama':
    from transformers import LlamaTokenizer
    vocab_file = working_dir + '/tokenizer/llama_vocab/llama_zh_hf/tokenizer_2.model'
    tokenizer = LlamaTokenizer.from_pretrained(vocab_file)

    EOT = tokenizer.eos_token_id
    PAD = tokenizer.unk_token_id

if args.tokenizer == 'baichuan2':
    from transformers import AutoTokenizer
    vocab_path = working_dir + '/tokenizer/baichuan2'
    tokenizer = AutoTokenizer.from_pretrained(vocab_path, trust_remote_code=True)

    EOT = tokenizer.eos_token_id
    PAD = tokenizer.unk_token_id
############################

# PAD = tokenizer.pad_id
# EOT = tokenizer.eot_id
print('pad id :', PAD)# 128297
print('eot id :', EOT)# 128298
# print('vocab size :', tokenizer.vocab_size)

##-----------------------corpus change-------------------
### pad id : 128297
### eot id : 128298
### vocab size : 128320

langs_ID = {'zh': 128301, 'ko': 128302, 'vi': 128303,
            'de': 128317, 'en': 128318, 'nl': 128132,
            'ms': 128109, 'id': 128110, 'tl': 128111,
            'mn': 128103, 'my': 128104, 'th': 128105, 'lo': 128106, 'km':128107,
            'lt': 128112, 'et': 128113, 'lv': 128133, 'hu': 128115,
            'pl': 128116, 'cs': 128117, 'sk': 128118, 'sl': 128119, 'hr': 128120, 'bs': 128121, 'sr': 128306, 'bg': 128304,
            'mk': 128122, 'ru': 128305, 'uk': 128307, 'be': 128123,
            'sq': 128124, 'el': 128125, 'ka': 128126, 'hy': 128127,
            'ro': 128108, 'fr': 128100, 'es': 128102, 'pt': 128101,
            'fa': 128310, 'he': 128311, 'ar': 128308, 'ps': 128309,
            'tr': 128128, 'kk': 128129, 'uz': 128130, 'az': 128131,
            'hi': 128315, 'ta': 128316, 'ur': 128313, 'bn': 128312, 'si': 128314, 'ne': 128114}

translate_ID = 128300
##--------------------------------------------------------

def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
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


def clean_wikitext(string):
    """ cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" "+chr(176)+" ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def padding_eot(chunk):
    pad = [PAD] * (SEQ_LEN - len(chunk))
    chunk.extend(pad)
    return chunk

def neighbour_mix(content, content_new=[]):
    content_new = content_new
    ceil = []
    del_idx = []
    count = 0
    for idx, i in enumerate(content[:10]):
        # find 
        if len(ceil) + len(i) <= (SEQ_LEN-1):
            ceil.extend(i)
            del_idx.append(i)
        else:
            count += 1
    content_new.append(ceil)
    for i in del_idx:
        content.remove(i)
    if len(content) > 1:
        return neighbour_mix(content, content_new)
    elif len(content) == 1:
        content_new.append(content[0])
        len_data = [len(i) for i in content_new]
        print("***"*3, len(content_new), np.average(len_data))
        return content_new
    else:
        len_data = [len(i) for i in content_new]
        print("***"*3, len(content_new), np.average(len_data))
        return content_new

def neighbour_mix2(content):
    content_new = []
    ceil = []
    # len_c = len(content)
    # flag = True
    for idx, i in enumerate(content):
        if len(ceil) + len(i) <= (SEQ_LEN-1):
            ceil.extend(i)
        else:
            content_new.append(ceil)
            ceil = []
    len_data = [len(i) for i in content_new]
    print("***"*3, len(content_new), np.average(len_data))
    return content_new

def neighbour_mix3(content):
    content_new = []
    ceil = []
    del_ceil = []
    len_c = len(content)
    # flag = True
    for idx, i in enumerate(content):
        # 
        if i not in del_ceil:
            if len(ceil) + len(i) <= (SEQ_LEN-1):
                ceil.extend(i)
            # 
            elif (idx+1)<len_c:
                for j in content[idx+1:idx+50]:
                    if j not in del_ceil:
                        if len(ceil) + len(j) <= (SEQ_LEN-1):
                            ceil.extend(j)
                            del_ceil.append(j)
      
                content_new.append(ceil)
                ceil = []
            # end ...
            else:
                content_new.append(ceil)
                ceil = []
        #else:
        #    del_ceil.remove(i)
    len_data = [len(i) for i in content_new]
    print("***"*3, len(content_new), np.average(len_data))
    return content_new

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

def get_input_data_batch_slice_map3(input_ids, eod_id):
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
    batch_attention_mask = np.ones((len(input_ids), seq_length, seq_length))

    batch_eod_index = []
    # Loop through batches
    for bs_i, _ in enumerate(range(len(input_ids))):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find eod_of_document
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].tolist()

        eod_index.extend([int(0) for i in range(len(eod_index), seq_length//2)])
        batch_eod_index.append(eod_index)
        prev_index = 0
        for index in eod_index:
            if index != 0:
                index = int(index)
                # Reset position_ids and attention_mask considering EOD
                batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
                batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
                prev_index = index + 1
            else:
                break
    batch_eod_index = np.array(batch_eod_index, dtype=np.int32)
    return batch_input_ids, batch_position_ids, batch_attention_mask, batch_eod_index

def tokenize_openwebtext(iterator):
    """ tokenize openwebtext dataset"""
    content = []
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue

        if '.json' not in file_path:
            print(file_path, "jump...")
            continue
        else:
            with open(file_path, 'r', encoding='utf-8') as fl:
                for idx, info_one in enumerate(fl):
                    try:
                        para = eval(info_one)['text']
                        # para = info_one
                        if len(para)>=6:# and para != '\n':
                            #if 'res_' in file_path:
                            #    para = zh_process(para)
                            #tokenized_id = tokenizer.encode(''+para)
                            tokenized_id = tokenizer.encode(''+para, add_special_tokens=False)
                            # langs_id = tokenizer.convert_tokens_to_ids(tokenized_text_langs)
                            if len(tokenized_id) < (SEQ_LEN-1):
                                content.append(tokenized_id + [EOT])
                            else:
                                if 'code_' not in file_path:
                                    content.append(tokenized_id + [EOT])
                                    '''
                                    ratio = int(len(tokenized_id)/(SEQ_LEN-2))
                                    len_r = SEQ_LEN-2
                                    for mn in range(ratio):
                                        content.append(tokenized_id[mn*len_r:(mn+1)*len_r] + [EOT])
                                    if len(tokenized_id[ratio*len_r:]) >= 80:
                                        content.append(tokenized_id[ratio*len_r:] + [EOT])
                                    '''
                                else:
                                    content.append(tokenized_id + [EOT])
                                    #if len(tokenized_id) <= (SEQ_LEN + 2048):
                                    #    content.append(tokenized_id[:SEQ_LEN-2] + [EOT])

                    except Exception as error:
                        print("eval info Error, jump...", error)
        #print(content[:40])
        #print(content[0])
        #print(len(content), content[0])

    # read files batch default 300
    random.shuffle(content)
    #content_new = []
    #content_new = neighbour_mix3(content)
    #random.shuffle(content_new)
    content_new = []
    for i in content:
        content_new += i
        #print(len(content_new), type(content_new[0]))
        #exit()
    for chunk in chunks(content_new, SEQ_LEN):

    #for chunk in content_new:# chunks(content_new, SEQ_LEN):
        # print(type(chunk), type(chunk[0]), len(chunk))
        # print(chunk[0])
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            batch_position_ids = get_input_data_batch_slice_map(
                np.array([sample['input_ids']], dtype=np.int32),
                eod_id=EOT)
            sample['position_id'] = np.array(batch_position_ids[0], dtype=np.int32)
            # sample['eod_index'] = np.array(batch_eod_index[0], dtype=np.int32)
            yield sample
        else:
            sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
            batch_position_ids = get_input_data_batch_slice_map(
                np.array([sample['input_ids']], dtype=np.int32),
                eod_id=EOT)
            sample['position_id'] = np.array(batch_position_ids[0], dtype=np.int32)
            # sample['eod_index'] = np.array(batch_eod_index[0], dtype=np.int32)
            # print(sample)
            yield sample


def tokenize_openwebtext_padEachPara(iterator):
    """ tokenize openwebtext dataset"""
    content = []
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for para in f.read().split("\n\n"):
                if para:
                    content = []
                    ###############################################
                    tokenized_text = tokenizer.tokenize(para)
                    content += tokenizer.convert_tokens_to_ids(tokenized_text) + [EOT]
                    ###########################################

                    for chunk in chunks(content, SEQ_LEN):
                        sample = {}
                        if len(chunk) == SEQ_LEN:
                            sample['input_ids'] = np.array(chunk, dtype=np.int32)
                            yield sample
                        else:
                            sample['input_ids'] = np.array(padding_eot(chunk), dtype=np.int32)
                            yield sample


def tokenize_wiki(file_path):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith('=') is False:
                tokenized_text = tokenizer.tokenize(para)
                content += tokenizer.convert_tokens_to_ids(tokenized_text) + [
                    EOT]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_lambada(file_path):
    """tokenize lambada dataset"""
    content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            para = json.loads(line)['text'].replace(
                "“", '""').replace("”", '"').strip().strip(".")
            tokenized_text = tokenizer.tokenize(para)
            content += tokenizer.convert_tokens_to_ids(tokenized_text) + [EOT]
    for chunk in chunks(content, SEQ_LEN):
        sample = {}
        if len(chunk) == SEQ_LEN:
            sample['input_ids'] = np.array(chunk, dtype=np.int32)
            yield sample


def task_unit(iterator, parallel_writer=False):
    """task for each process"""
    p = current_process()
    index = p.pid if p.pid else 0

    item_iter = tokenize_openwebtext(iterator)
    # item_iter = tokenize_openwebtext_padEachPara(iterator)
    batch_size = 1024  # size of write batch
    count = 0
    while True:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            random.shuffle(data_batch)
            # print(data_batch[0].keys())
            # print(data_batch[0]['attention_mask'])
            writer.write_raw_data(data_batch, parallel_writer=parallel_writer)
            print("Process {} transformed {} records.".format(
                index, count))
        except StopIteration:
            try:
                if data_batch:
                    random.shuffle(data_batch)
                    # print(data_batch[0].keys())
                    # print(data_batch[0]['attention_mask'][:100])
                    writer.write_raw_data(data_batch,
                                          parallel_writer=parallel_writer)
                    print("Process again {} transformed {} records.".format(
                        index, count))
            except Exception as error:
                print(error)
            break

def conver_words_to_ids_ch(words, word2id, numchword):
    ids = []
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
#         else:
#             tokenized_text = tokenizer.tokenize(word)
#             tmp = tokenizer.convert_tokens_to_ids(tokenized_text)
#             ids.extend(list(np.array(tmp) + numchword))
    return ids


#python src/pre_process_chinese.py --input_glob "/raid-50/commmon-crawl/common-crawl-WET-ReCleaned-filter-v3-length200-by08-v2-reClean2-sample30G/train/*"
# --output_file="/raid-50/commmon-crawl/common-crawl-WET-ReCleaned-filter-v3-length200-by08-v2-reClean2-sample30G-mindrecord/train/
# common-crawl-WET-ReCleaned-filter-v3-length200-by08-v2-reClean2-sample30G-mindrecord"
if __name__ == '__main__':

    ###
    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    schema = {"input_ids": {"type": "int32", "shape": [-1]},
              "position_id": {"type": "int32", "shape": [-1]},}
              # "eod_index": {"type": "int32", "shape": [-1]}}
    writer = FileWriter(file_name=args.output_file,
                        shard_num=args.file_partition)
    writer.add_schema(schema, args.dataset_type)
    writer.open_and_set_header()
    ###
    transforms_count = 0
    if args.dataset_type == 'wiki':
        for x in tokenize_wiki(args.input_glob):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'lambada':
        for x in tokenize_lambada(args.input_glob):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == 'openwebtext':
        # file_iter = glob.iglob(args.input_glob)
        input_files = list(glob.iglob(args.input_glob))
        # input_files = input_files*2
        input_files.sort()
        random.seed(10)
        random.shuffle(input_files)
        # input_files = input_files[:le]

        num_worker = int(args.rankOfCluster.split('of')[1])
        rank = int(args.rankOfCluster.split('of')[0])
        print(rank, '  of   ', num_worker)
        print('num files of cluster : ',len(input_files))
        input_files = divideIntoNstrand(input_files, num_worker)[rank]
        print('num files of this machine : ', len(input_files))

        # input_files = input_files[:int(len(input_files)*50/2000)]
        # print('num input_files : ',len(input_files))

        file_iter = (x for x in input_files)
        with Pool(processes=args.num_process) as pool:
            pool.map(task_unit, package_file(file_iter, args.file_batch_size))
        ################################
        # file_iter = glob.iglob(args.input_glob2)
        # with Pool(processes=args.num_process) as pool:
        #     pool.map(task_unit, package_file(file_iter, args.file_batch_size))
        ################################
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type))

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += '0'
    print("Transform finished, output files refer: {}".format(out_file))
