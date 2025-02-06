# mPengC.mind

PengCheng.Mind-m1(Mind-m1) is a multilingual large model developed by Pengcheng Laboratory, China, which is officially open-sourced for research purposes.  It is built upon a 7 billion model called PengCheng.Mind which focuses primarily on Chinese through multilingual incremental learning with the high-quality multilingual data. 

# Highlights

* Fully open-sourced code and model
* Supports the understanding and generation of 53 languages
* Supports both GPU and NPU to facilitate different types of users


## Model architecture

| multilingual PengCheng.Mind 7B | Structural Parameters |
| :---- | :---- |
| seq_length | 4096 |
| vocab_size | 125952 |
| embedding_size | 4096 |
| num_layers | 32 |
|num_heads | 32 |


## Requirements
### * NPU model:
python >= 3.7.5, mindspore >= 2.0.0-beta

It is recommended to use MindSpore's official Docker image.

| Hardware Platform | OS | Framework | #Inference Devices | #Fine-tuning Devices |
| :--- | :--- | :--- | :--- | :--- |
| Ascend 910 | EulerOS-aarch64 | MindSpore | 1 | ≥16 |
### * GPU model:
Please refer to：https://huggingface.co/PCLNLP/mPengC.mind_gpu/tree/main

## Model evolution and open-source

NPU Model：

Huggingface：https://huggingface.co/PCLNLP/mPengC.mind_npu/tree/main

ModelScope：https://modelscope.cn/models/PCLNLP/mPengC.Mind_npu/files

GPU Model：

Huggingface：https://huggingface.co/PCLNLP/mPengC.mind_gpu/tree/main

ModelScope：https://modelscope.cn/models/PCLNLP/mPengC.Mind_gpu/files


## Inference

### 1. Inference of the 7B Mind-m1

Startup command (`--offline 1` indicates using the NPU bare - metal machine. Modify `local_ckpt_path` in the script to the path of the 7B pre-trained model. `--offline 0` indicates using the ModelArts environment. Modify `restore_checkpoint_bucket_dir` in the script to the path of the 7B pre-trained model):

```
python predict_mPCmind7B.py \
--run_type predict \
--mode 7B \
--vocab_size 125952 \
--seq_length 4096 \
--distribute true \
--use_pynative_op 1 \
--op_level_model_parallel_num 1 \
--device_num 1 \
--stage_num 1 \
--top_p 1.0 \
--top_k_num 1 \
--max_generate_length 150 \
--pre_trained true \
--lang_idx 0 \
--use_rope True \
--use_past True \
--offline 0
```

### 2. Translation Evaluation

We conducted translation tests on 53 languages using the FLORES-200 dataset, and the SacreBLEU metric was used to evaluate the test results.

Test Data：```/data/mPC_flores-devtest.json```

Test Results：

![My Image](docs/翻译评测结果.png)



## Multilingual Incremental Learning

### 1. Training Data

Reference script：```/tools/pre_process_data.py```

Store multiple `xxx.json` files in the `YOUR_DATASET_PATH` directory. If you have a large amount of training data, it's best to keep the size of each json file consistent and split the data into multiple json files. Each file can be about 1MB in size.
If there are traditional Chinese characters, you need to convert them to simplified Chinese. You can use `zhconv` for this conversion.

The format of each JSON file is as follows: 
- different samples need to be separated by line-break symbols
- samples are classified into general corpora (open), monolingual corpora (mono), and parallel corpora (parallel) according to the "flag" field.
```
{"text":"sample", "flag":"open"}
{"text":"zh_sample", "lang":"zh", "flag":"mono"}
{"text":{"src": "zh_sample", "tag": "en_sample"}, "src_lang":"zh", "tgt_lang":"en", "flag":"parallel"}
```

The monolingual corpora and parallel corpora each contain corresponding language identification fields. For the specific text format of the JSON sample, please refer to: ```/data/pretrain_sample.txt```.

```
python pre_process_data.py --input_glob "YOUR_DATASET_PATH/*.json" --output_file "YOUR_OUTPUT_PATH/mindrecord" --SEQ_LEN 4097
```

mindrecord* files will be generated in the `YOUR_OUTPUT_PATH` directory.

### 2. Incremental Learning with Pre-trained Model

Startup command (modify `restore_checkpoint_bucket_dir` and `restore_ckpt_name_prefix` in the script to the path of the 7B pre-trained model):

```
python train_mPCmind7B.py \
--data_url YOUR_DATA_PATH \
--mode 7B \
--vocab_size 125952 \
--embedding_size 4096 \
--num_layers 32 \
--num_heads 32 \
--seq_length 4096 \
--device_num YOUR_DEVICE_NUM \
--stage_num 4 \
--op_level_model_parallel 2 \
--epoch_size 1 \
--micro_size 16 \
--optimizer_shard 1 \
--micro_batch_interleaved 1 \
--full_batch 0 \
--per_batch_size 1 \
--sink_size 2\
--warmup_step 1000 \
--param_init_type fp16 \
--start_lr 6e-05 \
--end_lr 1e-06 \
--save_checkpoint True \
--save_checkpoint_steps 2000 \
--save_strategy_bucket_dir YOUR_SFT_strategySaving_OBS_PATH \
--offline 0 \
--save_checkpoint_path YOUR_SFT_LOCAL_PATH \
--save_checkpoint_bucket_dir YOUR_SFT_OBS_PATH \
--save_summary_bucket_dir YOUR_SFT_summarySaving_OBS_PATH \
--recompute False \
--sequence_parallel True \
--use_rope True \
--select_recompute False \
--ckpt_name_prefix YOUR_SFT_PREFIX \
--pre_trained True
```

### 3. Merging Checkpoints

The model after incremental learning is sharded. If you want to perform inference, you need to merge the checkpoints first.

Reference script：
```
python tools/merge_ckpt.py --local_ckpt_save_name YOUR_LOCAL_SAVE_PATH --obs_ckpt_save_name YOUR_OBS_SAVE_PATH --restore_checkpoint_bucket_dir YOUR_MODEL_PATH --restore_ckpt_name_prefix YOUR_MODEL_PREFIX --rank YOUR_DEVICE_NUM --strategy YOUR_MERGED_STRATEGY_FILE
```
You can use `mindspore.merge_pipeline_strategys` to merge the strategy files.

## Convert the NPU model to a GPU model

Converted model：https://huggingface.co/PCLNLP/mPengC.mind_gpu

## Statement

[Open-source license of the PengCheng.Mind-m1](/docs/鹏城·脑海模型开源协议.pdf)

