
from transformers import LlamaTokenizer
vocab_file = './llama_vocab/llama_zh_hf/tokenizer_2.model'
tokenizer = LlamaTokenizer.from_pretrained(vocab_file)
EOT = tokenizer.eos_token_id
PAD = tokenizer.unk_token_id


'''
input_str:This sentence is negative: the format gets used best ... to capture the dizzying heights achieved by motocross and bmx riders , whose balletic hotdogging occasionally ends in bone-crushing screwups .
input_length:50, mask_length:5
start_sentence:[910, 10541, 338, 8178, 29901, 278, 3402, 4947, 1304, 1900, 2023, 304, 10446, 278, 270, 466, 1537, 292, 3171, 29879, 14363, 491, 3184, 542, 2124, 322, 289, 16838, 8177, 414, 1919, 5069, 289, 3498, 41925, 7375, 26169, 3460, 23025, 10614, 297, 289, 650, 29899, 7283, 21616, 885, 3973, 14340, 869]
loss:[5.368824481964111]
'''


a = "This sentence is negative: the format gets used best ... to capture the dizzying heights achieved by motocross and bmx riders , whose balletic hotdogging occasionally ends in bone-crushing screwups ."


input = tokenizer.encode(a, add_special_tokens=False)

print(len(input), input)

output = tokenizer.decode(input[5:], skip_special_tokens=True)

print(output)







