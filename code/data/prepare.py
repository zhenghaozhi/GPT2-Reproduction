import os
import time
import sys
import json
import tiktoken
import re
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.train_config import block_size
from config.train_config import EOT_TOKEN
from config.train_config import PAD_TOKEN

enc = tiktoken.get_encoding("gpt2")
# EOT_TOKEN = 50256
# PAD_TOKEN = 54321

block_cnt = int(0)

def clean_text(text):
    newline_idx = text.find("\n\n")
    if newline_idx != -1:
        text = text[newline_idx+2:]
    else:
        text = text
    meaningless = set({"\n", '（）', '()', '「」', '（；）', '（，）', '《》', '“”', r'-{}-', "<div>"}) # 剔除数据中无意义字段
    for pattern in meaningless:
        text = re.sub(re.escape(pattern), '', text)
    # text = re.sub(r'\（[^）]*\）|\【[^】]*\】|\([^)]*\)|\[[^]]*\]|<[^>]*>|《[^》]*》|「[^」]*」|\{[^}]*\}', '', text)
    # text = re.sub(r'[^\S\r\n]+', ' ', text)
    return text

def process_data(data_chunk, train_file, val_file):
    global block_cnt
    
    token_ids = enc.encode_ordinary(data_chunk)
    eot_encoded = enc.encode_ordinary("。")

    last_period_idx = block_size
    for i in range(block_size, -1, -1):
        if i == 0:
            return
        if token_ids[i:i + len(eot_encoded)] == eot_encoded and i + len(eot_encoded) < block_size:
            last_period_idx = i + len(eot_encoded)
            break

    token_ids = token_ids[:last_period_idx] # 找到最后一个不超过block_size的句号位置截断, 确保句子完整
    # print(str(enc.decode(token_ids)))

    chunks = [token_ids[i:i + block_size] for i in range(0, len(token_ids), block_size)] # 组成chunks, 此处由于已经进行截断&补全操作故只会有一个chunk
    if len(chunks) != 1:
        print("ERROR")
        print("token_ids:" + str(enc.decode(token_ids)))
        print("len(token_ids):" + str(len(token_ids)))
        print(len(chunks))
        print(data_chunk)
        time.sleep(10) # 确保每个wiki词条被处理截断或补全至一个block
    block_cnt += 1

    for chunk in chunks: # 使用<EOT>和<PAD>处理每个block
        if len(chunk) < block_size:
            chunk += [EOT_TOKEN]
            if len(chunk) < block_size:
                chunk += [PAD_TOKEN] * (block_size - len(chunk))
        
        if np.random.rand() < 0.9: # 通过随机的方式划分train与val
            train_file.write(np.array(chunk, dtype=np.uint16).tobytes())
        else:
            val_file.write(np.array(chunk, dtype=np.uint16).tobytes())

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki")
# names = [f for f in os.listdir(directory) if f.endswith('.jsonl')]

train_file = open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed"), "train.bin"), 'wb')
val_file = open(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed"), "val.bin"), 'wb') # 二进制写入

print("block_size = " + str(block_size))
data_chunk = ""
for root, dirs, files in os.walk(directory): # 通过遍历的方法处理指定目录下的所有jsonl
    print(f"Currently in directory: {root}")
    for file in files:
        file_path = os.path.join(root, file)
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_cnt, line in enumerate(f, start=1):
                # if line_cnt % 100 == 0:
                #     print(f"Line {line_cnt}")
                json_line = json.loads(line)
                text = clean_text(json_line["text"])
                process_data(text, train_file, val_file)
                data_chunk = ""

print("Block count: " + str(block_cnt)) # 输出总block数量, 方便后续计算epoch
train_file.close()
val_file.close()