### prepare SFT data similar to `prepare.py`

import os
import sys
import json
import tiktoken
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.train_config import block_size
from config.train_config import EOT_TOKEN
from config.train_config import PAD_TOKEN
from config.train_config import A_TOKEN

enc = tiktoken.get_encoding("gpt2")

print("block_size = " + str(block_size))
print("A_TOKEN = " + str(A_TOKEN))
block_cnt = int(0)

data = ""
train_sft = open(os.path.join( os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_sft") , "train_sft.bin"), 'wb')
val_sft = open(os.path.join( os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_sft") , "val_sft.bin"), 'wb')
line_count = 0

def process_line(line):
    json_line = json.loads(line)
    q = json_line["question"]
    a = json_line["answer"]
    qa = enc.encode(f"{q}") + [A_TOKEN] + enc.encode(f"{a}") + [EOT_TOKEN] # 提取QA对并与<A>和<EOT>拼接
    if len(qa) > block_size: # 长度检测
        return None
    elif (len(qa) == block_size):
        return qa
    else:
        qa += [PAD_TOKEN] * (block_size - len(qa))
        return qa

print(f"Processing file: sft_saves.jsonl")
with open(os.path.join( os.path.dirname(os.path.abspath(__file__)) , "sft_saves.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        qa_ids = process_line(line)
        if qa_ids is None:
            continue
        block_cnt += 1
        if np.random.rand() < 0.9: # 通过随机的方式划分train与val
            np.array(qa_ids, dtype=np.uint16).tofile(train_sft)
        else:
            np.array(qa_ids, dtype=np.uint16).tofile(val_sft)

print("Block cnt: " + str(block_cnt))
train_sft.close()
val_sft.close()