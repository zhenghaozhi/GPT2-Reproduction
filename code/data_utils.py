import os

import torch
import numpy as np
import tiktoken

from config.train_config import block_size
from config.train_config import EOT_TOKEN
from config.train_config import PAD_TOKEN
from config.train_config import A_TOKEN

train_data = None
val_data = None

count_sets = int(0)
if_trav = False # 是否取消随机, 遍历预训练数据

# PAD_TOKEN = 54321
# EOT_TOKEN = 50256

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data_path = os.path.join(data_dir, 'train.bin')
    val_data_path = os.path.join(data_dir,'val.bin')

    # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    train_data = np.fromfile(train_data_path, dtype=np.uint16).reshape(-1,block_size)
    val_data = np.fromfile(val_data_path, dtype=np.uint16).reshape(-1,block_size) # memmap改为fromfile方法并通过reshape保证读入后block的完整性

def init_data_sft(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data_path = os.path.join(data_dir, 'train_sft.bin')
    val_data_path = os.path.join(data_dir,'val_sft.bin')
    # train_data = np.memmap(os.path.join(data_dir, 'train_sft.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val_sft.bin'), dtype=np.uint16, mode='r')
    train_data = np.fromfile(train_data_path, dtype=np.uint16).reshape(-1,block_size)
    val_data = np.fromfile(val_data_path, dtype=np.uint16).reshape(-1,block_size) # 同上

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    global count_sets, if_trav
    if (if_trav): # 尝试遍历预训练数据的所有block, 对比效果
        if count_sets >= len(data):
            print("refresh")
            count_sets %= len(data)
        if (count_sets < len(data) and batch_size + count_sets >= len(data)):
            ix = torch.arange(count_sets, len(data))
            ix_add = torch.arange(0, batch_size + count_sets - len(data))
            ix = torch.cat((ix, ix_add))
        else:
            ix = torch.arange(count_sets, count_sets + batch_size)
        count_sets += batch_size
    else:
        ix = torch.randint(0, len(data), (batch_size,)) # 对准备好的block进行随机选取
    x = torch.stack([torch.from_numpy((data[i]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.append(data[i][1:], PAD_TOKEN).astype(np.int64)) for i in ix])
    
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    loss_mask[y == EOT_TOKEN] = 0
    eot_positions = (y == EOT_TOKEN).nonzero(as_tuple=True)
    for batch_idx, token_idx in zip(*eot_positions):
        if token_idx < loss_mask.size(1):
            loss_mask[batch_idx, token_idx:] = 0 # 将全部EOT_TOKEN及以后的loss_mask置0
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
    
def get_batch_sft(split, batch_size, block_size, device): 

    enc = tiktoken.get_encoding("gpt2")
    global train_data, val_data
    data = train_data if split == 'train' else val_data

    ix = torch.randint(0, len(data), (batch_size,))
    x = torch.stack([torch.from_numpy((data[i]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.append(data[i][1:], PAD_TOKEN).astype(np.int64)) for i in ix])
    
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    loss_mask[y == EOT_TOKEN] = 1 # 需要模型学会<EOT>终止
    eot_positions = (y == EOT_TOKEN).nonzero(as_tuple=True)
    for batch_idx, token_idx in zip(*eot_positions):
        if token_idx < loss_mask.size(1):
            loss_mask[batch_idx, token_idx:] = 0  # <EOT>之后置0

    loss_mask[y == A_TOKEN] = 0 # <A>之前的问题不需要作为答案的一部分
    a_token_positions = (y == A_TOKEN).nonzero(as_tuple=True)
    for batch_idx, token_idx in zip(*a_token_positions):
        if token_idx < loss_mask.size(1):
            loss_mask[batch_idx, :token_idx + 1] = 0 # <A>之前置0

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask