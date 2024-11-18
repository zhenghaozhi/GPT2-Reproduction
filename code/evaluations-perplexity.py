from data_utils import init_data_pretrain, init_data_sft, get_batch_pretrain, get_batch_sft
from model import GPTConfig, MiniGPT
import tiktoken
from contextlib import nullcontext
import torch
from config.train_config import block_size
import math

model_path = "out-1722017513-newlarge-grad4-3-bestnow/ckpt.pt"
# model_path = "out-1721835496-server-block/ckpt.pt"
dataset_name = "processed_sft"
is_sft = True
batch_size = 8
iterations = 100

seed = 1234
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**checkpoint['model_args'])
    model = MiniGPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model



model = load_model(model_path)
if is_sft:
    init_data = init_data_sft
    get_batch = get_batch_sft
else:
    init_data = init_data_pretrain
    get_batch = get_batch_pretrain

init_data(dataset_name)
total_loss, total_count = 0, 0

for _ in range(iterations):
    batch_of_data = get_batch('val', batch_size, block_size, device)
    idx, targets, loss_mask = batch_of_data
    _, loss = model.forward(idx, targets, loss_mask)
    total_loss += loss.item() * (loss_mask != 0).sum().item()
    total_count += (loss_mask != 0).sum().item()

avg_loss = total_loss / total_count
print("Perplexity: " + str(math.exp(avg_loss)))