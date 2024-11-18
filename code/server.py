from flask import Flask, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tiktoken
from model import GPTConfig, MiniGPT
from torch.nn import functional as F

app = Flask(__name__)

model_path = "out-super-4/ckpt.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
host_ip = "192.168.194.139" #port = 5000, 挂载至远程内网

def load_model(ckpt_path): # 加载本地模型
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
def load_tokenizer():
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)
    return encode, decode

tokenizer, detokenizer = load_tokenizer()
model = load_model(model_path)

def generate_stream(input_text, max_new_tokens=50, temperature=0.3, top_k=10): # 生成
    tmp = torch.tensor(tokenizer(input_text), dtype=torch.long, device=device)
    idx = tmp.unsqueeze(0).to(dtype=torch.long)
    global model
    config = model.config
    lst = []
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        lst.extend(idx_next[0].tolist())
        if 50256 in lst:
            return token_str + '\n'
        token_str = detokenizer(lst)
        yield token_str + '\n' 

@app.route('/predict', methods=['POST'])
def predict(): # POST请求
    data = request.json
    input_text = data['text']
    max_new_tokens = data['max_new_tokens']
    temperature = data['temperature']
    top_k = data['top_k']
    print(input_text)
    return Response(generate_stream(input_text, max_new_tokens, temperature, top_k), content_type='text/plain')

if __name__ == '__main__': # 启动Flask服务
    app.run(host=host_ip, port=5000)