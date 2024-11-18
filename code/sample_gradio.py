import gradio as gr
import torch
import time
from model import GPTConfig, MiniGPT
import tiktoken
import threading
import requests
import pandas as pd
import json
from queue import Queue, Empty
from torch.nn import functional as F
from contextlib import nullcontext

# local ckpt
local_model_paths = { # 挂载本地模型
    # "120M-max": "out-1721926184-largest/ckpt.pt",
    "100M-block": "out-1721835496-server-block/ckpt.pt",
    "120M-block": "out-1721971377-less-largest/ckpt.pt",
    "120M-largest": "out-1721926184-largest/ckpt.pt",
    "100M-block-good-sft": "out-1721922188-grad4-good/ckpt.pt",
    "100M-block-best-sft": "out-1721921239-grad4-best/ckpt.pt",
    "120M-largest-good-sft": "out-1721986584-large-grad4-1-good/ckpt.pt",
    "120M-largest-best-sft": "out-1721991956-large-grad4-3-best/ckpt.pt",
    "120M-largest-good-newsft": "out-1722012148-newlarge-grad4-1-good/ckpt.pt",
    "120M-largest-best-newsft": "out-1722017513-newlarge-grad4-3-bestnow/ckpt.pt",
    "block-super-3": "out-super-3/ckpt.pt",
    "largest-super-4": "out-super-4/ckpt.pt",
}
model_paths = { # 初始化模型列表
    # "120M-max": "out-1721926184-largest/ckpt.pt",
    "100M-block": "out-1721835496-server-block/ckpt.pt",
    "120M-block": "out-1721971377-less-largest/ckpt.pt",
    "120M-largest": "out-1721926184-largest/ckpt.pt",
    "100M-block-good-sft": "out-1721922188-grad4-good/ckpt.pt",
    "100M-block-best-sft": "out-1721921239-grad4-best/ckpt.pt",
    "120M-largest-good-sft": "out-1721986584-large-grad4-1-good/ckpt.pt",
    "120M-largest-best-sft": "out-1721991956-large-grad4-3-best/ckpt.pt",
    "120M-largest-good-newsft": "out-1722012148-newlarge-grad4-1-good/ckpt.pt",
    "120M-largest-best-newsft": "out-1722017513-newlarge-grad4-3-bestnow/ckpt.pt",
    "block-super-3": "out-super-3/ckpt.pt",
    "largest-super-4": "out-super-4/ckpt.pt",
    "# 郑皓之LAPTOP": "192.168.194.164:5000",
    "# 郑皓之 PC": "192.168.194.139:5000",
    "# 詹晓宇": "192.168.194.181:5000",
    "# 陈禹默": "192.168.194.32:5000",
}

seed = 114514
device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda or cpu
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

from model import MiniGPT
def generate_gradio(self, idx, max_new_tokens, temperature=1.0, top_k=None): # 重载model
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if not torch.equal(idx[:, -1:], idx_next):
                idx = torch.cat((idx, idx_next), dim=1)
                output_tokens = idx[0].tolist()
                try:
                    end_idx = output_tokens.index(50256) # 终止符扫描
                    output_tokens = output_tokens[:end_idx]
                except:
                    output = decode(output_tokens)
                    if("##" in output): # 对问题进行划分后, 保证答案不包括问题自身
                        output = output.split("##")[1]
                    yield output
                yield decode(output_tokens)
MiniGPT.generate_gradio = generate_gradio

def load_model(ckpt_path): # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def load_tokenizer(): # 加载编解码
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)
    return encode, decode

models_dict = {name: load_model(path) for name, path in local_model_paths.items()}
encode, decode = load_tokenizer()

def fetch_streamed_output(text, ip, max_new_tokens, temperature, top_k): # 调用api
    url = f"http://{ip}/predict"
    headers = {'Content-Type': 'application/json'}
    data = {
        "text": text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k
    }
    response = requests.post(url, headers=headers, json=data, stream=True)
    for chunk in response.iter_lines():
        if chunk:
            yield chunk.decode('utf-8')

def generate_text_stream(prompt, max_new_tokens, temperature, top_k, model_name, queue): # 生成
    if model_name in models_dict:
        model = models_dict[model_name]
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        start_time = time.time()
        for output in model.generate_gradio(x, max_new_tokens, temperature=temperature, top_k=top_k):
            elapsed_time = time.time() - start_time
            if '##' in output:
                output = output.split('##')[1]
            queue.put(f"Time: {elapsed_time:.2f}s\n" + output)
    else:
        ip = model_paths[model_name]
        start_time = time.time()
        streamed_text = fetch_streamed_output(prompt, ip, max_new_tokens, temperature, top_k)
        for chunk in streamed_text:
            elapsed_time = time.time() - start_time
            ans = prompt + chunk
            if '##' in ans:
                ans = ans.split('##')[1]
            queue.put(f"Time: {elapsed_time:.2f}s\n" + ans)

def single_model_interface(prompt, max_new_tokens, temperature, top_k, model): # 单人模式
    queue = Queue()
    thread = threading.Thread(target=generate_text_stream, args=(prompt, max_new_tokens, temperature, top_k, model, queue))
    thread.start()
    outputs = ""
    while thread.is_alive() or not queue.empty():
        try:
            output = queue.get_nowait()
            outputs = output
        except Empty:
            pass
        # yield [outputs.replace("�", "")]
        yield outputs.replace("�", "")
    thread.join()

def dual_model_interface(prompt, max_new_tokens, temperature, top_k, model1, model2): # 多人模式
    queue1 = Queue()
    queue2 = Queue()
    thread1 = threading.Thread(target=generate_text_stream, args=(prompt, max_new_tokens, temperature, top_k, model1, queue1))
    thread2 = threading.Thread(target=generate_text_stream, args=(prompt, max_new_tokens, temperature, top_k, model2, queue2))
    thread1.start()
    thread2.start()
    outputs1, outputs2 = "", ""
    while thread1.is_alive() or thread2.is_alive() or not queue1.empty() or not queue2.empty():
        try:
            output1 = queue1.get_nowait()
            outputs1 = output1
        except Empty:
            pass
        try:
            output2 = queue2.get_nowait()
            outputs2 = output2
        except Empty:
            pass
        yield outputs1.replace("�", ""), outputs2.replace("�", "")
        # yield [outputs1, outputs2]
        # yield [''.join(outputs1).replace("�", ""), ''.join(outputs2).replace("�", "")]
    thread1.join()
    thread2.join()

chat_history = [] # chatbot的浏览记录

def main_interface(prompt, mode, max_new_tokens, temperature, top_k, model1, model2=None):
    if mode == "Single Model":
        global chat_history
        for output in single_model_interface(prompt, max_new_tokens, temperature, top_k, model1):

            # yield output, ""
            found = False
            for i, (p, o) in enumerate(chat_history):
                if p == prompt:
                    chat_history[i][1] = output
                    found = True
                    break
            if not found:
                chat_history.append([prompt, output])

            yield chat_history, "", ""
    elif mode == "Dual Models":
        for output1, output2 in dual_model_interface(prompt, max_new_tokens, temperature, top_k, model1, model2):
            # yield output1, output2
            yield [], output1, output2

def update_dropdown(mode): # 根据模式切换当前页面布局
    if mode == "Single Model":
        # return gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif mode == "Dual Models":
        # return gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

eg = pd.read_csv("examples.csv")

with gr.Blocks() as demo: # 主框架

    outputs = [
        gr.Chatbot(label="Chatbot"), # 单人
        gr.Text(label="Output from Model 1", visible=False), # 多人
        gr.Text(label="Output from Model 2", visible=False)  # 多人
    ]
    mode_dropdown = gr.Dropdown(["Single Model", "Dual Models"], label="Select Mode", value="Single Model")
    inputs = [
        gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        mode_dropdown,
        gr.Slider(minimum=1, maximum=512, step=1, value=225, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.01, value=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Top K"),
        gr.Dropdown(list(model_paths.keys()), label="Select Model 1"),
        gr.Dropdown(list(model_paths.keys()), label="Select Model 2", visible=False)
    ]
    
    mode_dropdown.change(update_dropdown, inputs=mode_dropdown, outputs=[inputs[6], outputs[0], outputs[1], outputs[2]])

    # submit_button = gr.Button("submit", variant="primary")
    # submit_button.click(main_interface, inputs=inputs, outputs=outputs)

    interface = gr.Interface(
        fn=main_interface,
        inputs=inputs,
        outputs=outputs,
        title="MiniGPT Model Text Generation",
        description="Generate text using one or two MiniGPT models.",
        live=False,
        examples=eg.values.tolist()
    )

    demo.launch(server_name="192.168.194.139", server_port=7860) # 挂载至远程内网

if __name__ == '__main__':
    interface.launch(server_name="192.168.194.139", server_port=7860, share=False)
