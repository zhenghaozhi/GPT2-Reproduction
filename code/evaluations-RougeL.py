from model import GPTConfig, MiniGPT
import tiktoken
import numpy as np
from contextlib import nullcontext
import torch

Q_A_pair = [
    ("金火凌日的计算公式是？##", "金火凌日的计算公式为1/(1/P-1/Q)。"),
    ("羊侃的字是什么？##", "羊侃的字是祖忻。"),
    ("拜仁慕尼黑二队的主场在哪里？##", "拜仁慕尼黑二队的主场位于慕尼黑的绿森林体育场。"),
    # ("金星凌日是当金星运行到太阳和火星之间", "发生的一种罕见的天文现象。当金星凌日出现，从火星上可以看到金星像一个黑色圆盘从太阳表面划过。上次火星上的金星凌日发生在1998年8月21日。金火凌日的周期是333.92日，它是运用公式1/(1/P-1/Q)运算，其中P(金星)的轨道周期是224.701日，而Q(火星)的轨道周期是686.98日。"),
    # ("周士庄站位于山西省大同市大同县聚乐乡，", "邮政编码037305，建于1911年，是京包铁路大张段的一个车站。离北京站366公里，离包头站466公里。距上行车站阳高站41公里，距下行车站大同东站11公里。该站隶属太原铁路局大同铁路分局。办理客运业务（旅客乘降，行李、包裹托运）和货运业务（办理整车，不办理危险货物发到），为四等车站，本站及相邻上下行区间均为电气化区段。"),
    # ("一氧化锰是锰的一种氧化物，化学式", "MnO，在自然界中以罕见的方锰矿形式存在。一氧化锰可由氢气还原锰的高价氧化物得到，如：商业上用氢气、一氧化碳或甲烷还原二氧化锰制得：一氧化锰也可由碳酸锰的热分解制得：一氧化锰不溶于水，是一种碱性氧化物，溶于酸形成锰(II)盐。一氧化锰有着与氯化钠晶体相同的结构，而一氧化锰的组成可由MnO变化到MnO。118 K以下时，一氧化锰具有反铁磁性。一氧化锰在1951年被发现，其由中子衍射决定其磁性。"),
]
max_new_tokens, temperature, top_k = 256, 0.1, 20
model_path = "out-1722017513-newlarge-grad4-3-bestnow/ckpt.pt" # 微调
# model_path = "out-1721926184-largest" # 预训练

seed = 1234
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 也可以自行设置
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
def load_model(ckpt_path): # 模型载入
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
model = load_model(model_path)

def LCS_len(s1, s2): # LCS length
	if len(s1) > len(s2):
		return LCS_len(s2, s1)
	dp = np.zeros((len(s1) + 5, len(s2) + 5), dtype=np.int32)
	l1, l2 = len(s1), len(s2)
	for i in range(l1):
		for j in range(l2):
			dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
			if s1[i] == s2[j]:
				dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
	return dp[l1 - 1][l2 - 1]

def Rouge_L(X, Y, beta): # X: Standard A, YL Model A, beta length
	tmp_l = LCS_len(X, Y)
	R_LCS = tmp_l / len(X)
	P_LCS = tmp_l / len(Y)
	return ((1 + beta * beta) * R_LCS * P_LCS) / (R_LCS + beta * beta * P_LCS)

for (question, answer) in Q_A_pair: # QA对Rouge-L计算
    tok = enc.encode(question)
    input_tokens = torch.tensor(tok, device=device).unsqueeze(0).to(dtype=torch.long)
    y = model.generate(input_tokens, max_new_tokens, temperature, top_k)
    output_tokens = y[0].tolist()

    try:
        end_idx = output_tokens.index(50256)
        output_tokens = output_tokens[:end_idx]
    except:
        pass

    prediction = enc.decode(output_tokens[len(tok):])

    print("--------------------------------------")
    print("Question: " + question)
    print("Answer: " + answer)
    print("Prediction: " + prediction)
    print("Rouge-L: ", Rouge_L(answer, prediction, len(question)))