import time

dataset = ''
out_dir = f'out-' + str(int(time.time()))
# eval_interval = 250 # keep frequent because we'll overfit
eval_interval = 1000
eval_iters = 200
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False



gradient_accumulation_steps = 2
batch_size = 32
block_size = 512 # context of up to 256 previous characters

iterations_per_epoch = block_cnt / batch_size # block_cnt为prepare阶段最终统计的总block数

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.01

learning_rate = 5e-4 # with baby networks can afford to go a bit higher

# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually

max_iters = iterations_per_epoch * 3
lr_decay_iters = max_iters

min_lr = 5e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 200 # not super necessary potentially

EOT_TOKEN = 50256
PAD_TOKEN = 220
A_TOKEN = 2235

# on macbook also add
# device = 'cpu'  # run on cpu only
