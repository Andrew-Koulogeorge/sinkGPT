# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'attention_sinks'
wandb_run_name='gpt2-124M_base'

# these make the total batch size be ~0.5M
# b batch size * 1024 block size * 5 gradaccum * x GPUs = 
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 5 * 2

# this makes total number of tokens be y
max_iters = 10_000
lr_decay_iters = 10_000

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
attention_type="base"
