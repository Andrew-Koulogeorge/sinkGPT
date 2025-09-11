"""
Author: Andrew Koulogeorge
Measure attention sink frequency within the heads of a pre-trained autoregressive llm 
"""
import torch
import numpy as np
import os
from model_base import GPTBase, GPTConfig

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/data/user_data/akouloge/attention_sinks/openwebtext"

def load_model(state_dict_path:str = None):
    """
    Load pretrained model based on state_dict. If no path is passed in, load fresh model
    """
    if not state_dict_path:
        print(f"downloading gpt2 model from hugging face...")
        model = GPTBase.from_pretrained(model_type="gpt2")
    return model 

def get_batch(global_pointer:int = 0,
              block_size:int = 50, 
              batch_size: int = 3, 
              split:str = "eval"):
    """
    go back later and make it so we iterate over this dataset instead of taking random samples
    """
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    else:
        data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
        total_tokens = block_size*batch_size 
        x = torch.from_numpy((data[global_pointer:global_pointer+total_tokens]).astype(np.int64)).reshape(shape=(batch_size,block_size))
        y = torch.from_numpy((data[global_pointer+1:global_pointer+1+total_tokens]).astype(np.int64)).reshape(shape=(batch_size,block_size))

    if DEVICE_TYPE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(DEVICE_TYPE, non_blocking=True), y.pin_memory().to(DEVICE_TYPE, non_blocking=True)
    else:
        x, y = x.to(DEVICE_TYPE), y.to(DEVICE_TYPE)
    return x, y

def evaluate_sinks(state_dict_path: str, 
                   eps:float = 0.3):
    """
    bare-bones implementation; compute attention sink existance on 1 batch of eval data
    """
    NUM_BATCHES = 2

    model = load_model(state_dict_path)    
    all_attns = []           # all_attn stores list of (B x Layers x H x N x N')       
    
    global_pointer = 0
    for _ in range(NUM_BATCHES):
        x,y = get_batch(global_pointer)
        global_pointer += x.numel()
        output = model(x,y,return_attn_scores=True)
        attn_output = output["all_attn"]
        stacked_on_layers = torch.stack(attn_output, dim=1)        # stack along layers
        all_attns.append(stacked_on_layers)
    all_attns = torch.cat(all_attns, dim=0)                  # (total_examples x Layers x H x N x N')
    print(f"shape of all attention heads: {all_attns.shape}")
    # compute what % of heads across all layers and over all batches have sinks in the first location 
    importance_scores = all_attns[...,0].mean(dim=-1)        # total_examples x Layers x H

    sink_score = torch.sum(importance_scores >= eps, dim=(0,1,2)) / importance_scores.numel()
    print(f"% of attention heads with an attention sink in the first token: {sink_score * 100}")


if __name__ == "__main__":
    state_dict_path = ""
    evaluate_sinks(state_dict_path)