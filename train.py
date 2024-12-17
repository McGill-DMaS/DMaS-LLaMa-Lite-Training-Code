# Implementation in this file is modified from https://github.com/karpathy/build-nanogpt

import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
# -----------------------------------------------------------------------------
from data_loader import DataLoaderLite
from presets import *
# -----------------------------------------------------------------------------
import tiktoken
import numpy as np
import wandb
from utils import *

# -----------------------------------------------------------------------------
# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 train.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse


# Create the parser
parser = argparse.ArgumentParser(description='Train a model')

# Add the arguments
parser.add_argument('--for_develop', type=bool, default=False, help='For development')
parser.add_argument('--model_name', type=str, default='llama', help='Model name')
parser.add_argument('--project_name', type=str, default='dmas-llama-lite', help='Project name')
parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
parser.add_argument('--data_root', type=str, default="edu_fineweb10B", help='Root of Data folder')
parser.add_argument('--model_imp', type=str, default='custom', help='Model name')
parser.add_argument('--siz', type=str, default='124m', help='Size')
parser.add_argument('--log_dir', type=str, default='log_gpt2_variant', help='Log directory')

parser.add_argument('--B', type=int, default=32, help='Micro batch size')
parser.add_argument('--T', type=int, default=1024, help='Sequence length')
parser.add_argument('--start_shard', type=int, default=0, help='Start shard')

parser.add_argument('--total_batch_size', type=int, default=524288, help='Total batch size')

parser.add_argument('--continue_training', type=bool, default=False, help='Continue training')
parser.add_argument('--ckpt_path', type=str, default='log/model_05000.pt', help='Checkpoint path')

parser.add_argument('--max_lr', type=float, default=6e-4, help='Max learning rate')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Min learning rate')
parser.add_argument('--warmup_steps', type=int, default=715, help='Warmup steps')

parser.add_argument('--total_token', type=int, default=10**10, help='Total token')

parser.add_argument('--max_steps', type=int, default=19073, help='Max steps')

parser.add_argument('--evaluate_every', type=int, default=250, help='Evaluate every')
parser.add_argument('--evaluate_hella_every', type=int, default=250, help='Evaluate Hella every')
parser.add_argument('--save_every', type=int, default=2000, help='Save every')

parser.add_argument('--generate_every', type=int, default=250, help='Generate every')
parser.add_argument('--generate_prompt', type=str, default='Hello, I am a language model,', help='Generate prompt')
parser.add_argument('--num_return_sequences', type=int, default=4, help='Number of return sequences')
parser.add_argument('--generate_max_length', type=int, default=32, help='Generate max length')

# Parse the arguments
args = parser.parse_args()


wandb.init(project=args.project_name, name=args.exp_name, resume='allow')

# Now you can use the arguments as variables in your code
for_develop = args.for_develop
model_name = args.model_name
siz = args.siz
log_dir = args.log_dir
model_imp = args.model_imp

data_root = args.data_root

B = args.B
T = args.T

total_batch_size = args.total_batch_size

continue_training = args.continue_training
ckpt_path = args.ckpt_path

max_lr = args.max_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps

total_token = args.total_token

max_steps = args.max_steps

evaluate_every = args.evaluate_every
evaluate_hella_every = args.evaluate_hella_every
save_every = args.save_every

generate_every = args.generate_every
generate_prompt = args.generate_prompt
num_return_sequences = args.num_return_sequences
generate_max_length = args.generate_max_length
start_shard = args.start_shard
start_position = 0


if for_develop:
    model_name = "gpt2"
    siz = '10m'
    log_dir = "devel"
    max_steps = 100
    evaluate_every = 10
    evaluate_hella_every = 10
    save_every = 10
    generate_every = 10
    B = 2 # micro batch size
    total_batch_size = 2048

#============================================================================


number_of_steps_per_epoch = total_token // total_batch_size
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
enc = tiktoken.get_encoding("gpt2")


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"


grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train",
        data_root = data_root, master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, data_root = data_root, split="val", master_process=master_process)

torch.set_float32_matmul_precision('high')

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

start_step = 0
if model_imp == 'hf':
    print("training with HuggingFace transformers implementation")
    from transformers import GPT2LMHeadModel, GPT2Config
    hf_conf = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(hf_conf)
else:
    print(f"training with custom implementation of {model_name}")
    if model_name == "gpt2":
        from models.gpt2 import GPT, GPTConfig
        # create model
        if continue_training:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'current_shard' in checkpoint:
                start_shard = checkpoint['current_shard']
                start_position = checkpoint['current_position']
            config = checkpoint['config']
            model = GPT(config)
            model, ini_step = load_model(model, ckpt_path, True)
            start_step = ini_step
        else:
            model = GPT(model_presets[model_name][siz])
    elif model_name == "gpt2variant":
        from models.gpt2_variant import GPTVariant, GPTVariantConfig
        model = GPTVariant(model_presets[model_name][siz])
    elif model_name == 'llama':
        from models.llama import LlamaTransformer, LlamaConfig
        if continue_training:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if 'current_shard' in checkpoint:
                start_shard = checkpoint['current_shard']
                start_position = checkpoint['current_position']

            config = checkpoint['config']
            model = LlamaTransformer(config)
            model, ini_step = load_model(model, ckpt_path, True)
            start_step = ini_step
        else:
            model = LlamaTransformer(model_presets[model_name][siz])
train_loader.set_shard_and_pos(start_shard, start_position)
print(f"trainin from shard {start_shard} position {start_position}")
#if continue_training:
#    model, ini_step = load_model(model, ckpt_path, True)
#    start_step = ini_step

num_params = sum([p.numel() for p in model.parameters()])
if master_process:
    print(f"number of parameters: {num_params:,}")

model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
if model_imp == 'hf':
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
else:
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type,master_process=master_process)

# create the log directory we will write checkpoints to and log to
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
#with open(log_file, "w") as f: # open for writing to clear the file
#    pass

max_steps = max_steps+start_step
for step in range(start_step,max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if (step % evaluate_every == 0 and step > 0) or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y, current_valid_shard, current_valid_position = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if model_imp == 'hf':
                        loss, logits, _ = model(x, labels=y, return_dict=False)
                    else:
                        logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        master_process = True
        if master_process:
            wandb.log({"validation_loss": val_loss_accum.item()}, step=step)
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    if step > start_step and (step % save_every == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'current_shard': current_train_shard,
            'current_position': current_train_position
            #'val_loss': val_loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % evaluate_hella_every == 0 or last_step) and (not use_compile) and step > start_step:
        num_correct_norm,num_total,acc_norm = evaluate_hella_swag(model, device, device_type,model_imp, dist,ddp,master_process,ddp_rank,ddp_world_size,log_file,step)
        master_process = True
        if master_process:
            wandb.log({"hella_accuracy": acc_norm}, step=step)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > start_step and step % generate_every == 0) or last_step) and (not use_compile):
        completion(model, enc, generate_prompt, device, device_type,model_imp, generate_max_length, num_return_sequences, rank=ddp_rank)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y, current_train_shard, current_train_position = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            # only sync gradients after completing all the micro-steps to reduce unnecessary communication
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
             if model_imp == 'hf':
                 loss, logits, _ = model(x, labels=y, return_dict=False)
             else:
                 logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    progress = step / max_steps
    estimated_rest_time = (max_steps - step) * dt / 3600
    master_process = True
    if master_process:
        wandb.log({"training_loss": loss_accum.item(), "learning_rate": lr, "norm": norm, "tokens_per_sec": tokens_per_sec}, step=step)
        print(f"step {step:5d} | progress {progress*100:.2f}% | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | est: {estimated_rest_time:.2f}h")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
