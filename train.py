import os
import math
import time
import inspect

import torch
from torch.optim.lr_scheduler import LambdaLR

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPTModel, GPTConfig
from fineweb import FineWebDataLoader

ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
  init_process_group(backend="nccl")

  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  device = f"cuda:{local_rank}"
  torch.cuda.set_device(device)

  master_process = (local_rank == 0)

else:
  rank = 0
  local_rank = 0
  world_size = 1

  device = "cuda" if torch.cuda.is_available else "cpu"

  master_process = True

device_type = "cuda" if "cuda" in device else "cpu"

total_batch_size = 2**19 # ~0.5M in number of tokens
micro_batch_size = 64
max_input_tokens = 1024
grad_accm_steps = total_batch_size // (micro_batch_size * max_input_tokens * world_size)

weight_decay = 0.1
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 15 # for test
warmup_steps = 5 # for test

def get_optimizer(model, weight_decay, learning_rate, device_type):
  param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

  decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
  nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

  optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
  ]

  fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and device_type == "cuda"

  optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

  return optimizer

def get_scheduler(optimizer, max_lr=6e-4, min_lr=6e-4*0.1, max_steps=19073, warmup_steps=715):
  def get_lr_factor(step):
    if step < warmup_steps:
      return (step + 1) / warmup_steps
    if step >= max_steps:
      return min_lr / max_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))

    return min_lr / max_lr + coeff * (1 - min_lr / max_lr)
  
  return LambdaLR(optimizer, get_lr_factor)

torch.set_float32_matmul_precision("high")

raw_model = GPTModel(GPTConfig(vocab_size=50304)).to(device)
torch.compile(raw_model)

model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model

optimizer = get_optimizer(raw_model, weight_decay, learning_rate=max_lr, device_type=device_type)
optimizer_scheduler = get_scheduler(optimizer, max_lr, min_lr, max_steps, warmup_steps)
optimizer_scheduler.step()

train_loader = FineWebDataLoader(micro_batch_size, max_input_tokens, local_rank, world_size, "train")
val_loader = FineWebDataLoader(micro_batch_size, max_input_tokens, local_rank, world_size, "val")

for step in range(max_steps):
  if master_process:
    start = time.time()

  last = (step == max_steps - 1)

  model.train()
  optimizer.zero_grad()

  loss_accm = 0.0

  for micro_step in range(grad_accm_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accm_steps - 1)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      _, loss = model(x, y)

    loss /= grad_accm_steps
    loss_accm += loss.detach()

    loss.backward()

  if ddp:
    dist.all_reduce(loss_accm, op=dist.ReduceOp.AVG)

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = optimizer_scheduler.get_last_lr()[0]

  optimizer.step()
  optimizer_scheduler.step()

  if device_type == "cuda":
    torch.cuda.synchronize() # for logging

  if master_process:
    end = time.time()
    duration = end - start

    proccess_tokens = total_batch_size
    tokens_per_sec = proccess_tokens / duration

    print(f"step: {step:3} | "
          f"loss: {loss_accm.item():8.3f} | "
          f"lr: {lr:.3e} | "
          f"norm: {norm:.3f} | "
          f"duration: {duration*1000:.3f} ms | "
          f"tps: {tokens_per_sec:.3f} tok/s")

if ddp:
  destroy_process_group()
