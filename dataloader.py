import os

import torch

class FineWebDataLoader:
  def __init__(self, batch_size, max_input_tokens, proc_rank, num_procs, split="train", root_dir="data"):
    self.batch_size = batch_size
    self.max_input_tokens = max_input_tokens

    self.proc_rank = proc_rank
    self.num_procs = num_procs

    self.split = split

    self.root_dir = root_dir

    os.makedirs(root_dir, exist_ok=True)

  def next_batch(self):
    x = torch.randint(0, 100, (self.batch_size, self.max_input_tokens + 1)).long()
    return x[:, :-1], x[:, 1:]
