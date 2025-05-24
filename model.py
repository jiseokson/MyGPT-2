from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from transformers import GPT2LMHeadModel

@dataclass
class GPTConfig:
  vocab_size: int = 50257
  n_ctx: int = 1024

  n_layer: int = 12
  n_embd: int = 768
  n_head: int = 12

  embd_pdrop: float = 0.1
  attn_pdrop: float = 0.1
  resid_pdrop: float = 0.1

  layer_norm_epsilon = 1e-5

  initializer_range: float = 0.02

class GPTMLP(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config

    self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x):
    return self.proj(F.gelu(self.fc(x)))
  
class GPTAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config

    self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.proj = nn.Linear(config.n_embd, config.n_embd)

    self.attn_dropout = nn.Dropout(config.attn_pdrop)

    self.register_buffer("tril", torch.tril(torch.ones(config.n_ctx, config.n_ctx)))

  def forward(self, x):
    B, T, C = x.shape

    qkv = self.attn(x)
    q, k, v = qkv.split(self.config.n_embd, dim=2)

    head_dim = self.config.n_embd // self.config.n_head
    q = q.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, self.config.n_head, head_dim).transpose(1, 2)

    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = self.attn_dropout(y)

    y = y.transpose(1, 2).reshape(B, T, C)

    return self.proj(y)
  
class GPTBlock(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config

    self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    self.attn = GPTAttention(config)

    self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    self.mlp = GPTMLP(config)

    self.resid_dropout = nn.Dropout(config.resid_pdrop)

  def forward(self, x):
    x = x + self.resid_dropout(self.attn(self.ln_1(x)))
    x = x + self.resid_dropout(self.mlp(self.ln_2(x)))

    return x

class GPTModel(nn.Module):
  configs = {
    "gpt2": GPTConfig(
      n_layer=12,
      n_embd=768,
      n_head=12,
    ),
    "gpt2-medium": GPTConfig(
      n_layer=24,
      n_embd=1024,
      n_head=16,
    ),
    "gpt2-large": GPTConfig(
      n_layer=36,
      n_embd=1280,
      n_head=20,
    ),
    "gpt2-xl": GPTConfig(
      n_layer=48,
      n_embd=1600,
      n_head=25,
    ),
  }

  def __init__(self, config):
    super().__init__()

    self.config = config

    self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
    self.pos_embedding = nn.Embedding(config.n_ctx, config.n_embd)

    self.token_dropout = nn.Dropout(config.embd_pdrop)

    self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])

    self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.token_embedding.weight = self.lm_head.weight

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

      if module.bias is not None:
        module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

    for name, param in module.named_parameters():
      if name == "proj.weight":
        param.data.normal_(mean=0.0, std=(self.config.initializer_range * (2 * self.config.n_layer) ** -0.5))

  def forward(self, inputs, targets=None):
    B, T = inputs.shape

    token_embd = self.token_embedding(inputs)
    pos_embd = self.pos_embedding(torch.arange(T).to(inputs.device))

    inputs = self.token_dropout(token_embd + pos_embd)

    for block in self.blocks:
      inputs = block(inputs)

    inputs = self.ln_f(inputs)

    logits = self.lm_head(inputs)

    loss = None

    if targets is not None:
      B, T, C = logits.shape

      logits = logits.view(B*T, -1)
      targets = targets.view(-1)

      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, ids, max_new_tokens):
    B, T = ids.shape

    for _ in range(max_new_tokens):
      ids_cond = ids[:, -self.config.n_ctx:]

      logits, _ = self(ids_cond)

      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)

      ids_next = torch.multinomial(probs, num_samples=1)

      ids = torch.concat([ids, ids_next], dim=1)

    return ids

  @classmethod
  def from_pretrained(cls, checkpoint):
    pre_model = GPT2LMHeadModel.from_pretrained(checkpoint)
    config = cls.configs[checkpoint]

    model = cls(config)
    states = {}

    for key, (pre_key, pre_param) in zip([key for key in model.state_dict().keys() if "tril" not in key], pre_model.state_dict().items()):
      if "blocks" in key and "weight" in key and pre_param.ndim == 2:
        states[key] = pre_param.t()
      else:
        states[key] = pre_param

    for i in range(config.n_layer):
      states[f"blocks.{i}.attn.tril"] = torch.tril(torch.ones(config.n_ctx, config.n_ctx))

    model.load_state_dict(states)

    return model
