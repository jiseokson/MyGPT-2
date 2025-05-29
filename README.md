# GPT-2 from Scratch with FineWeb and Distributed Training

## Overview

This project implements GPT-2 from scratch using PyTorch and trains it on the FineWeb dataset using distributed data-parallel (DDP) training across 8×A100 80GB GPUs. The goal is to understand the end-to-end process of building and training a large-scale language model, starting from data preprocessing and model definition to multi-GPU training orchestration.

Key features:
- Pure PyTorch implementation of a GPT-2 style transformer
- Efficient token-level streaming DataLoader for large-scale training
- Distributed training support via PyTorch's native DDP
- Compatible with multi-node/multi-GPU infrastructure

## Installation & Environment

This project was tested on the following environment:

- Python 3.11 
- PyTorch 2.8
- CUDA 12.8
- NCCL backend (for DDP)

Install dependencies:

```bash
pip install -r transformers datasets tiktoken wandb
```

## Training

### 1. Dataset Download & Preprocessing

We use the FineWeb dataset in NumPy shard format. To download and preprocess the dataset:

```bash
python fineweb.py
```

This script performs tokenization and serialization into compact .npy format

### 2. Launch Training
```bash
torchrun --nproc_per_node=8 train.py
```

## Model Architecture


## Results