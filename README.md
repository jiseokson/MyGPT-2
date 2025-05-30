# GPT-2 from Scratch with FineWeb and Distributed Training

## Overview

This project implements GPT-2 from scratch using PyTorch and trains it on the FineWeb dataset using distributed data-parallel (DDP) training across 8×A100 80GB GPUs. The goal is to understand the end-to-end process of building and training a large-scale language model, starting from data preprocessing and model definition to multi-GPU training orchestration.

Key features:
- Pure PyTorch implementation of a GPT-2 style transformer
- Efficient token-level streaming DataLoader for large-scale training
- Distributed training support via PyTorch's native DDP
- Compatible with multi-GPU infrastructure

## Installation & Environment

This project was tested on the following environment:

- Python 3.11 
- PyTorch 2.8
- CUDA 12.8
- NCCL backend (for DDP)

Install dependencies:

```bash
pip install transformers datasets tiktoken wandb
```

## Training

### 1. Dataset Download & Preprocessing

We use the FineWeb dataset in NumPy shard format.

To download and preprocess the dataset:

```bash
python fineweb.py
```

This script performs tokenization and serialization into compact `.npy` format
> ⚠️ Make sure you have at least 20GB of free disk space available for storing the preprocessed dataset, as the script will generate tokenized `.npy` files under the `data/` directory.

### 2. Launch Training

We trained the model on RunPod using a single node with 8×A100 80GB GPUs. Distributed training was conducted via PyTorch's `torchrun` launcher with the NCCL backend.

To launch training:

```bash
torchrun --nproc_per_node=8 train.py
```

## Results

Loss decreased rapidly during the first 3,000 steps then continued to decline steadily.
The gap between training and validation loss remained minimal indicating no signs of overfitting.

<p align="center">
 <img src="https://github.com/user-attachments/assets/a64fc811-73e0-4f73-833f-944902b5c361" width="45%"/>
 <img src="https://github.com/user-attachments/assets/ac13fb4a-c265-4d24-9f4d-04039be50e9d" width="45%"/>
</p>
