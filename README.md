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

## Experiment 1 - How Well Does Our Model Understand Context?

**LAMBADA**
- Language Modeling Broadened to Account for Discourse Aspects
- Extracted from BookCorpus; a collection of freely available English novels
- Consists of sentences that are difficult to complete without full context

**Evaluation Setup**
- Prompt : Full sentence excluding the final word
- Target : The final word
- Metric: accuracy — percentage of exact matches between prediction and target

|| GPT-2 Small | Our Model |
|-|-|-|
| Accuracy (%) | 45.99 | 16.03</br>(826/5153) |

> *Advanced performance excluding stop words; estimated increase of about 10%*

## Experiment 2 - How Well Does Our Model Predict the Most Contextual Word?

**CBT**
- Children's Book Test
- A single word (Common Noun, Named Entity, Verb, Preposition) is removed from a sentence
- 10 candidate words are provided, only one is correct

**Evaluation Setup**
- Prompt: Sentence with a missing word (CN, NE) (ex. "The cat chased the XXXX")
- Answer: One correct target among 10 candidates
- Metric: Common Noun(CN), Named Entity(NE) Accuracy
  1. For each candidate, compute the full-sequence loss after inserting it into the blank CN accuracy (%)
  2. The word with the lowest loss is selected as the model's prediction
 
|| GPT-2 Small | Our Model |
|-|-|-|
| CN Accuracy (%) | 87.65 | 72.51 (1807/2492) |
| NE Accuracy (%) | 83.40 | 51.14 (1275/2493) |
| Total Accuracy (%) | - | 61.83 (3082/4985) |

## Experiment 3 - Verifying Our GPT-2 Architecture Reproduction

| | LAMBADA accuracy (%) | CBT NE-accuracy (%) |
|-|-|-|
| | Paper | Our Model | Paper | Our Model |
| GPT2-small | - | 26.06 | 83.4 | 59.33 |
| GPT2-medium | - | 37.76 | 87.1 | 67.11 |
| GPT2-large | - | 40.58 | 88 | 68.95 |
| GPT2-XL | 52.66 | 44.69 | 89.5 | 72.32 |

## Additional Resources

[Building GPT-2.pdf](https://github.com/user-attachments/files/20561738/Building.GPT-2.pdf)
