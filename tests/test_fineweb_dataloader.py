import logging
from fineweb import FineWebDataLoader

logger = logging.getLogger(__name__)

def test_fineweb_dataloader():
  iter=20000
  for rank in range(8):
    local_rank = rank
    world_size = 8

    total_batch_size = 2**19 # ~0.5M in number of tokens
    micro_batch_size = 64
    max_input_tokens = 1024
    grad_accm_steps = total_batch_size // (micro_batch_size * max_input_tokens * world_size)

    train_loader = FineWebDataLoader(micro_batch_size, max_input_tokens, local_rank, world_size, "train")
    val_loader = FineWebDataLoader(micro_batch_size, max_input_tokens, local_rank, world_size, "val")

    logger.info(f"Rank: {rank} | Train Loader")
    for _ in range(iter):
      train_loader.next_batch()

    logger.info(f"Rank: {rank} | Val Loader")
    for _ in range(iter):
      val_loader.next_batch()
