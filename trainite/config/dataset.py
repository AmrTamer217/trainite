from __future__ import annotations

from pydantic import BaseModel


class StringReverseDatasetConfig(BaseModel):
    vocab_size: int = 32
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    seq_len: int = 16
    num_workers: int = 0
    seed: int = 7


DATASET_CONFIGS: dict[str, type[BaseModel]] = {
    "string-reverse": StringReverseDatasetConfig,
}
