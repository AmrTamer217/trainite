from __future__ import annotations

from pydantic import BaseModel


class StringReverseDatasetConfig(BaseModel):
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    min_seq_len: int = 1
    max_seq_len: int = 16
    fixed_length: bool = True
    num_workers: int = 2
    seed: int = 7


DATASET_CONFIGS: dict[str, type[BaseModel]] = {
    "string-reverse": StringReverseDatasetConfig,
}
