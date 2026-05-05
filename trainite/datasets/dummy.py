from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from trainite.config import Config


class SyntheticPretrainingDataset(Dataset):
    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randint(
            low=0,
            high=vocab_size,
            size=(size, seq_len),
            generator=generator,
        )
        self.labels = self.inputs.clone()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }


def build_dummy_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_dataset = SyntheticPretrainingDataset(
        size=config.dataset.train_size,
        seq_len=config.dataset.seq_len,
        vocab_size=config.model.vocab_size,
        seed=config.dataset.seed,
    )
    val_dataset = SyntheticPretrainingDataset(
        size=config.dataset.val_size,
        seq_len=config.dataset.seq_len,
        vocab_size=config.model.vocab_size,
        seed=config.dataset.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
    )
    return train_loader, val_loader
