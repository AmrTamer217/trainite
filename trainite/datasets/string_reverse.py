from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from trainite.config.dataset import StringReverseDatasetConfig


class StringReverseDataset(Dataset):
    def __init__(
        self,
        size: int,
        min_seq_len: int,
        max_seq_len: int,
        seed: int,
        fixed_length: bool = True,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
    ) -> None:
        self.alphabet = alphabet
        vocab_size = len(alphabet)
        self.char_to_id = {c: i + 1 for i, c in enumerate(alphabet)}
        self.id_to_char = {i + 1: c for i, c in enumerate(alphabet)}
        
        generator = torch.Generator().manual_seed(seed)
        
        if fixed_length:
            self.inputs = torch.randint(
                low=1,
                high=vocab_size + 1,
                size=(size, max_seq_len),
                generator=generator,
            )
            self.labels = torch.flip(self.inputs, dims=[1])
        else:
            self.inputs = []
            for _ in range(size):
                length = torch.randint(
                    low=min_seq_len,
                    high=max_seq_len + 1,
                    size=(1,),
                    generator=generator,
                ).item()

                seq = torch.randint(
                    low=1,
                    high=vocab_size + 1,
                    size=(length,),
                    generator=generator,
                )
                self.inputs.append(seq)
            
            self.labels = [torch.flip(seq, dims=[0]) for seq in self.inputs]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }

    def decode(self, ids: torch.Tensor) -> str:
        return "".join([self.id_to_char[idx.item()] for idx in ids if idx.item() != 0])


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
    }


def build_string_reverse_dataloaders(
    config: StringReverseDatasetConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = StringReverseDataset(
        size=config.train_size,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        fixed_length=config.fixed_length,
        alphabet=config.alphabet,
    )
    val_dataset = StringReverseDataset(
        size=config.val_size,
        min_seq_len=config.min_seq_len,
        max_seq_len=config.max_seq_len,
        seed=config.seed + 1,
        fixed_length=config.fixed_length,
        alphabet=config.alphabet,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
