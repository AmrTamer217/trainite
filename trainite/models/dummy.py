from __future__ import annotations

import torch
from torch import nn

from trainite.config import ModelConfig


class DummyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.proj(hidden)


def build_dummy_model(config: ModelConfig) -> DummyLanguageModel:
    return DummyLanguageModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )
