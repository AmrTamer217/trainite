from __future__ import annotations

import torch
from torch import nn

from trainite.config.model import TransformerModelConfig


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.proj(hidden)


def build_transformer_model(config: TransformerModelConfig) -> TransformerModel:
    return TransformerModel(config)
