from __future__ import annotations

from pydantic import BaseModel


class TransformerModelConfig(BaseModel):
    vocab_size: int = 32
    hidden_size: int = 64


MODEL_CONFIGS: dict[str, type[BaseModel]] = {
    "transformer": TransformerModelConfig,
}
