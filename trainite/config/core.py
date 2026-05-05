from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    vocab_size: int = 32
    hidden_size: int = 64


class DatasetConfig(BaseModel):
    train_size: int = 256
    val_size: int = 64
    batch_size: int = 32
    seq_len: int = 16
    num_workers: int = 0
    seed: int = 7


class TrainerConfig(BaseModel):
    epochs: int = 3
    device: str = "cpu"
    learning_rate: float = 1e-3
    log_every_steps: int = 10
    grad_clip_norm: float | None = None


class OutputConfig(BaseModel):
    root: str = "output"
    run_name: str = "dummy-pretrain"


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 42


def default_config() -> Config:
    return Config()


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    return Config.model_validate(data)


def dump_config(config: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
