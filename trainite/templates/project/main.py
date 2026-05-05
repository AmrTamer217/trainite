from __future__ import annotations

import argparse
from pathlib import Path

from trainite.config import default_config, load_config
from trainite.trainers import PreTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else default_config()
    trainer = PreTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
