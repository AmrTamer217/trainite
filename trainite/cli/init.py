from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from trainite.config import default_config, dump_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def _replace_many(text: str, replacements: Iterable[tuple[str, str]]) -> str:
	for old, new in replacements:
		text = text.replace(old, new)
	if not text.endswith("\n"):
		text += "\n"
	return text


def _render_template(path: Path, replacements: Iterable[tuple[str, str]] = ()) -> str:
	return _replace_many(path.read_text(), replacements)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="trainite")
	subparsers = parser.add_subparsers(dest="command", required=True)

	init_parser = subparsers.add_parser("init", help="Generate a starter training project")
	init_parser.add_argument("project_dir", help="Directory to create the starter project in")
	init_parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite existing starter files",
	)
	init_parser.set_defaults(func=init_project)

	return parser


def _write_file(path: Path, content: str, force: bool) -> None:
	if path.exists() and not force:
		raise FileExistsError(f"{path} already exists; pass --force to overwrite it")
	path.write_text(content)


def _project_directory(raw_project_dir: str, force: bool) -> Path:
	project_dir = Path(raw_project_dir).expanduser().resolve()
	if project_dir.exists():
		if not project_dir.is_dir():
			raise SystemExit(f"{project_dir} is not a directory")
		if any(project_dir.iterdir()) and not force:
			raise SystemExit(
				f"{project_dir} is not empty; choose an empty directory or pass --force to overwrite starter files"
			)
	project_dir.mkdir(parents=True, exist_ok=True)
	return project_dir


def init_project(args: argparse.Namespace) -> None:
	project_dir = _project_directory(args.project_dir, args.force)

	starter_config = default_config()
	starter_config.output.run_name = project_dir.name

	templates = {
		"config.py": _render_template(PACKAGE_ROOT / "config" / "core.py"),
		"model.py": _render_template(
			PACKAGE_ROOT / "models" / "dummy.py",
			[
				("from trainite.config import ModelConfig", "from config import ModelConfig"),
				("def build_dummy_model(", "def build_model("),
			],
		),
		"dataset.py": _render_template(
			PACKAGE_ROOT / "datasets" / "dummy.py",
			[
				("from dataclasses import dataclass\n\n", ""),
				("from trainite.config import Config", "from config import Config"),
				("def build_dummy_dataloaders(", "def build_dataloaders("),
			],
		),
		"trainer.py": _render_template(
			PACKAGE_ROOT / "trainers" / "pretrainer.py",
			[
				("from trainite.config import Config, dump_config", "from config import Config, dump_config"),
				("from trainite.datasets import build_dummy_dataloaders", "from dataset import build_dataloaders"),
				("from trainite.models import build_dummy_model", "from model import build_model"),
				("class PreTrainer:", "class Trainer:"),
				("build_dummy_model(config.model)", "build_model(config.model)"),
				("build_dummy_dataloaders(config)", "build_dataloaders(config)"),
				("PreTrainer", "Trainer"),
			],
		),
		"main.py": _render_template(
			PROJECT_ROOT / "main.py",
			[
				("from trainite.config import default_config, load_config", "from config import default_config, load_config"),
				("from trainite.trainers import PreTrainer", "from trainer import Trainer"),
				("trainer = PreTrainer(config)", "trainer = Trainer(config)"),
			],
		),
	}

	dump_config(starter_config, project_dir / "config.yaml")
	for filename, content in templates.items():
		_write_file(project_dir / filename, content, args.force)

	print(f"Generated starter project in {project_dir}")
	for filename in ["config.yaml", *templates]:
		print(f"- {filename}")


def main(argv: Sequence[str] | None = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)
