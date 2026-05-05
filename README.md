# Trainite Prototype

Trainite is a cookiecutter-style toolbox for language-model training with PyTorch-Ignite. This repository is the initial prototype: it provides a small but working training loop, a starter-project generator, and the scaffolding needed to grow into the fuller spec.

The goal is simple: get from zero to a runnable experiment fast, without hiding Ignite or forcing a large framework on top of it.

## What It Does

- Loads a typed YAML config with Pydantic
- Runs a small Ignite-based pretraining loop
- Logs metrics to TensorBoard
- Saves `last.pt` and `best.pt` checkpoints
- Generates a starter project with `trainite init`

## Quick Start

Install the project dependencies:

```bash
uv sync
```

Run the prototype in this repository:

```bash
uv run main.py
```

Generate a new starter project:

```bash
uv run trainite init ./my-experiment
```

Then enter the generated folder and run its training entrypoint:

```bash
cd my-experiment
python main.py config.yaml
```

## CLI

The current CLI exposes one command:

```bash
trainite init <project-dir>
```

This creates a small, editable training project in the target directory. The generated files are real source files the user owns and can modify freely.

Generated project files:

```text
config.yaml
config.py
model.py
dataset.py
trainer.py
main.py
```

## Repository Layout

```text
trainite_prototype/
├── main.py
├── config.yaml
├── trainite/
│   ├── cli/
│   ├── config/
│   ├── datasets/
│   ├── models/
│   └── trainers/
└── test/
```

## Current Scope

This prototype currently focuses on the pretraining path and the project generator.

- `PreTrainer` provides the current training loop
- The built-in dataset and model are synthetic placeholders for the starter workflow
- The CLI generates a local project template from the package source files

## Notes

- This is an early prototype, not the full spec implementation
- The package is structured so the CLI and template source stay separate from generated user code
- The spec still leaves room for richer trainer types, built-in models, and deeper config support

## Roadmap

- Expand the config schema to cover optimizer, scheduler, loss, metrics, and handlers
- Add `BaseTrainer` and `RLTrainer`
- Add more built-in datasets and model templates
- Add unit tests for the CLI and generated project structure

