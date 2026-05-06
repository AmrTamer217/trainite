# trainite — prototype

trainite is a small prototype CLI and library for generating minimal, hackable training projects.

Key ideas:

- Lightweight CLI: `trainite init` generates a starter project from package templates.
- Semi-interactive generator: defaults are pre-filled; missing values are prompted; use `--yes` to accept defaults non-interactively.
- Standalone config: generated `config.py` is fully inlined and self-contained (Pydantic classes + YAML helpers) so generated projects are easy to edit.
- Registry-driven components: models, datasets, and trainers are selected from a registry inside the package so swapping implementations is simple.

This repository is a prototype — the goal is to make a minimal, reproducible starting point you can hack on.

**Quick features**

- `trainite init <path>` — scaffold a new project. Accepts flags to select model/dataset/trainer and to override defaults.
- Generated files: `config.py`, `model.py`, `dataset.py`, `trainer.py`, `main.py` (all intended to be simple and editable).
- Registry: add new components by updating the package registry under `trainite/config/registry.py`.

**Recommended workflow**

1. Install the package in a dev environment:

```bash
pip install -e .
```

2. Create a new project quickly (non-interactive):

```bash
trainite init ./my-project --yes
```

Or run the module directly (no installation):

```bash
python -m trainite.cli init ./my-project --yes
```

3. Inspect and edit the generated `config.py` — it contains Pydantic v2 models and YAML helpers and is intentionally standalone and hackable.

4. Run the generated `main.py` from the generated project (it expects the generated `config.py` to be next to it):

```bash
python main.py
```

**CLI options (short)**

- `--model` choose a model (by name in the registry).
- `--dataset` choose a dataset (by name in the registry).
- `--trainer` choose a trainer (by name in the registry).
- `--run-name` set the default output run name.
- `--force` overwrite an existing output directory.
- `--yes` accept defaults and skip prompts.

See the CLI help for full details:

```bash
trainite init --help
```

**Generated project layout**

The generator emits a compact project containing:

- `config.py` — standalone ProjectConfig (inlined Pydantic classes) and YAML helpers.
- `model.py` — a small model builder and example model.
- `dataset.py` — an example dataset and dataloader builder.
- `trainer.py` — a tiny trainer that wires model, optimizer, and dataloaders.
- `main.py` — minimal entrypoint that loads `config.py`, builds components, and runs training.

These files are intentionally simple and editable — they are the recommended place to experiment.

**Extending the prototype**

- To add new components (models/datasets/trainers), add the implementation and its Pydantic config in `trainite/config`, then register the new entry in `trainite/config/registry.py`.
- The generator uses the registry to pick implementations and inlines the selected config classes into generated `config.py`.

**Notes & next steps**

- `config.py` is generated to be self-contained; the other generated files aim to be similarly straightforward but may still reference small package utilities — feel free to make them fully standalone in your generated template if you prefer.
- This prototype will evolve: add more model/dataset templates, richer trainer examples (RLTrainer), and tests for generated projects.

License: MIT-style prototype (see project metadata)

Enjoy — and tell me if you want the generator to inline more helpers or to add new templates.


