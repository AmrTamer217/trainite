# String-Reverse Example

A simple sequence-to-sequence task where a Transformer learns to reverse strings composed of characters. This example provides a complete, standalone template for training models on synthetic symbolic tasks.

## Getting Started

This example was generated using the Trainite CLI:

```bash
uv run trainite init examples/string-reverse --model transformer --dataset string-reverse -y
```

All files in this directory are self-contained and do not depend on the `trainite` library. You can modify them freely for your own experiments.

## Files

| File | Description |
|---|---|
| `config.yaml` | Hyperparameters including model dimensions, sequence lengths, and alphabet |
| `config.py` | Pydantic classes for type-safe configuration loading |
| `model.py` | Transformer architecture with positional encoding and multi-head attention |
| `dataset.py` | Dataset logic for generating character sequences and their reversals |
| `trainer.py` | Full training loop with automated metric logging and checkpointing |
| `main.py` | Main entry point to load config and start training |

## Running Training

From the project root (`trainite/`):

```bash
cd examples/string-reverse
uv run python main.py config.yaml
```

### Expected Output

```text
2026-05-12 16:21:43 [INFO] trainer: starting run in output/transformer__string-reverse/...
2026-05-12 16:21:44 [INFO] trainer: Evaluating on training set...
2026-05-12 16:21:44 [INFO] trainer: Evaluating on validation set...
2026-05-12 16:21:44 [INFO] trainer: epoch=1 train_loss=3.3850 train_acc=0.0461 val_loss=3.3971 val_acc=0.0500
...
```

## Task Description

The model is trained to reverse a sequence of characters from a defined alphabet.

- **Input**: A character sequence, e.g. `['a', 'b', 'c', 'd']`
- **Output**: The reversed sequence, e.g. `['d', 'c', 'b', 'a']`
- **Vocabulary**: Characters defined in the `alphabet` config string.
- **Variable Length**: The dataset supports sequences of varying lengths, using a padding token (Index 0) to align batches.

## Configuration

You can customize the task by editing `config.yaml`. For example, to change the sequence complexity:

```yaml
dataset:
  alphabet: "abcdefghijklmnopqrstuvwxyz"
  min_seq_len: 4
  max_seq_len: 32
  fixed_length: false
```
