# Evaluation Module

Model evaluation and metrics.

## Overview

The evaluation module provides:

- **evaluate_model** - Core evaluation logic with metrics
- **load_model_from_checkpoint** - Load models from various checkpoint formats
- **CLI interface** - Command-line evaluation with Typer
- **Hydra integration** - Automatic checkpoint discovery

## Functions

::: ct_scan_mlops.evaluate.evaluate_model
    options:
      heading_level: 3

::: ct_scan_mlops.evaluate.load_model_from_checkpoint
    options:
      heading_level: 3

## CLI Commands

::: ct_scan_mlops.evaluate.evaluate_cli
    options:
      heading_level: 3

::: ct_scan_mlops.evaluate.evaluate_hydra
    options:
      heading_level: 3

## Metrics

The evaluator computes:

| Metric | Description |
|--------|-------------|
| `test_accuracy` | Overall classification accuracy |
| `test_{class}_precision` | Per-class precision |
| `test_{class}_recall` | Per-class recall |
| `test_{class}_f1` | Per-class F1 score |
| `test_macro_avg_f1` | Macro-averaged F1 |
| `test_weighted_avg_f1` | Weighted-averaged F1 |

## Checkpoint Formats

The loader supports multiple formats:

| Format | Key | Description |
|--------|-----|-------------|
| Lightning | `state_dict` | Full Lightning checkpoint with `model.` prefix |
| Full | `model_state_dict` | Custom checkpoint with optimizer state |
| Simple | (raw dict) | Plain PyTorch state dict |

## Usage Examples

### CLI Evaluation

```bash
# Basic
invoke evaluate --checkpoint outputs/.../best_model.ckpt

# With W&B logging
invoke evaluate --checkpoint path/to/model.ckpt --wandb --wandb-entity YOUR_USERNAME

# Custom batch size
invoke evaluate --checkpoint path/to/model.ckpt --batch-size 64
```

### Programmatic Evaluation

```python
from pathlib import Path
from ct_scan_mlops.evaluate import evaluate_model, load_model_from_checkpoint
from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.utils import get_device

# Load model
device = get_device()
model = load_model_from_checkpoint(Path("model.ckpt"), cfg, device)

# Create test dataloader
_, _, test_loader = create_dataloaders(cfg)

# Evaluate
metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
    save_confusion_matrix=True,
    output_dir=Path("results"),
)

print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
```

### Hydra-based Evaluation

```bash
uv run python -m ct_scan_mlops.evaluate
```

This automatically finds the most recent checkpoint.
