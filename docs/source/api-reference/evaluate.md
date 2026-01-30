# Evaluation Module

Model evaluation and metrics.

> **Note**: The evaluation module has been refactored into a modular analysis system.
> The `ct_scan_mlops.evaluate` module now provides backward compatibility wrappers.
> For new code, use the `ct_scan_mlops.analysis` package directly.

## Overview

The evaluation system provides:

- **ModelDiagnostician** - Performance evaluation and error analysis
- **ModelLoader** - Load models from various checkpoint formats
- **InferenceEngine** - Unified inference pipeline
- **CLI interface** - Command-line evaluation via `invoke evaluate`

## Legacy Functions (Backward Compatibility)

::: ct_scan_mlops.evaluate.evaluate_model
    options:
      heading_level: 3

::: ct_scan_mlops.evaluate.load_model_from_checkpoint
    options:
      heading_level: 3

## Modern Analysis API

For new code, use the modular analysis system:

```python
from ct_scan_mlops.analysis.core import ModelLoader, InferenceEngine
from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician

# Load model
loader = ModelLoader(checkpoint_path="model.ckpt", device=device)
loaded_model = loader.load()

# Run inference
engine = InferenceEngine(
    model=loaded_model.model,
    device=device,
    uses_features=loaded_model.uses_features
)
results = engine.run_inference(test_loader)

# Evaluate
diagnostician = ModelDiagnostician(results=results, output_dir=Path("results"))
metrics = diagnostician.evaluate_performance()
```

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

### Programmatic Evaluation (Legacy)

```python
from pathlib import Path
from ct_scan_mlops.evaluate import evaluate_model, load_model_from_checkpoint
from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.utils import get_device

# DEPRECATED: Use ct_scan_mlops.analysis instead
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

### Modern Approach (Recommended)

```python
from pathlib import Path
from ct_scan_mlops.analysis.core import ModelLoader, InferenceEngine
from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician
from ct_scan_mlops.data import create_dataloaders

# Load model using new API
loader = ModelLoader(checkpoint_path=Path("model.ckpt"), device=device)
loaded = loader.load()

# Create test dataloader
_, _, test_loader = create_dataloaders(loaded.config)

# Run inference
engine = InferenceEngine(
    model=loaded.model,
    device=device,
    uses_features=loaded.uses_features
)
results = engine.run_inference(test_loader)

# Diagnose performance
diagnostician = ModelDiagnostician(results=results, output_dir=Path("results"))
metrics = diagnostician.evaluate_performance()

print(f"Accuracy: {metrics['accuracy']:.4f}")
```
