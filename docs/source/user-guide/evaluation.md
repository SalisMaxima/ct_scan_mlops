# Evaluation

Guide to evaluating trained CT scan classification models.

## Basic Evaluation

Evaluate a trained model on the test set:

```bash
invoke evaluate --checkpoint outputs/.../best_model.ckpt
```

## Checkpoint Formats

The evaluator supports multiple checkpoint formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| Lightning | `.ckpt` | Full training checkpoint with optimizer state |
| PyTorch | `.pt` | Model state dict only (lighter, recommended for deployment) |

## Command Options

```bash
# Basic evaluation
invoke evaluate --checkpoint path/to/model.ckpt

# With W&B logging
invoke evaluate --checkpoint path/to/model.ckpt --wandb --wandb-entity YOUR_USERNAME

# Custom batch size
invoke evaluate --checkpoint path/to/model.ckpt --batch-size 64

# Custom output directory
invoke evaluate --checkpoint path/to/model.ckpt --output results/
```

## Evaluation Outputs

### Metrics

The evaluator computes and displays:

- **Test Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall/F1**: Detailed per-class metrics
- **Macro/Weighted F1**: Aggregate F1 scores
- **Confusion Matrix**: Visual representation of predictions

### Classification Report

```
              precision    recall  f1-score   support

adenocarcinoma       0.95      0.93      0.94        45
large_cell_carcinoma 0.91      0.89      0.90        38
squamous_cell_carcinoma 0.88  0.92      0.90        42
      normal         0.96      0.97      0.96        50

    accuracy                           0.93       175
   macro avg         0.92      0.93      0.93       175
weighted avg         0.93      0.93      0.93       175
```

### Confusion Matrix

A confusion matrix is saved to the output directory as `confusion_matrix.png`.

## W&B Integration

Log evaluation results to W&B:

```bash
invoke evaluate \
    --checkpoint path/to/model.ckpt \
    --wandb \
    --wandb-project CT_Scan_MLOps \
    --wandb-entity YOUR_USERNAME
```

Logged metrics include:

- Test accuracy
- Per-class precision, recall, F1
- Confusion matrix image

## Programmatic Evaluation

### Using the Modern Analysis API (Recommended)

```python
from pathlib import Path
from ct_scan_mlops.analysis.core import ModelLoader, InferenceEngine
from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician
from ct_scan_mlops.data import create_dataloaders

# Load model using new modular API
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

# Evaluate performance
diagnostician = ModelDiagnostician(results=results, output_dir=Path("results"))
metrics = diagnostician.evaluate_performance()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Using Legacy API (Backward Compatibility)

```python
from ct_scan_mlops.evaluate import evaluate_model, load_model_from_checkpoint
from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.utils import get_device

# DEPRECATED: Use ct_scan_mlops.analysis instead
model = load_model_from_checkpoint(checkpoint_path, cfg, device)

# Create test dataloader
_, _, test_loader = create_dataloaders(cfg)

# Evaluate
metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=get_device(),
    save_confusion_matrix=True,
    output_dir=Path("results"),
)

print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
```

## Hydra-based Evaluation

For integration with the training pipeline:

```bash
uv run python -m ct_scan_mlops.evaluate
```

This automatically finds the most recent checkpoint and evaluates it.
