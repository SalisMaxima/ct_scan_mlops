# Training Module

Training pipeline with PyTorch Lightning.

## Overview

The training module provides:

- **LitModel** - Lightning module wrapping the classifier
- **train_model** - Main training function with full MLOps features
- **Logging** - Configurable file and console logging
- **Reproducibility** - Seed management for reproducible training

## Lightning Module

::: ct_scan_mlops.train.LitModel
    options:
      heading_level: 3
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - test_step
        - configure_optimizers
        - on_train_start
        - on_train_epoch_end

## Training Functions

::: ct_scan_mlops.train.train_model

::: ct_scan_mlops.train.train

## Utility Functions

::: ct_scan_mlops.train.configure_logging

::: ct_scan_mlops.train.set_seed

## Training Features

### Callbacks

The training pipeline includes:

- **ModelCheckpoint** - Saves best model based on validation metric
- **EarlyStopping** - Stops training when metric plateaus
- **LearningRateMonitor** - Logs learning rate to W&B

### Profiling

One-time profiling on the first batch (configurable):

```yaml
train:
  profiling:
    enabled: true
    steps: 5
```

### Artifacts

Training automatically logs to W&B:

- Training/validation metrics
- Sample images
- Training curves
- Model artifacts with metadata

## Usage Examples

### Hydra Entry Point

```bash
# Default training
uv run python -m ct_scan_mlops.train

# With overrides
uv run python -m ct_scan_mlops.train model=resnet18 train.max_epochs=50
```

### Invoke Command

```bash
invoke train --args "model=resnet18"
```

### Programmatic Training

```python
from ct_scan_mlops.train import train_model, LitModel
from ct_scan_mlops.data import ChestCTDataModule
from hydra import compose, initialize

with initialize(config_path="configs"):
    cfg = compose(config_name="config")

# Create Lightning model and datamodule
lit_model = LitModel(cfg)
datamodule = ChestCTDataModule(cfg)

# Train
trainer = pl.Trainer(max_epochs=10)
trainer.fit(lit_model, datamodule=datamodule)
```
