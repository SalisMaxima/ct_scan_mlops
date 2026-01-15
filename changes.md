# Changes Summary: MLOps Best Practices Refactor

## Overview

This refactor aligns the codebase with production MLOps standards, adding proper configuration management, experiment tracking, data preprocessing, and reproducibility features.

---

## Files Modified

| File | Lines Changed | Summary |
|------|--------------|---------|
| `src/ct_scan_mlops/model.py` | +211 | Configurable CNN, ResNet18, build_model(cfg) |
| `src/ct_scan_mlops/data.py` | +482 | Preprocessing, ProcessedDataset, Albumentations |
| `src/ct_scan_mlops/train.py` | +388 | Hydra, W&B, Loguru, reproducibility, early stopping |

---

## What Changed

### 1. model.py - Configurable Model Architecture

**Before:** Fixed architecture with hardcoded layer sizes (16->32->64 channels).

**After:** Fully configurable CNN that reads parameters from Hydra config files.

#### New Features:
- `CustomCNN` accepts configurable parameters:
  - `hidden_dims`: List of channel sizes (e.g., `[32, 64, 128, 256]`)
  - `fc_hidden`: Fully connected layer size
  - `dropout`: Dropout rate
  - `batch_norm`: Whether to use BatchNorm layers
  - `kernel_size`: Convolution kernel size

- `ResNet18` class with transfer learning support:
  - `freeze_backbone()` / `unfreeze_backbone()` methods
  - Pretrained weights loading

- `build_model(cfg)` factory function that reads from Hydra config

---

### 2. data.py - Preprocessing Pipeline & Fast Data Loading

**Before:** Loaded raw images on-the-fly with basic transforms.

**After:** Two-stage pipeline with preprocessing and fast tensor loading.

#### New Features:

| Function/Class | Description |
|----------------|-------------|
| `preprocess()` | Loads raw images, resizes, normalizes, saves as `.pt` tensors |
| `ProcessedChestCTDataset` | Fast dataset loading from pre-computed tensors |
| `ChestCTDataset` | On-the-fly loading with Albumentations transforms |
| `get_transforms()` | Configurable augmentation pipeline |
| `create_dataloaders(cfg)` | Auto-detects processed data, creates DataLoaders |
| `normalize()` | Normalizes tensors to mean=0, std=1 |

#### Preprocessing Output:
```
data/processed/
├── train_images.pt   # Normalized tensors
├── train_labels.pt
├── valid_images.pt
├── valid_labels.pt
├── test_images.pt
├── test_labels.pt
└── stats.pt          # Mean, std, metadata
```

---

### 3. train.py - Full MLOps Training Pipeline

**Before:** Simple Typer CLI with print statements, no tracking.

**After:** Production-grade training with Hydra, W&B, logging, and reproducibility.

#### New Features:

| Feature | Description |
|---------|-------------|
| Hydra Config | All hyperparameters from YAML files, CLI overrides |
| Weights & Biases | Metrics, images, model artifacts, experiment tracking |
| Loguru Logging | Structured logs to file + console |
| Reproducibility | Seed setting for torch, numpy, random |
| Device Detection | CUDA -> MPS (Apple Silicon) -> CPU priority |
| Early Stopping | Configurable patience and monitoring |
| LR Scheduler | Cosine annealing with configurable parameters |
| Gradient Clipping | Prevents exploding gradients |
| Model Artifacts | Saved to W&B with full metadata |
| Training Curves | Automatic plot generation |

---

## Project Pipeline

```
1. DOWNLOAD DATA
   $ dvc pull
   -> data/raw/chest-ctscan-images/Data/{train,valid,test}/

2. PREPROCESS (run once)
   $ invoke preprocess-data
   -> data/processed/*.pt (normalized tensors)

3. TRAIN MODEL
   $ python src/ct_scan_mlops/train.py
   -> outputs/{experiment}/{timestamp}/
      ├── model.pt, best_model.pt
      ├── training.log
      └── training_curves.png
   -> W&B: metrics, artifacts, images
```

---

## How to Use

### Preprocess Data (Run Once)

```bash
# Using invoke
invoke preprocess-data

# Or directly
python -m ct_scan_mlops.data --image-size 224 --output-dir data/processed
```

### Train Model

```bash
# Basic training (uses configs/config.yaml defaults)
python src/ct_scan_mlops/train.py

# Quick test (2 epochs)
python src/ct_scan_mlops/train.py train.max_epochs=2

# Use ResNet18 instead of CNN
python src/ct_scan_mlops/train.py model=resnet18

# Change learning rate
python src/ct_scan_mlops/train.py train.optimizer.lr=0.0001

# Disable W&B (offline mode)
python src/ct_scan_mlops/train.py wandb.mode=disabled

# Combine multiple overrides
python src/ct_scan_mlops/train.py model=resnet18 train.max_epochs=10 train.optimizer.lr=0.0005

# profiler for trainer
python tensorboard --logdir tb_profiler

# Using invoke with W&B entity
invoke train --entity your-wandb-username
invoke train --entity your-wandb-username --args "model=resnet18"
```

---

## Configuration Files

All hyperparameters are in YAML files under `configs/`:

### configs/config.yaml (Main)
```yaml
defaults:
  - model: cnn
  - data: chest_ct
  - train: default

seed: 42
experiment_name: ct_scan_classifier

wandb:
  project: CT_Scan_MLOps
  mode: online  # or "disabled"
```

### configs/model/cnn.yaml
```yaml
name: custom_cnn
num_classes: 4
hidden_dims: [32, 64, 128, 256]
fc_hidden: 512
dropout: 0.3
batch_norm: true
```

### configs/model/resnet18.yaml
```yaml
name: resnet18
num_classes: 4
pretrained: true
freeze_backbone: false
```

### configs/train/default.yaml
```yaml
max_epochs: 50
min_epochs: 10
optimizer:
  lr: 0.001
  weight_decay: 0.0001
early_stopping:
  enabled: true
  patience: 10
gradient_clip_val: 1.0
```

### configs/data/chest_ct.yaml
```yaml
batch_size: 32
num_workers: 4
image_size: 224
normalize:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
augmentation:
  train:
    horizontal_flip: true
    vertical_flip: true
    rotation_limit: 15
```

---

## Data Compatibility

The pipeline handles the Kaggle chest CT scan dataset:

| Property | Raw Data | After Preprocessing |
|----------|----------|---------------------|
| Image sizes | Variable (300x200 to 940x627) | Fixed 224x224 |
| Color mode | RGBA/RGB | RGB (3 channels) |
| Format | PNG | PyTorch tensors |
| Normalization | None | mean=0, std=1 |

---

## Key Improvements

| Before | After |
|--------|-------|
| Hardcoded hyperparameters | YAML configs with CLI overrides |
| Print statements | Structured logging (file + console) |
| No experiment tracking | Full W&B integration |
| No reproducibility | Seeded random, numpy, torch |
| Slow image loading | Preprocessed tensor loading |
| No normalization | Dataset-specific normalization |
| No augmentation | Albumentations pipeline |
| No early stopping | Configurable patience |
| Fixed architecture | Fully configurable CNN |

---

## TODO: Profiler Improvements (commit bb59610)

The torch-tb-profiler was added in commit `bb59610`. The following issues should be addressed when time permits:

| Issue | Priority | Description |
|-------|----------|-------------|
| Hardcoded profiler | Medium | Profiler always runs on first batch - add config option `train.profiler.enabled: false` |
| Print statement | Low | Uses `print()` instead of `logger.info()` - inconsistent with rest of codebase |
| Hardcoded output path | Medium | `"tb_profiler"` hardcoded, should respect Hydra's `output_dir` |
| CPU-only profiling | Low | Only uses `ProfilerActivity.CPU`, add CUDA support when GPU available |

### Suggested Fix

Add to `configs/train/default.yaml`:
```yaml
profiler:
  enabled: false
  output_dir: ${output_dir}/profiler
  activities: [cpu]  # or [cpu, cuda]
```

Then make `training_step` check `self.cfg.train.profiler.enabled` before running profiler.

---

## TODO: Use Lightning's Built-in Profiler (Best Practice)

The current profiler implementation uses a custom `torch.profiler` in `training_step`. PyTorch Lightning recommends using the Trainer's built-in profiler argument instead.

### Current Implementation (in `training_step`)
```python
if batch_idx == 0 and self.current_epoch == 0:
    with profile(activities=[ProfilerActivity.CPU], ...):
        # custom profiling code
```

### Recommended Implementation
```python
from pytorch_lightning.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath=output_path,
    filename="profile",
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path / "tb_profiler")),
)

trainer = pl.Trainer(
    ...
    profiler=profiler,  # or profiler="simple" / "advanced" for basic profiling
)
```

### Benefits
- Cleaner separation of concerns (profiling config outside model code)
- Configurable via Hydra without code changes
- Supports multiple profiler types (`simple`, `advanced`, `pytorch`, `xla`)
- Automatic integration with Lightning's training loop
