# Configuration

Guide to the Hydra configuration system.

## Overview

The project uses [Hydra](https://hydra.cc/) for configuration management. Configs are composable YAML files in the `configs/` directory.

## Config Structure

```
configs/
├── config.yaml           # Main config (imports model, data, train)
├── data/
│   └── chest_ct.yaml     # Dataset settings
├── model/
│   ├── cnn.yaml          # CustomCNN hyperparameters
│   └── resnet18.yaml     # ResNet18 hyperparameters
├── train/
│   └── default.yaml      # Training hyperparameters
└── sweeps/
    └── train_sweep.yaml  # W&B sweep configuration
```

## Main Config

The main `config.yaml` composes other configs:

```yaml
defaults:
  - model: cnn              # or resnet18
  - data: chest_ct
  - train: default

experiment_name: ct_scan_classifier
seed: 42

paths:
  data_dir: data/raw
  output_dir: outputs

wandb:
  project: CT_Scan_MLOps
  entity: mathiashl-danmarks-tekniske-universitet-dtu
```

## Model Configs

### CustomCNN (`configs/model/cnn.yaml`)

```yaml
name: custom_cnn
num_classes: 4
hidden_dims: [32, 64, 128, 256]
fc_hidden: 512
dropout: 0.3
batch_norm: true
```

### ResNet18 (`configs/model/resnet18.yaml`)

```yaml
name: resnet18
num_classes: 4
pretrained: true
freeze_backbone: false
```

## Data Config

`configs/data/chest_ct.yaml`:

```yaml
batch_size: 32
num_workers: 4
image_size: 224
processed_path: data/processed

normalize:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

augmentation:
  train:
    horizontal_flip: true
    vertical_flip: false
    rotation_limit: 15
    brightness_limit: 0.1
    contrast_limit: 0.1
```

## Training Config

`configs/train/default.yaml`:

```yaml
max_epochs: 50
gradient_clip_val: 1.0

optimizer:
  lr: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

scheduler:
  eta_min: 0.00001

checkpoint:
  monitor: val_acc
  mode: max
  save_top_k: 1
  save_last: true

early_stopping:
  enabled: true
  monitor: val_loss
  patience: 10
  mode: min
```

## Command Line Overrides

Override any config value from the command line:

```bash
# Single override
invoke train --args "train.max_epochs=100"

# Multiple overrides
invoke train --args "model=resnet18 train.optimizer.lr=0.0001"

# Nested overrides
invoke train --args "train.early_stopping.patience=20"

# List overrides
invoke train --args "model.hidden_dims=[64,128,256,512]"
```

## Multirun (Grid Search)

Run multiple configurations:

```bash
uv run python -m ct_scan_mlops.train --multirun \
    model=cnn,resnet18 \
    train.optimizer.lr=0.001,0.0001
```

## Environment Variables

Some settings can be overridden via environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `CONFIG_PATH` | Config file path | `configs/config.yaml` |
| `MODEL_PATH` | Model checkpoint path | `models/model.pt` |
| `WANDB_MODE` | W&B mode | `online` |

## Adding New Configs

1. Create a new YAML file in the appropriate directory
2. Add it to `defaults` in `config.yaml` or reference it from command line

Example: Add a new model config `configs/model/efficientnet.yaml`:

```yaml
name: efficientnet
num_classes: 4
pretrained: true
```

Use it:

```bash
invoke train --args "model=efficientnet"
```
