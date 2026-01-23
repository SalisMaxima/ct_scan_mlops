# Training

Guide to training CT scan classification models.

## Basic Training

Train with the default configuration (CustomCNN):

```bash
invoke train
```

## Model Selection

Choose between available models:

```bash
# CustomCNN (default)
invoke train

# ResNet18 with transfer learning
invoke train --args "model=resnet18"
```

## Training Configuration

Override training parameters from the command line:

```bash
# Custom epochs
invoke train --args "train.max_epochs=50"

# Custom learning rate
invoke train --args "train.optimizer.lr=0.0001"

# Multiple overrides
invoke train --args "model=resnet18 train.max_epochs=30 train.optimizer.lr=0.001"
```

## W&B Experiment Tracking

All training runs are logged to Weights & Biases by default.

### Disable W&B Logging

```bash
invoke train --args "wandb.mode=disabled"
```

### Custom W&B Settings

```bash
# Custom experiment name
invoke train --args "experiment_name=my_experiment"

# Add tags
invoke train --args "wandb.tags=[experiment1,baseline]"
```

## Hyperparameter Sweeps

Use W&B Sweeps for automated hyperparameter optimization.

### Create and Run a Sweep

```bash
# 1. Create the sweep
invoke sweep

# 2. Start an agent (use the printed sweep ID)
invoke sweep-agent --sweep-id ENTITY/PROJECT/SWEEP_ID

# 3. Get the best parameters
invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
```

### Custom Sweep Configuration

Edit `configs/sweeps/train_sweep.yaml` to customize the search space:

```yaml
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [16, 32, 64]
```

## Training Outputs

Training outputs are saved to `outputs/{experiment_name}/{date}/{time}/`:

| File | Description |
|------|-------------|
| `best_model.ckpt` | Lightning checkpoint with best validation accuracy |
| `model.pt` | PyTorch state dict (for inference) |
| `training_curves.png` | Loss and accuracy plots |
| `training.log` | Detailed training logs |
| `.hydra/config.yaml` | Full configuration used |

## Early Stopping

Early stopping is enabled by default. Configure it in `configs/train/default.yaml`:

```yaml
early_stopping:
  enabled: true
  monitor: val_loss
  patience: 10
  mode: min
```

## GPU Training

GPU is automatically detected and used when available.

```bash
# Check available devices
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

For specific GPU configuration:

```bash
invoke train --args "train.accelerator=gpu train.devices=1"
```

## Reproducibility

Training is reproducible with the same seed:

```bash
invoke train --args "seed=42"
```

## Next Steps

- [Evaluation Guide](evaluation.md) - Evaluate trained models
- [Configuration Guide](configuration.md) - Full configuration reference
