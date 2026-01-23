# Quick Start

Essential commands to get up and running quickly.

## Environment

```bash
invoke bootstrap          # Create new venv and install deps
invoke sync               # Sync dependencies
invoke dev                # Install with dev dependencies
```

## Data Preprocessing

Before training, preprocess the raw images into tensors for faster loading:

```bash
invoke preprocess-data    # Preprocess images (run once)
```

This creates `data/processed/` with normalized tensors.

## Training

```bash
# Train with default CNN
invoke train

# Train ResNet18
invoke train --args "model=resnet18"

# Custom epochs
invoke train --args "train.max_epochs=20"

# Train without W&B logging
invoke train --args "wandb.mode=disabled"
```

## Hyperparameter Sweeps

Run W&B Sweeps for hyperparameter optimization:

```bash
# Create a sweep
invoke sweep

# Start an agent (use the printed sweep ID)
invoke sweep-agent --sweep-id ENTITY/PROJECT/SWEEP_ID

# Get best parameters
invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
```

## Code Quality

```bash
invoke ruff               # Run linter + formatter
invoke lint --fix         # Auto-fix linting issues
invoke test               # Run tests with coverage
```

## Data Management (DVC)

```bash
invoke dvc-pull           # Download data from GCS
invoke dvc-push           # Upload data to GCS
```

## Docker

```bash
invoke docker-build       # Build CPU docker images
invoke docker-build-cuda  # Build GPU docker image
invoke docker-train       # Run training in container
```

## Git Shortcuts

```bash
invoke git --message "Your commit message"  # Add, commit, push
invoke git-status                           # Show git status
```

## All Commands

See all available invoke commands:

```bash
invoke --list
```
