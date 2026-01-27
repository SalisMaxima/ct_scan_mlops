# ct_scan_mlops

Chest CT scan multi-classification model for lung tumor detection using PyTorch Lightning.

---

## Project Goal

The overall goal of this project is to build an image multi-classification model that can detect whether a chest CT scan shows signs of three different types of tumor:

1. **Adenocarcinoma** (left lower lobe)
2. **Large cell carcinoma** (left hilum)
3. **Squamous cell carcinoma** (left hilum)
4. **Normal**

We use the Kaggle dataset [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images). In the end, we will have our own model architecture for comparison with ResNet18. Additionally, usage of course material throughout the project is as vast and inclusive as possible to demonstrate knowledge and understanding of MLOps concepts.

## Framework

We use **PyTorch** as the deep learning framework since it is one of the most flexible, widely used frameworks and the de facto academic standard. We use **PyTorch Lightning** to replace as much boilerplate code as possible.

## Data

**Source:** [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle

The dataset contains ~1000 images with somewhat balanced volume across 4 classifications. We apply data augmentation with rotations and translations to make our model more robust to variations in scan positioning.

**Preprocessing:**

1. Standardize data to the correct file format (nearly all images are PNG while 12 images are JPEG - these are converted to PNG)
2. Data is pre-split into training (70%), validation (20%), and test (10%) sets to minimize data leakage

## Models

We create our own **CNN model** that takes images and performs multi-classification as output. The CNN model serves as a baseline for comparing performance with **ResNet18** through transfer learning.

---

## Getting Started

See [GetStarted.md](docs/GetStarted.md) for setup instructions.

For a complete list of all project dependencies, see [DEPENDENCIES.md](docs/DEPENDENCIES.md).

---

## Team Collaboration

This project uses **Weights & Biases** for experiment tracking with team collaboration.

**For Team Members:** See [COLLABORATION.md](docs/COLLABORATION.md) for:

- How to join the W&B team
- Running training with automatic attribution
- Viewing and comparing runs
- Best practices

**Quick Start:**

1. Accept W&B team invitation email
2. Run: `wandb team` (verify you're in `mathiashl-danmarks-tekniske-universitet-dtu`)
3. Train: `invoke train` (entity already configured)

**Team Dashboard:** https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps

---

## Quick Reference Commands

All commands use [invoke](https://www.pyinvoke.org/). Run `invoke --list` to see all available commands.

### Environment

```bash
invoke bootstrap          # Create new venv and install deps
invoke sync               # Sync dependencies
invoke dev                # Install with dev dependencies
```

### Data Preprocessing

```bash
invoke preprocess-data                            # Preprocess images (run once)
```

Creates `data/processed/` with normalized tensors for fast training.

### Training

```bash
invoke train                                      # Train with default CNN
invoke train --args "model=resnet18"              # Train ResNet18
invoke train --args "train.max_epochs=20"         # Custom epochs
invoke train --args "wandb.mode=disabled"         # Train without W&B logging
```

### Hyperparameter Sweeps (W&B)

We support W&B Sweeps for hyperparameter search.

**Why there is a separate sweep entrypoint**

W&B agents typically launch training with CLI flags like `--lr=1e-3` and `--batch-size=32`.
Hydra expects overrides like `train.optimizer.lr=1e-3` and `data.batch_size=32`.

To bridge this, the repo includes a sweep-compatible entrypoint (`ct_scan_mlops.sweep_train`) that:

- Accepts sweep parameters as normal CLI options (e.g., `--lr`, `--batch-size`, `--model`)
- Translates them into Hydra overrides
- Calls the existing Lightning training pipeline

**How to run**

```bash
# 1) Create the sweep (prints the sweep id)
invoke sweep

# 2) Start an agent (replace with the printed id)
invoke sweep-agent --sweep-id ENTITY/PROJECT/SWEEP_ID
# 3) Get the best parameters
invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
```

If you want to run sweeps under your own account/project:

```bash
invoke sweep --entity YOUR_USERNAME
invoke sweep --entity YOUR_USERNAME --project YOUR_PROJECT
```

**Where it is defined**

- Sweep config: `configs/sweeps/train_sweep.yaml`
- Sweep entrypoint: `src/ct_scan_mlops/sweep_train.py`

Important: the sweep YAML runs training via `uv run ...` to ensure the local package is importable on Windows.
If you edit the `command:` section, keep `uv run`.

**How to edit/customize the sweep**

1. Edit the search space in `configs/sweeps/train_sweep.yaml` under `parameters:`.
   - Example: add `dropout`, `fc_hidden`, augmentation params, etc.
2. Make sure every new parameter you add is supported by the sweep entrypoint.
   - If you add a new parameter in the YAML, you must add a matching CLI option in `src/ct_scan_mlops/sweep_train.py`
     and map it to the correct Hydra key (e.g., `--dropout` → `model.dropout=...`).
3. Re-create the sweep after editing the YAML (sweeps are immutable once created):

```bash
invoke sweep
```

**Notes / tips**

- Sweeps disable the one-time PyTorch profiling run by default (the `ct_scan_mlops.sweep_train` entrypoint sets
  `train.profiling.enabled=false`) because it slows down hyperparameter search.
- The metric used by the sweep is `val_acc` (logged by Lightning).

**How to find the best parameters**

- In the W&B UI: open the sweep → sort runs by `val_acc`.
- From the terminal (prints best run + config as JSON):

```bash
invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
```

**Running Sweeps with Dual Pathway Models**

The sweep system now supports the dual pathway model (CNN + Radiomics). Before starting sweeps with dual pathway models, you must extract features.

Prerequisites:

```bash
# Extract top 16 features (recommended for sweeps)
invoke extract-features --features top_features

# Or extract all 50 features
invoke extract-features

# Or prepare all feature configs at once
invoke prepare-sweep-features
```

Available sweep configurations:

- `configs/sweeps/train_sweep.yaml` - General sweep (all models including dual_pathway)
- `configs/sweeps/dual_pathway_sweep.yaml` - Optimized for dual pathway only (Bayesian optimization)
- `configs/sweeps/model_comparison_sweep.yaml` - Compare CNN vs ResNet18 vs dual pathway (Grid search)

Examples:

```bash
# Create sweep with dual pathway optimization
invoke sweep --sweep-config configs/sweeps/dual_pathway_sweep.yaml

# Or use the default sweep (includes all models)
invoke sweep

# Run agents
invoke sweep-agent --sweep-id <SWEEP_ID>
```

Troubleshooting:

- **Error: "Features not found"** - Run `invoke extract-features --features top_features` first
- **Error: "Dimension mismatch"** - Re-extract features with correct config matching your model

### Code Quality

```bash
invoke ruff               # Run linter + formatter
invoke lint --fix         # Auto-fix linting issues
invoke test               # Run tests with coverage
```

### Data Management (DVC)

```bash
invoke dvc-pull           # Download data from GCS
invoke dvc-push           # Upload data to GCS
```

### Docker

```bash
invoke docker-build       # Build CPU docker images
invoke docker-build-cuda  # Build GPU docker image
invoke docker-train       # Run training in container
```

### Git Shortcuts

```bash
invoke git --message "Your commit message"  # Add, commit, push
invoke git-status                           # Show git status
```

---

## Configuration (Hydra)

Configs are in `configs/`. Override any parameter from command line:

```bash
# Examples (replace YOUR_USERNAME with your wandb username)
python -m ct_scan_mlops.train wandb.entity=YOUR_USERNAME model=resnet18
python -m ct_scan_mlops.train wandb.entity=YOUR_USERNAME train.max_epochs=100
python -m ct_scan_mlops.train wandb.mode=disabled  # Disable W&B logging
```

### Config Structure

```
configs/
├── config.yaml           # Main config
├── model/
│   ├── cnn.yaml          # Custom CNN (default)
│   └── resnet18.yaml     # ResNet18 transfer learning
├── data/
│   └── chest_ct.yaml     # Dataset settings
└── train/
    └── default.yaml      # Training hyperparameters
```

---

## Project Structure

```
ct_scan_mlops/
├── configs/                  # Hydra configuration files
├── data/
│   ├── raw/                  # Original dataset (DVC tracked)
│   └── processed/            # Preprocessed data
├── dockerfiles/
│   ├── train.dockerfile      # CPU training
│   ├── train_cuda.dockerfile # GPU training
│   └── api.dockerfile        # API serving
├── models/                   # Saved model checkpoints
├── reports/
│   └── README.md             # PROJECT REPORT (for grading!)
├── src/ct_scan_mlops/
│   ├── data.py               # Data loading & preprocessing
│   ├── model.py              # Model architectures
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── api.py                # FastAPI inference API
├── tests/                    # Unit tests
├── pyproject.toml            # Project dependencies
├── tasks.py                  # Invoke commands
└── uv.lock                   # Locked dependencies
```

---

## Workflow for Contributing

1. **Pull latest changes:**

   ```bash
   git pull
   dvc pull  # If data changed
   ```

2. **Create a branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and test:**

   ```bash
   invoke ruff    # Format code
   invoke test    # Run tests
   ```

4. **Commit and push:**

   ```bash
   git add .
   git commit -m "Add your feature"
   git push -u origin feature/your-feature-name
   ```

5. **Create Pull Request** on GitHub

---

## Links

- **GitHub Repo:** [https://github.com/DTU-MLOps-Group-2/ct_scan_mlops](https://github.com/DTU-MLOps-Group-2/ct_scan_mlops)
- **Dataset:** [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Course Material:** [https://skaftenicki.github.io/dtu_mlops/](https://skaftenicki.github.io/dtu_mlops/)

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
