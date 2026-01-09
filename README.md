# ct_scan_mlops

Chest CT scan multi-classification model for lung tumor detection using PyTorch Lightning.

**Dataset:** [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) (4 classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal)

---

## Getting Started (For Group Members)

### Prerequisites

Make sure you have these installed:

| Tool | Installation |
|------|--------------|
| **Python 3.12** | [python.org](https://www.python.org/downloads/) |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Linux/Mac) or `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| **Git** | [git-scm.com](https://git-scm.com/downloads) |
| **DVC** | Installed automatically with project dependencies |
| **GCP Access** | Ask Mathias for access to the GCP project |

### Step 1: Clone the Repository

```bash
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops
```

### Step 2: Set Up Environment

```bash
# Create virtual environment and install all dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

uv sync --all-groups
```

### Step 3: Authenticate with GCP (for DVC)

```bash
# Login to Google Cloud (one-time setup)
gcloud auth login
gcloud auth application-default login
```

### Step 4: Pull the Dataset

```bash
# Download the chest CT scan dataset from GCS
dvc pull
```

This downloads ~120MB of CT scan images to `data/raw/`.

### Step 5: Verify Setup

```bash
# Check everything works
invoke python      # Should show Python 3.12
invoke test        # Run tests (may have placeholder tests initially)
```

---

## Quick Reference Commands

All commands use [invoke](https://www.pyinvoke.org/). Run `invoke --list` to see all available commands.

### Environment

```bash
invoke bootstrap          # Create new venv and install deps
invoke sync               # Sync dependencies
invoke dev                # Install with dev dependencies
```

### Training

```bash
invoke train                              # Train with default config (custom CNN)
invoke train --args "model=resnet18"      # Train with ResNet18
invoke train --args "train.max_epochs=100"  # Override hyperparameters
```

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
# Examples
python -m ct_scan_mlops.train model=resnet18
python -m ct_scan_mlops.train train.max_epochs=100 data.batch_size=64
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

## Troubleshooting

### "Module not found" errors
```bash
uv sync --all-groups  # Reinstall all dependencies
```

### DVC authentication issues
```bash
gcloud auth application-default login
```

### CUDA/GPU not detected
Make sure you have NVIDIA drivers installed and use `invoke docker-build-cuda` for GPU training.

### Windows-specific issues
- Use PowerShell or Git Bash (not CMD)
- Replace `source .venv/bin/activate` with `.venv\Scripts\activate`

---

## Links

- **GitHub Repo:** https://github.com/SalisMaxima/ct_scan_mlops
- **Dataset:** https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- **Course Material:** https://skaftenicki.github.io/dtu_mlops/

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
