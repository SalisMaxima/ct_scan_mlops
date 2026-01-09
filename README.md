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
3. Augmentation with linear transformations for horizontal/vertical flips and rotations, increasing model robustness through significant increases in data volume

## Models

We create our own **CNN model** that takes images and performs multi-classification as output. The CNN model serves as a baseline for comparing performance with **ResNet18** through transfer learning.

---

## Getting Started

See [GetStarted.md](GetStarted.md) for setup instructions.

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
invoke train --entity YOUR_WANDB_USERNAME                    # Train with default CNN
invoke train --entity YOUR_WANDB_USERNAME --args "model=resnet18"  # Train ResNet18
invoke train --args "wandb.mode=disabled"                    # Train without wandb
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

- **GitHub Repo:** https://github.com/SalisMaxima/ct_scan_mlops
- **Dataset:** https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
- **Course Material:** https://skaftenicki.github.io/dtu_mlops/

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
