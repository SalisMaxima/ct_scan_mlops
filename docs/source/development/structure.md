# Project Structure

Complete index of the CT Scan MLOps repository structure.

## Root Directory

```
ct_scan_mlops/
├── pyproject.toml          # Project metadata, dependencies, tool configs
├── tasks.py                # Invoke task definitions (CLI commands)
├── uv.lock                 # Locked dependencies (uv package manager)
├── LICENSE                 # MIT License
│
├── README.md               # Main project documentation
├── CLAUDE.md               # AI assistant instructions (source of truth)
│
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── .gitignore              # Git ignore rules
├── .dvcignore              # DVC ignore rules
├── .python-version         # Python version (3.12)
└── .dockerignore           # Docker ignore rules
```

## Source Code (`src/ct_scan_mlops/`)

Main application code for the ML pipeline.

```
src/ct_scan_mlops/
├── __init__.py             # Package initialization
├── model.py                # Neural network architectures (CNN, ResNet18)
├── data.py                 # PyTorch Lightning DataModule
├── dataset.py              # Custom PyTorch Dataset classes
├── train.py                # Training script with Hydra config
├── evaluate.py             # Model evaluation and metrics
├── api.py                  # FastAPI inference endpoint
├── get_data.py             # Data download utilities
├── sweep_train.py          # W&B hyperparameter sweep training
├── sweep_best.py           # Best sweep configuration extraction
├── utils.py                # Shared utility functions
└── visualize.py            # Visualization utilities
```

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | Defines `CustomCNN` and `ResNet18` |
| `data.py` | `ChestCTDataModule` for data loading |
| `train.py` | Hydra-configured training with W&B |
| `evaluate.py` | Evaluation metrics, confusion matrix |
| `api.py` | FastAPI app with `/predict` endpoint |

## Configuration (`configs/`)

Hydra configuration files.

```
configs/
├── config.yaml             # Main config (imports model, data, train)
├── data/
│   └── chest_ct.yaml       # Data paths, batch size, transforms
├── model/
│   ├── cnn.yaml            # CustomCNN hyperparameters
│   └── resnet18.yaml       # ResNet18 hyperparameters
├── train/
│   └── default.yaml        # Default training config
└── sweeps/
    └── train_sweep.yaml    # W&B sweep configuration
```

## Tests (`tests/`)

Test suite using pytest.

```
tests/
├── __init__.py
├── conftest.py             # Shared fixtures
├── test_api.py             # API endpoint tests
├── test_config.py          # Configuration validation tests
├── test_data.py            # DataModule and dataset tests
├── test_evaluate.py        # Evaluation function tests
├── test_model.py           # Model architecture tests
└── test_train.py           # Training pipeline tests
```

## Docker (`dockerfiles/`)

Container definitions.

```
dockerfiles/
├── api.dockerfile          # FastAPI inference server
├── train.dockerfile        # CPU training environment
└── train_cuda.dockerfile   # GPU (CUDA) training environment
```

## GitHub Actions (`.github/workflows/`)

CI/CD pipeline definitions.

```
.github/workflows/
├── cml_data.yaml           # CML data validation
├── docker-build.yaml       # Docker image build
├── docker-publish.yaml     # Docker image publish
├── linting.yaml            # Code linting checks
├── pre-commit.yaml         # Pre-commit hook checks
├── tests.yaml              # Test suite execution
└── deploy_docs.yaml        # Documentation deployment
```

## Data & Models

### Data Directory (`data/`)

```
data/
├── raw.dvc                 # DVC tracking for raw data
├── raw/                    # Original dataset (DVC managed)
│   └── chest-ctscan-images/
└── processed/              # Preprocessed tensors
    ├── train_images.pt
    ├── train_labels.pt
    ├── valid_images.pt
    ├── valid_labels.pt
    ├── test_images.pt
    ├── test_labels.pt
    └── stats.pt
```

### Models Directory (`models/`)

```
models/
├── .gitkeep                # Placeholder
├── best_model.ckpt         # Lightning checkpoint (after training)
└── model.pt                # PyTorch state dict (for inference)
```

## Documentation (`docs/`)

```
docs/
├── mkdocs.yaml             # MkDocs configuration
├── source/                 # Documentation source files
│   ├── index.md
│   ├── getting-started/
│   ├── user-guide/
│   ├── api-reference/
│   └── development/
├── GetStarted.md           # Legacy setup guide
├── Structure.md            # This file (legacy)
└── COLLABORATION.md        # W&B guide (legacy)
```

## Key Entry Points

| Task | File | Command |
|------|------|---------|
| Training | `src/ct_scan_mlops/train.py` | `invoke train` |
| Evaluation | `src/ct_scan_mlops/evaluate.py` | `invoke evaluate` |
| API Server | `src/ct_scan_mlops/api.py` | `invoke api` |
| Sweeps | `src/ct_scan_mlops/sweep_train.py` | `invoke sweep` |
| Preprocessing | `src/ct_scan_mlops/data.py` | `invoke preprocess-data` |
