# Repository Structure

This document provides a comprehensive index of the CT Scan MLOps repository structure.

## Table of Contents

- [Root Directory](#root-directory)
- [Source Code](#source-code-srcct_scan_mlops)
- [Configuration](#configuration-configs)
- [Tests](#tests-tests)
- [Docker](#docker-dockerfiles)
- [GitHub Actions](#github-actions-githubworkflows)
- [Documentation](#documentation-docs)
- [Data & Models](#data--models)
- [Development Tools](#development-tools)

---

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
├── docs/                   # Documentation (see below)
│   ├── GetStarted.md       # Setup guide
│   ├── COLLABORATION.md    # W&B team workflow guide
│   ├── DEPENDENCIES.md     # Dependency documentation
│   ├── TEAM_SETUP.md       # Team configuration guide
│   └── PROJECT_OVERVIEW.md # MLOps checklist mapping
│
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── .gitignore              # Git ignore rules
├── .dvcignore              # DVC ignore rules
├── .python-version         # Python version (3.12)
├── .secrets.baseline       # detect-secrets baseline
└── .dockerignore           # Docker ignore rules
```

---

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
└── visualize.py            # Visualization utilities (placeholder)
```

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | Defines `SimpleCNN` and `ResNet18Classifier` with PyTorch Lightning |
| `data.py` | `ChestCTDataModule` for data loading, transforms, train/val/test splits |
| `dataset.py` | `ChestCTDataset` custom dataset implementation |
| `train.py` | Hydra-configured training with W&B logging |
| `evaluate.py` | Evaluation metrics, confusion matrix, classification report |
| `api.py` | FastAPI app with `/predict` endpoint for inference |

---

## Configuration (`configs/`)

Hydra configuration files for flexible experiment management.

```
configs/
├── config.yaml             # Main config (imports model, data, train)
├── data/
│   └── chest_ct.yaml       # Data paths, batch size, transforms
├── model/
│   ├── cnn.yaml            # SimpleCNN hyperparameters
│   └── resnet18.yaml       # ResNet18 hyperparameters
├── train/
│   ├── default.yaml        # Default training config
│   └── sweep-training-parameters.yaml  # Sweep parameters
└── sweeps/
    └── train_sweep.yaml    # W&B sweep configuration
```

### Configuration Hierarchy

```yaml
# config.yaml structure
defaults:
  - model: cnn              # or resnet18
  - data: chest_ct
  - train: default
```

---

## Tests (`tests/`)

Comprehensive test suite using pytest.

```
tests/
├── __init__.py
├── conftest.py             # Shared fixtures
├── README.md               # Testing documentation
├── test_api.py             # API endpoint tests
├── test_config.py          # Configuration validation tests
├── test_data.py            # DataModule and dataset tests
├── test_evaluate.py        # Evaluation function tests
├── test_model.py           # Model architecture tests
├── test_train.py           # Training pipeline tests
└── test_training_logic.py  # Training logic unit tests
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_model.py` | Model instantiation, forward pass, output shapes |
| `test_data.py` | Data loading, transforms, splits |
| `test_train.py` | End-to-end training smoke tests |
| `test_api.py` | FastAPI endpoint functionality |
| `test_config.py` | Hydra configuration loading |

---

## Docker (`dockerfiles/`)

Container definitions for different environments.

```
dockerfiles/
├── api.dockerfile          # FastAPI inference server
├── train.dockerfile        # CPU training environment
└── train_cuda.dockerfile   # GPU (CUDA) training environment
```

---

## GitHub Actions (`.github/workflows/`)

CI/CD pipeline definitions.

```
.github/
├── copilot-instructions.md # GitHub Copilot instructions (synced from CLAUDE.md)
├── dependabot.yaml         # Dependency update configuration
├── labeler.yml             # PR auto-labeling rules
└── workflows/
    ├── cml_data.yaml       # CML data validation
    ├── docker-build.yaml   # Docker image build
    ├── docker-publish.yaml # Docker image publish to registry
    ├── linting.yaml        # Code linting checks
    ├── model_registry.yaml # Model registry operations
    ├── pre-commit.yaml     # Pre-commit hook checks
    ├── pre-commit-update.yaml # Auto-update pre-commit hooks
    ├── pr_labeler.yml      # PR labeler workflow
    ├── security_audit.yml  # Security vulnerability scanning
    ├── tests.yaml          # Test suite execution
    └── train_vertex.yml    # Vertex AI training trigger
```

### Workflow Purposes

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yaml` | PR, push | Run pytest suite |
| `linting.yaml` | PR, push | Ruff linting |
| `docker-build.yaml` | PR | Build and test Docker images |
| `docker-publish.yaml` | Release | Publish to container registry |
| `train_vertex.yml` | Manual | Cloud training on Vertex AI |

---

## Documentation (`docs/`)

Project and course documentation.

```
docs/
├── README.md               # Docs index
├── Structure.md            # This file
├── Feature_Extraction.md   # Feature extraction guide
├── Pre_Commit_Hooks.md     # Pre-commit setup guide
├── mkdocs.yaml             # MkDocs configuration
├── source/
│   └── index.md            # API documentation index
└── course/                 # DTU MLOps course materials
    ├── apis.md
    ├── boilerplate.md
    ├── cli.md
    ├── cloud_deployment.md
    ├── cloud_setup.md
    ├── cloud_usage.md
    ├── cml.md
    ├── code_structure.md
    ├── command_line.md
    ├── config_files.md
    ├── data_drifting.md
    ├── data_loading.md
    ├── debugging.md
    ├── deep_learning_software.md
    ├── distributed_training.md
    ├── docker.md
    ├── docker_Opt.md
    ├── documentation.md
    ├── dvc.md
    ├── editor.md
    ├── frontend.md
    ├── git.md
    ├── github_actions.md
    ├── good_coding_practice.md
    ├── hpc.md
    ├── logging.md
    ├── ml_deployment.md
    ├── package_managers.md
    ├── pre_commit.md
    ├── profiling.md
    ├── scalable_inference.md
    └── unit_testing.md
```

---

## Data & Models

### Data Directory (`data/`)

```
data/
├── .gitignore              # Ignore data files
├── raw.dvc                 # DVC tracking for raw data
├── raw/                    # Original dataset (DVC managed)
│   └── Data/               # Chest CT images by class
└── processed/              # Preprocessed data
    ├── train/
    ├── val/
    └── test/
```

### Outputs Directory (`outputs/`)

All training outputs, model checkpoints, analysis reports, and profiling data are consolidated here.

```
outputs/
├── runs/                   # Hydra training runs (timestamped)
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/     # Hydra config files
│           └── wandb/      # W&B run files
├── checkpoints/            # Trained model files (.pt, .ckpt, .onnx)
├── reports/                # Generated analysis reports
│   ├── confusion_analysis/
│   ├── diagnostics/
│   ├── error_analysis/
│   ├── feature_importance/
│   └── sweep_analysis/
├── profiling/              # Performance profiling data
├── sweeps/                 # W&B sweep results
└── logs/                   # Training logs
```

---

## Development Tools

### DVC (`.dvc/`)

Data Version Control configuration.

```
.dvc/
├── config                  # Remote storage configuration
├── cache/                  # Local DVC cache
└── tmp/                    # Temporary files
```

### DevContainer (`.devcontainer/`)

VS Code development container setup.

```
.devcontainer/
├── devcontainer.json       # Container configuration
└── post_create.sh          # Post-creation setup script
```

### Claude Configuration (`.claude/`)

AI assistant configuration and commands.

```
.claude/
├── SWEEP_PLAN.md
├── agents/
│   ├── architecture-reviewer.md
│   └── structural-completeness-reviewer.md
├── commands/
│   ├── arch-review.md
│   ├── arewedone.md
│   └── learn.md
└── docs/
    └── learnings.md
```

---

## Output Directories

All training outputs, analysis reports, and artifacts are consolidated under `outputs/` (see [Data & Models](#data--models) section above).

### W&B Directory (`wandb/`)

Weights & Biases run data and artifacts (project-level).

### Reports (`reports/`)

Exam submission template (static content, not generated outputs).

```
reports/
├── README.md               # Exam template
├── report.py               # Exam report generator
└── figures/                # Static exam figures (architecture diagrams, etc.)
```

**Note**: Generated analysis reports are now in `outputs/reports/`, not here.

---

## Quick Reference

### Essential Commands

```bash
# Development
invoke ruff          # Lint and format code
invoke test          # Run tests
invoke train         # Train model

# Data
invoke dvc-pull        # Pull data from remote
invoke preprocess-data # Process raw data

# Docker
invoke docker-build    # Build Docker image
invoke docker-run      # Run Docker container

# AI Config
invoke sync-ai-config  # Sync CLAUDE.md -> copilot-instructions.md
```

### Key Entry Points

| Task | File | Command |
|------|------|---------|
| Training | `src/ct_scan_mlops/train.py` | `invoke train` |
| Evaluation | `src/ct_scan_mlops/analysis/cli.py` | `invoke evaluate` |
| API Server | `src/ct_scan_mlops/api.py` | `invoke api` |
| Sweeps | `src/ct_scan_mlops/sweep_train.py` | `invoke sweep` |
| Analysis | `src/ct_scan_mlops/analysis/` | `invoke compare-models`, `invoke analyze-features` |

### Technology Stack

- **Language**: Python 3.12
- **ML Framework**: PyTorch + PyTorch Lightning
- **Config**: Hydra
- **Experiment Tracking**: Weights & Biases
- **Data Versioning**: DVC
- **Package Manager**: uv
- **Task Runner**: invoke
- **Linting**: Ruff
- **Testing**: pytest
- **API**: FastAPI
- **Containerization**: Docker
