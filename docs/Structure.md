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
├── GEMINI.md               # Gemini CLI integration guide
├── CHANGELOG.md            # Version history and release notes
├── ToDo.md                 # Architecture improvement roadmap
│
├── Dockerfile              # Root-level Dockerfile
├── test_api.sh             # API testing script
│
├── docs/                   # Documentation (see below)
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks
│
├── artifacts/              # Build artifacts
├── cloudbuild/             # Google Cloud Build configurations
├── cloudbuild-api.yaml     # Cloud Build API configuration
│
├── logs/                   # Training and application logs
├── models/                 # Saved models (gitignored)
├── wandb/                  # W&B local run data
├── wandb_artifacts/        # W&B artifact cache
│
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── .gitignore              # Git ignore rules
├── .dvcignore              # DVC ignore rules
├── .dockerignore           # Docker ignore rules
├── .gcloudignore           # Google Cloud ignore rules
├── .python-version         # Python version (3.12)
└── .secrets.baseline       # detect-secrets baseline
```

---

## Source Code (`src/ct_scan_mlops/`)

Main application code for the ML pipeline with dual pathway architecture (v2.0).

```
src/ct_scan_mlops/
├── __init__.py             # Package initialization
│
├── model.py                # Neural network architectures (CNN, ResNet18, DualPathway)
├── data.py                 # PyTorch Lightning DataModule
├── train.py                # Training script with Hydra config
├── evaluate.py             # Model evaluation and metrics
├── losses.py               # Custom loss functions (focal loss, etc.)
│
├── api.py                  # FastAPI inference endpoint
├── inference_onnx.py       # ONNX model inference
├── onnx_export.py          # ONNX model export utility
├── promote_model.py        # Model promotion logic for registry
│
├── sweep_train.py          # W&B hyperparameter sweep training
├── sweep_best.py           # Best sweep configuration extraction
├── utils.py                # General utilities
│
├── analysis/               # ⭐ Analysis and evaluation module
│   ├── __init__.py
│   ├── cli.py              # Command-line interface for analysis
│   ├── core.py             # Core analysis functions
│   ├── comparison.py       # Model comparison utilities
│   ├── benchmark.py        # Benchmarking tools
│   ├── diagnostics.py      # Model diagnostics
│   ├── diagnose_data.py    # Data quality diagnostics
│   ├── explainability.py   # Model explainability (GradCAM, etc.)
│   ├── sweep_report.py     # W&B sweep analysis reports
│   └── utils.py            # Analysis utilities
│
├── features/               # ⭐ Feature extraction module (dual pathway)
│   ├── __init__.py
│   ├── extractor.py        # Main feature extraction pipeline
│   ├── extract_radiomics.py # Radiomics feature extraction
│   ├── intensity.py        # Intensity-based features
│   ├── shape.py            # Shape-based features
│   ├── texture.py          # Texture features (GLCM, etc.)
│   ├── wavelet.py          # Wavelet transform features
│   └── region.py           # Region-based features
│
├── monitoring/             # ⭐ Data drift monitoring
│   ├── __init__.py
│   ├── drift_api.py        # Drift monitoring API
│   ├── drift_check.py      # Drift detection logic
│   ├── drift_gaussian_noise.py # Noise injection for testing
│   └── extract_stats.py    # Statistical feature extraction
│
└── frontend/               # Streamlit frontend
    └── pages/              # Frontend pages
```

### Key Files

| File | Purpose |
|------|---------|
| `model.py` | Defines `SimpleCNN`, `ResNet18Classifier`, and `DualPathwayCNN` architectures |
| `data.py` | `ChestCTDataModule` for data loading, transforms, train/val/test splits |
| `train.py` | Hydra-configured training with W&B logging and dual pathway support |
| `evaluate.py` | Evaluation metrics, confusion matrix, classification report |
| `api.py` | FastAPI app with `/predict` endpoint for inference |
| `losses.py` | Custom loss functions including focal loss for class imbalance |
| `analysis/cli.py` | Command-line interface for model analysis and comparison |
| `features/extractor.py` | Feature extraction pipeline for dual pathway models |
| `monitoring/drift_api.py` | API for drift monitoring and alerting |

---

## Configuration (`configs/`)

Hydra configuration files for flexible experiment management, including dual pathway configs.

```
configs/
├── config.yaml                    # Main config (imports model, data, train)
├── config_production.yaml         # Production deployment config
│
├── data/
│   └── chest_ct.yaml              # Data paths, batch size, transforms
│
├── model/
│   ├── cnn.yaml                   # SimpleCNN hyperparameters
│   ├── resnet18.yaml              # ResNet18 hyperparameters
│   ├── dual_pathway.yaml          # ⭐ Dual pathway base config
│   ├── dual_pathway_top_features.yaml  # Dual pathway with top features
│   ├── dual_pathway_bn_finetune_kygevxv0.yaml  # Best sweep config
│   └── dual_pathway_bn_finetune_kygevxv0_clean.yaml  # Cleaned best config
│
├── features/                      # ⭐ Feature extraction configs
│   ├── default.yaml               # Default feature set
│   └── top_features.yaml          # Top performing features
│
├── train/
│   ├── default.yaml               # Default training config
│   └── sweep-training-parameters.yaml  # Sweep parameters
│
└── sweeps/
    ├── train_sweep.yaml           # Basic sweep configuration
    ├── dual_pathway_sweep.yaml    # ⭐ Dual pathway sweep
    ├── dual_pathway_finetune.yaml # ⭐ Dual pathway fine-tuning sweep
    └── model_comparison_sweep.yaml # Model architecture comparison
```

### Configuration Hierarchy

```yaml
# config.yaml structure
defaults:
  - model: cnn              # or resnet18, dual_pathway, dual_pathway_top_features
  - data: chest_ct
  - train: default
  - features: default       # optional, for dual pathway models
```

---

## Tests (`tests/`)

Comprehensive test suite using pytest.

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── README.md                   # Testing documentation
│
├── test_api.py                 # API endpoint tests
├── test_config.py              # Configuration validation tests
├── test_data.py                # DataModule and dataset tests
├── test_evaluate.py            # Evaluation function tests
├── test_model.py               # Model architecture tests
├── test_train.py               # Training pipeline tests
├── test_training_logic.py      # Training logic unit tests
│
├── test_analysis_utils.py      # ⭐ Analysis utilities tests
├── test_checkpoint.py          # ⭐ Checkpoint loading/saving tests
├── test_features.py            # ⭐ Feature extraction tests
├── test_frontend.py            # ⭐ Frontend tests
├── test_promote_model.py       # ⭐ Model promotion tests
│
├── locustfile.py               # ⭐ Load testing for API
└── assets/                     # Test assets (sample images, etc.)
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_model.py` | Model instantiation, forward pass, output shapes (all architectures) |
| `test_data.py` | Data loading, transforms, splits |
| `test_train.py` | End-to-end training smoke tests |
| `test_api.py` | FastAPI endpoint functionality |
| `test_config.py` | Hydra configuration loading |
| `test_features.py` | Feature extraction pipeline |
| `test_analysis_utils.py` | Analysis and evaluation utilities |
| `test_checkpoint.py` | Model checkpoint handling |
| `locustfile.py` | API load and performance testing |

---

## Docker (`dockerfiles/`)

Container definitions for different environments.

```
dockerfiles/
├── api.dockerfile              # FastAPI inference server
├── api.cloudrun.dockerfile     # ⭐ Google Cloud Run API deployment
├── drift_api.dockerfile        # ⭐ Drift monitoring API
├── train.dockerfile            # CPU training environment
└── train_cuda.dockerfile       # GPU (CUDA) training environment
```

### Dockerfile Purposes

| Dockerfile | Purpose |
|------------|---------|
| `api.dockerfile` | Standard FastAPI inference server |
| `api.cloudrun.dockerfile` | Cloud Run optimized API (port 8080) |
| `drift_api.dockerfile` | Data drift monitoring service |
| `train.dockerfile` | CPU-based training (CI/CD) |
| `train_cuda.dockerfile` | GPU training (local/cloud) |

---

## GitHub Actions (`.github/workflows/`)

CI/CD pipeline definitions.

```
.github/
├── copilot-instructions.md # GitHub Copilot instructions (synced from CLAUDE.md)
├── dependabot.yaml         # Dependency update configuration
├── labeler.yml             # PR auto-labeling rules
│
└── workflows/
    ├── tests.yaml          # Test suite execution
    ├── linting.yaml        # Code linting checks
    ├── pre-commit.yaml     # Pre-commit hook checks
    ├── pre-commit-update.yaml # Auto-update pre-commit hooks
    ├── security_audit.yml  # Security vulnerability scanning
    │
    ├── api-tests.yaml      # ⭐ API integration tests
    │
    ├── docker-build.yaml   # Docker image build
    ├── docker-publish.yaml # Docker image publish to registry
    │
    ├── cml_data.yaml       # CML data validation
    ├── model_registry.yaml # Model registry operations
    ├── promote_model.yml   # ⭐ Model promotion workflow
    │
    ├── deploy_cloudrun.yml # ⭐ Google Cloud Run deployment
    ├── deploy_docs.yaml    # ⭐ Documentation deployment
    ├── train_vertex.yml    # Vertex AI training trigger
    ├── push_wandb_model_to_gcs.yml # ⭐ W&B to GCS sync
    │
    └── pr_labeler.yml      # PR labeler workflow
```

### Workflow Purposes

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yaml` | PR, push | Run pytest suite |
| `linting.yaml` | PR, push | Ruff linting |
| `api-tests.yaml` | PR, push | API integration tests |
| `docker-build.yaml` | PR | Build and test Docker images |
| `docker-publish.yaml` | Release | Publish to container registry |
| `deploy_cloudrun.yml` | Manual/push | Deploy API to Google Cloud Run |
| `train_vertex.yml` | Manual | Cloud training on Vertex AI |
| `promote_model.yml` | Manual | Promote model to production registry |
| `push_wandb_model_to_gcs.yml` | Manual | Sync W&B models to Google Cloud Storage |

---

## Documentation (`docs/`)

Project and course documentation.

```
docs/
├── README.md                      # Docs index
├── Structure.md                   # This file
│
├── GetStarted.md                  # Setup guide
├── COLLABORATION.md               # W&B team workflow
├── DEPENDENCIES.md                # Dependency documentation
├── TEAM_SETUP.md                  # Team configuration
├── PROJECT_OVERVIEW.md            # MLOps checklist mapping
│
├── MIGRATION_DUAL_PATHWAY.md      # ⭐ Dual pathway migration guide (v2.0)
├── MIGRATION_QUICK_REFERENCE.md   # Quick reference card for migration
│
├── ExperimentPlan_Phase2.md       # Phase 2 experiment methodology
├── ExperimentAnalysis_20260129.md # Detailed results analysis
├── AdenocarcinomaImprovementPlan.md # Improvement strategy
│
├── Feature_Extraction.md          # Feature extraction guide
├── Pre_Commit_Hooks.md            # Pre-commit setup guide
│
├── DOCKER_GUIDE.md                # ⭐ Docker best practices and troubleshooting
├── DOCKER_FIXES_SUMMARY.md        # ⭐ Docker fixes documentation
├── drifting_robustness.md         # ⭐ Data drift documentation
│
├── mkdocs.yaml                    # MkDocs configuration
├── site/                          # ⭐ Built documentation site
├── source/
│   └── index.md                   # API documentation index
│
└── course/                        # DTU MLOps course materials
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
│
├── checkpoints/            # Trained model files (.pt, .ckpt, .onnx)
│
├── reports/                # Generated analysis reports
│   ├── confusion_analysis/
│   ├── diagnostics/
│   ├── error_analysis/
│   ├── feature_importance/
│   └── sweep_analysis/
│
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

# Sweeps (hyperparameter optimization)
invoke extract-features --features top_features  # Prepare features for dual pathway sweeps
invoke prepare-sweep-features                    # Extract all feature configs for sweeps
invoke sweep                                     # Create W&B sweep
invoke sweep-agent <SWEEP_ID>                    # Run sweep agent

# Analysis commands (use after training)
invoke compare-baselines --baseline path/to/baseline.ckpt --improved path/to/improved.ckpt
invoke analyze-features --checkpoint path/to/model.ckpt
invoke analyze-errors --checkpoint path/to/model.ckpt

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
| Feature Extraction | `src/ct_scan_mlops/features/extractor.py` | `invoke extract-features` |
| Drift Monitoring | `src/ct_scan_mlops/monitoring/drift_api.py` | `invoke drift-monitor` |

### Technology Stack

- **Language**: Python 3.12
- **ML Framework**: PyTorch + PyTorch Lightning
- **Architecture**: Dual Pathway CNN (v2.0) with feature engineering
- **Config**: Hydra
- **Experiment Tracking**: Weights & Biases
- **Data Versioning**: DVC
- **Package Manager**: uv
- **Task Runner**: invoke
- **Linting**: Ruff
- **Testing**: pytest + Locust (load testing)
- **API**: FastAPI
- **Containerization**: Docker
- **Cloud Platform**: Google Cloud (Vertex AI, Cloud Run, GCS)
- **Model Format**: ONNX (for inference optimization)

---

## Version 2.0 Highlights

The repository has evolved significantly with the dual pathway architecture:

- **Dual Pathway Models**: Combines CNN features with engineered radiomics features
- **Feature Engineering**: Comprehensive feature extraction pipeline (intensity, texture, shape, wavelet)
- **Advanced Analysis**: Model diagnostics, comparison tools, explainability (GradCAM)
- **Production Ready**: ONNX export, Cloud Run deployment, drift monitoring
- **Enhanced Testing**: Feature tests, checkpoint tests, load testing with Locust

See `docs/MIGRATION_DUAL_PATHWAY.md` for detailed migration guide.
