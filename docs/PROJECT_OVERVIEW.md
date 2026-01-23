# CT Scan MLOps - Project Overview & Checklist Mapping

This document maps our project to the [DTU MLOps Checklist](https://skaftenicki.github.io/dtu_mlops/pages/projects/) to help you navigate the codebase and understand what we've built.

---

## 🎯 Project Goal

**Task:** Chest CT scan multi-classification for lung tumor detection
**Classes:** Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal
**Dataset:** [Kaggle Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) (~1000 images)
**Models:** Custom CNN (baseline) + ResNet18 (transfer learning)

---

## 📁 Project Structure Quick Map

```
ct_scan_mlops/
├── 📝 Configuration & Setup
│   ├── pyproject.toml           # Dependencies (uv package manager)
│   ├── tasks.py                 # All invoke commands
│   ├── .pre-commit-config.yaml  # Pre-commit hooks
│   └── configs/                 # Hydra configs (model/data/training)
│
├── 🔬 ML Code
│   └── src/ct_scan_mlops/
│       ├── data.py              # Data loading & preprocessing
│       ├── dataset.py           # PyTorch Dataset
│       ├── model.py             # CNN & ResNet18 models
│       ├── train.py             # Training script (PyTorch Lightning)
│       ├── evaluate.py          # Evaluation script
│       ├── sweep_train.py       # W&B sweep entrypoint
│       ├── api.py               # FastAPI inference server
│       └── visualize.py         # Visualization utilities
│
├── 🧪 Testing & Quality
│   └── tests/
│       ├── test_data.py         # Data loading tests
│       ├── test_model.py        # Model architecture tests
│       ├── test_train.py        # Training logic tests
│       └── test_api.py          # API endpoint tests
│
├── 🐳 Deployment
│   └── dockerfiles/
│       ├── train.dockerfile      # CPU training
│       ├── train_cuda.dockerfile # GPU training
│       └── api.dockerfile        # API serving
│
├── 🔄 CI/CD
│   └── .github/workflows/
│       ├── tests.yaml           # Multi-OS unit tests
│       ├── linting.yaml         # Ruff linting
│       ├── docker-build.yaml    # Docker image builds
│       ├── pre-commit.yaml      # Pre-commit CI
│       └── cml_data.yaml        # Data validation
│
├── 📊 Data & Artifacts
│   ├── data/
│   │   ├── raw/                 # Original images (DVC tracked)
│   │   └── processed/           # Preprocessed tensors
│   ├── models/                  # Saved checkpoints
│   ├── artifacts/profiling/     # PyTorch profiler output
│   └── outputs/                 # Hydra run outputs
│
└── 📖 Documentation
    ├── README.md                # Main documentation
    ├── GetStarted.md            # Setup guide
    ├── COLLABORATION.md         # W&B team workflow
    ├── CLAUDE.md                # Quick reference for AI
    └── reports/                 # Project report (for grading)
```

---

## ✅ MLOps Checklist Mapping

### 🟢 **Week 1: Foundation & Setup** (COMPLETED)

| Requirement | Implementation | Location |
|------------|----------------|----------|
| **Git repository** | ✅ GitHub with team access | https://github.com/SalisMaxima/ct_scan_mlops |
| **Dependencies** | ✅ uv (modern pip replacement) | `pyproject.toml`, `uv.lock` |
| **Project structure** | ✅ Cookiecutter template | Root directory |
| **Data pipeline** | ✅ PyTorch Lightning DataModule | `src/ct_scan_mlops/data.py` |
| **Model** | ✅ CNN + ResNet18 | `src/ct_scan_mlops/model.py` |
| **Training** | ✅ PyTorch Lightning Trainer | `src/ct_scan_mlops/train.py` |
| **Code quality** | ✅ Ruff (linting + formatting) | `.pre-commit-config.yaml` |
| **Data versioning** | ✅ DVC with GCS backend | `.dvc/`, `data.dvc` |
| **CLI** | ✅ Invoke tasks + Hydra configs | `tasks.py`, `configs/` |
| **Docker** | ✅ CPU & GPU images | `dockerfiles/` |
| **Config files** | ✅ Hydra (model/data/train) | `configs/` |
| **Profiling** | ✅ PyTorch Profiler integration | `train.py:L85-L95`, `artifacts/profiling/` |
| **Experiment tracking** | ✅ Weights & Biases | Team: `mathiashl-danmarks-tekniske-universitet-dtu` |

**Key Commands:**
```bash
invoke --list          # See all commands
invoke train           # Train model (W&B enabled)
invoke ruff            # Lint + format
invoke dvc-pull        # Get data from GCS
```

---

### 🟢 **Week 2: Testing & Deployment** (COMPLETED)

| Requirement | Implementation | Location |
|------------|----------------|----------|
| **Unit tests** | ✅ pytest (data, model, training) | `tests/` |
| **Code coverage** | ✅ pytest-cov | CI: `.github/workflows/tests.yaml:L56` |
| **CI** | ✅ GitHub Actions (multi-OS) | `.github/workflows/tests.yaml` |
| **Caching** | ✅ uv cache in CI | `.github/workflows/tests.yaml:L40` |
| **Pre-commit hooks** | ✅ Ruff + secrets check | `.pre-commit-config.yaml` |
| **Cloud storage** | ✅ GCS bucket for data | DVC remote (GCS) |
| **Docker builds** | ✅ CI builds on push | `.github/workflows/docker-build.yaml` |
| **Cloud training** | ✅ Docker + invoke commands | `invoke docker-train` |
| **FastAPI** | ✅ Prediction endpoint | `src/ct_scan_mlops/api.py` |
| **API deployment** | ✅ Docker image | `dockerfiles/api.dockerfile` |
| **API tests** | ✅ httpx test client | `tests/test_api.py` |
| **Load testing** | ✅ Locust | `pyproject.toml:L66` |

**Key Commands:**
```bash
invoke test                  # Run tests with coverage
invoke docker-build          # Build Docker images
invoke docker-train          # Train in container
uv run uvicorn ct_scan_mlops.api:app --reload  # Run API locally
```

**CI Workflows:**
- `tests.yaml`: Multi-OS testing (Ubuntu, Windows, macOS)
- `linting.yaml`: Ruff linting
- `docker-build.yaml`: Docker image builds
- `pre-commit.yaml`: Pre-commit hook validation

---

### 🟡 **Week 3: Production Readiness** (IN PROGRESS)

| Requirement | Status | Notes |
|------------|--------|-------|
| **Data drift** | 🔄 | Could add Evidently AI |
| **Robustness tests** | ✅ | Data augmentation (rotations/flips) in `data.py:L66-L82` |
| **System monitoring** | 🔄 | Could add Prometheus/Grafana |
| **Alerting** | ⚠️ | Basic CI notifications only |
| **Distributed data** | ✅ | PyTorch DataLoader (multi-worker) |
| **Training optimization** | ✅ | PyTorch Lightning + profiling |
| **Model optimization** | 🔄 | Could add ONNX/quantization |

**Implemented:**
- Data augmentation for robustness
- Multi-worker data loading
- PyTorch profiler for bottleneck detection

**Potential additions:**
- Evidently AI for data drift monitoring
- ONNX export for faster inference
- Model quantization for edge deployment

---

## 🛠️ Technology Stack

### Core ML Stack
- **Framework:** PyTorch 2.7 + PyTorch Lightning 2.1
- **Models:** Custom CNN, ResNet18 (transfer learning)
- **Augmentation:** Albumentations
- **Package Manager:** uv (modern pip replacement)

### MLOps Tools
- **Experiment Tracking:** Weights & Biases (team workspace)
- **Config Management:** Hydra
- **Data Versioning:** DVC + Google Cloud Storage
- **Profiling:** PyTorch Profiler + TensorBoard

### DevOps
- **CI/CD:** GitHub Actions (multi-OS testing)
- **Containerization:** Docker (CPU & CUDA images)
- **Code Quality:** Ruff (linting + formatting), pre-commit hooks
- **Testing:** pytest, pytest-cov, coverage
- **API:** FastAPI + Uvicorn
- **Load Testing:** Locust

---

## 🚀 Common Workflows

### 1️⃣ **First-time Setup**
```bash
# Clone repo
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops

# Install dependencies
uv sync --all-extras --dev

# Get data from GCS
invoke dvc-pull

# Verify setup
invoke test
```

### 2️⃣ **Development Workflow**
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes to code...

# Run quality checks
invoke ruff           # Format + lint
invoke test           # Run tests

# Commit & push
git add .
git commit -m "Your message"
git push -u origin feature/your-feature
```

### 3️⃣ **Training Workflow**
```bash
# Train with default CNN
invoke train

# Train with ResNet18
invoke train --args "model=resnet18"

# Train with custom epochs
invoke train --args "train.max_epochs=50"

# Train without W&B logging
invoke train --args "wandb.mode=disabled"

# Hyperparameter sweep (W&B)
invoke sweep                    # Create sweep
invoke sweep-agent --sweep-id ENTITY/PROJECT/SWEEP_ID
invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
```

### 4️⃣ **API Workflow**
```bash
# Run API locally
uv run uvicorn ct_scan_mlops.api:app --reload

# Test API
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.png"

# Run in Docker
invoke docker-build
docker run -p 8000:8000 ct_scan_mlops-api
```

---

## 🔍 Where to Find Things

### "Where is the model architecture?"
- **CNN:** `src/ct_scan_mlops/model.py:L12-L95` (class `ChestCTCNN`)
- **ResNet18:** `src/ct_scan_mlops/model.py:L98-L153` (class `ResNet18Transfer`)

### "Where is the training logic?"
- **Main script:** `src/ct_scan_mlops/train.py`
- **PyTorch Lightning module:** `src/ct_scan_mlops/train.py:L24-L119` (class `ChestCTModule`)
- **Training config:** `configs/train/default.yaml`

### "Where are the tests?"
- **All tests:** `tests/`
- **Data tests:** `tests/test_data.py`
- **Model tests:** `tests/test_model.py`
- **Training tests:** `tests/test_train.py`
- **API tests:** `tests/test_api.py`

### "Where is the CI/CD?"
- **All workflows:** `.github/workflows/`
- **Test workflow:** `.github/workflows/tests.yaml`
- **Linting workflow:** `.github/workflows/linting.yaml`
- **Docker workflow:** `.github/workflows/docker-build.yaml`

### "Where are the configs?"
- **Main config:** `configs/config.yaml`
- **Model configs:** `configs/model/` (cnn.yaml, resnet18.yaml)
- **Data config:** `configs/data/chest_ct.yaml`
- **Training config:** `configs/train/default.yaml`
- **Sweep config:** `configs/sweeps/train_sweep.yaml`

### "Where is the data?"
- **Raw data:** `data/raw/chest-ctscan-images/` (DVC tracked)
- **Processed data:** `data/processed/` (created by preprocessing)
- **DVC config:** `.dvc/config` (points to GCS)

### "Where are the dependencies?"
- **Main file:** `pyproject.toml`
- **Lock file:** `uv.lock`
- **Dev dependencies:** `pyproject.toml:L54-L67`

### "How do I run things?"
- **All commands:** `tasks.py` (run `invoke --list`)
- **Direct Python:** Use `uv run python -m ct_scan_mlops.train` (not just `python`)

---

## 🤝 Team Collaboration

### Weights & Biases
- **Team:** `mathiashl-danmarks-tekniske-universitet-dtu`
- **Project:** `CT_Scan_MLOps`
- **Dashboard:** https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps
- **Setup:** See `COLLABORATION.md`

### Git Workflow
1. Pull latest: `git pull && dvc pull`
2. Create branch: `git checkout -b feature/name`
3. Make changes + test: `invoke ruff && invoke test`
4. Push: `git push -u origin feature/name`
5. Create PR on GitHub

---

## 🆘 Quick Troubleshooting

### "My tests are failing"
```bash
invoke test          # Run tests locally
invoke ruff          # Fix formatting issues
```

### "I can't run training"
```bash
invoke dvc-pull      # Make sure you have the data
uv sync --dev        # Reinstall dependencies
invoke train --args "wandb.mode=disabled"  # Disable W&B if issue
```

### "My environment is broken"
```bash
rm -rf .venv uv.lock
uv sync --all-extras --dev
```

### "I need to add a package"
```bash
uv add package-name           # Add to dependencies
uv add --dev package-name     # Add to dev dependencies
```

### "Where can I ask questions?"
- **Code questions:** Ask the team or check `README.md`
- **MLOps questions:** Check [course materials](https://skaftenicki.github.io/dtu_mlops/)
- **W&B questions:** See `COLLABORATION.md`

---

## 📚 Additional Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Complete project documentation |
| `GetStarted.md` | Setup instructions |
| `COLLABORATION.md` | W&B team workflow |
| `CLAUDE.md` | Quick reference (AI assistant) |
| `DEPENDENCIES.md` | Detailed dependency info |
| `reports/README.md` | Project report (for grading) |

---

## 🎯 Key Takeaways

1. **Project is well-structured** according to MLOps best practices
2. **Most Week 1 & 2 requirements are completed**
3. **Use `invoke` commands** for all common tasks (`invoke --list`)
4. **Always use `uv run`** for Python commands
5. **CI/CD is fully automated** (tests, linting, Docker builds)
6. **Weights & Biases** tracks all experiments automatically
7. **DVC manages data** on Google Cloud Storage
8. **Everything is containerized** (training & API)

**Start here:**
- Read `README.md` for full context
- Run `invoke --list` to see available commands
- Check `.github/workflows/` to understand CI/CD
- Look at `src/ct_scan_mlops/` for the ML code
- Review `tests/` to understand test coverage

---

**Questions to guide your exploration:**
1. "What does the data pipeline look like?" → `src/ct_scan_mlops/data.py`
2. "How do we handle different models?" → `configs/model/`
3. "What CI checks run on PRs?" → `.github/workflows/`
4. "How do I reproduce an experiment?" → W&B dashboard + Hydra configs
5. "What's our test coverage?" → `invoke test` (shows coverage report)
