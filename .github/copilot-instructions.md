# CT Scan MLOps - Copilot Instructions

Chest CT scan multi-classification for lung tumor detection (4 classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal).

## IMPORTANT RULES
- **ALWAYS activate environment first: `source .venv/bin/activate`**
- **After code changes run: `invoke quality.ruff`**
- **After changing CLAUDE.md run: `invoke core.sync-ai-config` to sync copilot-instructions.md (do not edit copilot-instructions.md directly; CLAUDE.md is the source of truth)**
- **Always use `uv run` for Python commands** (e.g., `uv run python`, `uv run pytest`)
- **Always use `uv add` to install packages** (never `pip install`)

## Stack
- Python 3.12, PyTorch Lightning, Hydra configs
- uv for package management, invoke for tasks
- W&B for experiment tracking, DVC for data versioning

## Task Namespaces
Tasks are organized into namespaces. Use `invoke <namespace>.<task>` or `invoke --list` to see all.

**Key namespaces:**
- `core` - Environment setup (bootstrap, sync, setup-dev)
- `data` - Data management (download, preprocess, stats, validate)
- `train` - Training & sweeps (train, train-dual, sweep)
- `eval` - Evaluation (analyze, benchmark, profile)
- `quality` - Code quality (ruff, test, ci, security-check)
- `deploy` - Deployment (api, frontend, promote-model)
- `docker` - Docker ops (build, train, clean)
- `git` - Git ops (status, commit, branch)
- `dvc` - DVC ops (pull, push, add)
- `utils` - Utilities (clean-all, env-info, check-gpu)

## Essential Commands
```bash
# Setup (run once)
invoke core.setup-dev                # Complete dev environment setup

# Development workflow
invoke quality.ruff                  # Lint + format (run after code changes)
invoke quality.test                  # Run tests
invoke quality.ci                    # Run full CI pipeline locally

# Data & Training
invoke data.download                 # Download dataset
invoke data.preprocess               # Preprocess data
invoke data.stats                    # Show dataset statistics
invoke train.train                   # Train model
invoke dvc.pull                      # Get data from remote

# Sweeps (hyperparameter optimization)
invoke data.extract-features --args "features=top_features"
invoke data.prepare-sweep-features   # Extract all feature configs
invoke train.sweep                   # Create W&B sweep
invoke train.sweep-agent --sweep-id <SWEEP_ID>

# Analysis & Benchmarking
invoke eval.analyze "diagnose --checkpoint path/to/model.ckpt"
invoke eval.benchmark --checkpoint path/to/model.ckpt
invoke eval.profile --checkpoint path/to/model.ckpt
invoke data.validate                 # Validate data integrity

# Deployment
invoke deploy.api                    # Run API server
invoke deploy.frontend               # Run Streamlit frontend
invoke deploy.export-onnx --run-dir outputs/model

# Utilities
invoke utils.clean-all               # Clean all artifacts
invoke utils.check-gpu               # Check GPU availability
invoke utils.env-info                # Show environment info
```

## Key Paths
- `src/ct_scan_mlops/` - Main source code
- `configs/` - Hydra configs (model/, data/, train/)
- `tests/` - Unit tests
- `tasks.py` - All invoke commands

## Docs (on-demand)
- `docs/Structure.md` - **Complete repository structure index**
- `README.md` - Full project overview and all commands
- `docs/GetStarted.md` - Setup instructions
- `docs/COLLABORATION.md` - W&B team workflow
- `docs/` - Course materials and detailed guides
