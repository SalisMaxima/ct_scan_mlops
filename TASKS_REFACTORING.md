# Tasks Refactoring Summary

## Overview
Refactored invoke tasks from a flat 48-task structure into organized namespaces for better maintainability and discoverability.

## Namespace Organization

### 📦 Core (`core`)
Environment setup and maintenance:
- `bootstrap` - Create UV virtual environment
- `sync` - Sync dependencies
- `dev` - Install dev dependencies
- **`setup-dev`** ⭐ NEW - One-command development environment setup
- `python` - Check Python version
- `sync-ai-config` - Sync CLAUDE.md to copilot-instructions.md

### 📊 Data (`data`)
Data management:
- `download` - Download CT scan dataset
- `preprocess` - Preprocess images
- `extract-features` - Extract radiomics features
- `prepare-sweep-features` - Extract all feature configs
- **`stats`** ⭐ NEW - Show dataset statistics
- **`validate`** ⭐ NEW - Validate data integrity

### 🎯 Train (`train`)
Training and hyperparameter tuning:
- `train` - Train model with W&B
- `train-dual` - Train dual pathway model
- `sweep` - Create W&B sweep
- `sweep-agent` - Run sweep agent
- `sweep-best` - Get best sweep run
- `sweep-report` - Generate sweep analysis

### 📈 Eval (`eval`)
Model evaluation and analysis:
- `analyze` - Run analysis CLI (diagnose/explain/compare)
- **`benchmark`** ⭐ NEW - Measure inference speed and throughput
- **`profile`** ⭐ NEW - Profile with cProfile
- **`model-info`** ⭐ NEW - Show model size/params/architecture

### ✅ Quality (`quality`)
Code quality and testing:
- `ruff` - Lint and format
- `test` - Run tests with coverage
- `test-unit` - Fast unit tests only
- `test-all` - All tests including slow ones
- `test-watch` - Watch mode for tests
- **`ci`** ⭐ NEW - Run full CI pipeline locally
- **`security-check`** ⭐ NEW - Run security scans (pip-audit + bandit)
- **`install-hooks`** ⭐ NEW - Install git hooks
- **`deps-outdated`** ⭐ NEW - Check outdated packages
- **`deps-tree`** ⭐ NEW - Show dependency tree

### 🚀 Deploy (`deploy`)
Deployment and serving:
- `promote-model` - Promote to W&B registry
- `export-onnx` - Export to ONNX
- `api` - Run FastAPI server
- `frontend` - Run Streamlit frontend

### 🐳 Docker (`docker`)
Docker operations:
- `build` - Build CPU images
- `build-cuda` - Build CUDA image
- `train` - Train in container
- `api` - Run API in container
- `api-frontend` - Full stack in containers
- `clean` - Clean Docker artifacts

### 📡 Monitor (`monitor`)
Model monitoring:
- `extract-stats` - Extract reference stats
- `check-drift` - Check for data drift

### 📝 Git (`git`)
Git operations:
- `status` - Show status
- `commit` - Commit and push
- `branch` - Create branch and push

### 💾 DVC (`dvc`)
Data versioning:
- `pull` - Pull data
- `push` - Push data
- `add` - Add and push data

### 📚 Docs (`docs`)
Documentation:
- `build` - Build docs
- `serve` - Serve docs

### 🛠️ Utils (`utils`)
Utilities:
- **`clean-all`** ⭐ NEW - Clean all artifacts (pyc, build, test)
- **`clean-pyc`** ⭐ NEW - Clean bytecode
- **`clean-build`** ⭐ NEW - Clean build artifacts
- **`clean-test`** ⭐ NEW - Clean test artifacts
- **`clean-outputs`** ⭐ NEW - Clean training outputs
- **`env-info`** ⭐ NEW - Show environment details
- **`env-export`** ⭐ NEW - Export environment
- **`check-gpu`** ⭐ NEW - Check GPU availability
- **`count-loc`** ⭐ NEW - Count lines of code
- **`find-todos`** ⭐ NEW - Find TODO comments
- **`port-check`** ⭐ NEW - Check port usage
- **`kill-port`** ⭐ NEW - Kill process on port

## High-Priority Tasks Implemented ✅

1. ✅ **`invoke quality.ci`** - Run full CI pipeline locally (ruff + tests)
2. ✅ **`invoke eval.benchmark`** - Measure inference speed
3. ✅ **`invoke data.stats`** - Show dataset statistics
4. ✅ **`invoke core.setup-dev`** - One-command dev environment setup
5. ✅ **`invoke utils.clean-all`** - Clean all build/cache artifacts
6. ✅ **`invoke eval.profile`** - Profile model/training performance
7. ✅ **`invoke quality.security-check`** - Run security scans

## New Python Modules Created

1. `src/ct_scan_mlops/stats.py` - Dataset statistics display
2. `src/ct_scan_mlops/validate.py` - Data integrity validation
3. `src/ct_scan_mlops/benchmark.py` - Inference benchmarking
4. `src/ct_scan_mlops/model_info.py` - Model information display

## Task Files Structure

```
tasks/
├── __init__.py          # Namespace collection
├── core.py              # Environment setup
├── data.py              # Data management
├── train.py             # Training & sweeps
├── eval.py              # Evaluation
├── quality.py           # Code quality
├── deploy.py            # Deployment
├── docker.py            # Docker ops
├── monitor.py           # Monitoring
├── git_tasks.py         # Git ops
├── dvc_tasks.py         # DVC ops
├── docs.py              # Documentation
└── utils.py             # Utilities

tasks.py                 # Main entry point (imports namespace)
tasks_old.py             # Backup of original flat structure
```

## Usage Examples

```bash
# Development workflow
invoke core.setup-dev                    # One-time setup
invoke quality.ci                        # Pre-commit checks
invoke quality.security-check            # Security audit

# Data pipeline
invoke data.download
invoke data.stats                        # Check dataset
invoke data.validate                     # Validate integrity

# Training & evaluation
invoke train.train --args "model=resnet18"
invoke eval.benchmark --checkpoint path/to/model.ckpt
invoke eval.profile --checkpoint path/to/model.ckpt

# Utilities
invoke utils.clean-all                   # Clean everything
invoke utils.env-info                    # Environment details
invoke utils.check-gpu                   # GPU status
```

## Dependencies Added

- `pip-audit` - Security vulnerability scanning
- `bandit` - Python security linter

## CLAUDE.md Updates

Updated to reflect new namespace organization and commands.

## Benefits

1. **Better Organization** - 48 tasks organized into 12 logical namespaces
2. **Easier Discovery** - `invoke --list` shows organized structure
3. **Maintainability** - Each namespace in separate file
4. **Scalability** - Easy to add new tasks to appropriate namespace
5. **Best Practices** - Follows invoke namespace patterns from real-world projects
6. **New Capabilities** - 20+ new utility and quality tasks added

## Backward Compatibility

All original tasks preserved in `tasks_old.py`. New namespace structure is additive.
