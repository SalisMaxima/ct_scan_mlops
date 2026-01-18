# Project Dependencies

This document describes all dependencies used in the CT Scan MLOps project. Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

## Python Version

- **Python >= 3.12** required

## Installation

```bash
# Install all dependencies (including dev)
uv sync --all-groups

# Install production dependencies only
uv sync
```

---

## Production Dependencies

### Deep Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.7.0 | PyTorch deep learning framework |
| `torchvision` | >= 0.16.0 | Image datasets, transforms, and pretrained models |
| `pytorch-lightning` | >= 2.1.0 | Lightning wrapper for cleaner training code |
| `torchmetrics` | >= 1.0.0 | Metrics computation (accuracy, F1, etc.) |
| `timm` | >= 0.9.0 | Pretrained models library (ResNet18, etc.) |
| `torch-tb-profiler` | >= 0.4.3 | TensorBoard profiler for PyTorch |

**Platform-specific PyTorch sources:**
- **Linux/Windows**: CUDA 12.8 (GPU acceleration)
- **macOS**: CPU-only (no CUDA support)

### Image Processing & Augmentation

| Package | Version | Purpose |
|---------|---------|---------|
| `albumentations` | >= 1.3.0 | Image augmentation for training |
| `pillow` | >= 10.0.0 | Image loading and manipulation |
| `opencv-python-headless` | >= 4.8.0 | Computer vision operations (headless for servers) |

### MLOps Tools

| Package | Version | Purpose |
|---------|---------|---------|
| `wandb` | >= 0.24.0 | Experiment tracking and model registry |
| `hydra-core` | >= 1.3.0 | Configuration management |
| `omegaconf` | >= 2.3.0 | YAML config with variable interpolation |
| `dvc` | >= 3.0.0 | Data version control |
| `dvc-gs` | >= 3.0.0 | DVC Google Cloud Storage support |
| `loguru` | >= 0.7.3 | Application logging |

### API & CLI

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.115.0 | REST API framework for inference |
| `uvicorn` | >= 0.30.0 | ASGI server for FastAPI |
| `typer` | >= 0.9.0 | CLI interface |
| `python-multipart` | >= 0.0.21 | Form data parsing for file uploads |

### Data Science

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24.0 | Numerical computing |
| `pandas` | >= 2.0.0 | Data manipulation and analysis |
| `scikit-learn` | >= 1.3.0 | Machine learning utilities |
| `matplotlib` | >= 3.8.0 | Plotting and visualization |
| `seaborn` | >= 0.13.0 | Statistical data visualization |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| `rich` | >= 13.0.0 | Rich progress bars and console output |
| `kagglehub` | >= 0.4.0 | Kaggle dataset access |

---

## Development Dependencies

These are only needed for development and testing, not for production deployment.

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 8.0.0 | Testing framework |
| `pytest-cov` | >= 4.1.0 | Test coverage reporting |
| `coverage` | >= 7.6.0 | Code coverage measurement |
| `ruff` | >= 0.8.0 | Linting and formatting |
| `pre-commit` | >= 4.0.0 | Git pre-commit hooks |
| `invoke` | >= 2.2.0 | Task automation |
| `httpx` | >= 0.27.0 | HTTP client for API testing |
| `locust` | >= 2.20.0 | Load testing |

### Documentation

| Package | Version | Purpose |
|---------|---------|---------|
| `mkdocs` | >= 1.6.0 | Documentation generator |
| `mkdocs-material` | >= 9.4.0 | Material theme for MkDocs |
| `mkdocstrings-python` | >= 1.12.0 | Python docstring extraction |

---

## Dependency Management

### Adding Dependencies

```bash
# Add a production dependency
uv add package-name

# Add a dev dependency
uv add --group dev package-name

# Add with version constraint
uv add "package-name>=1.0.0"
```

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package package-name

# Sync environment after updating
uv sync --all-groups
```

### Viewing Dependencies

```bash
# Show dependency tree
uv tree

# Show outdated packages
uv pip list --outdated
```

---

## Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependency specifications with version constraints |
| `uv.lock` | Locked versions for reproducible installs |

---

## Notes

- All dependencies are managed with [uv](https://docs.astral.sh/uv/)
- The lock file ensures reproducible builds across all environments
- Platform-specific PyTorch installation is handled automatically via `[tool.uv.sources]`
