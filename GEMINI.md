# CT Scan MLOps

## Project Overview
Chest CT scan multi-classification model for lung tumor detection. The goal is to detect four classes:
1. Adenocarcinoma
2. Large cell carcinoma
3. Squamous cell carcinoma
4. Normal

The project compares a custom CNN baseline against a ResNet18 model (transfer learning) and utilizes MLOps best practices including experiment tracking (W&B), data versioning (DVC), and configuration management (Hydra).

## Technologies
*   **Language:** Python 3.12
*   **Frameworks:** PyTorch, PyTorch Lightning
*   **Configuration:** Hydra
*   **Experiment Tracking:** Weights & Biases (W&B)
*   **Data Versioning:** DVC
*   **Package Management:** uv
*   **Task Runner:** invoke
*   **Containerization:** Docker
*   **API/Frontend:** FastAPI, Streamlit

## Building and Running

### Environment Setup
The project uses `uv` for dependency management and `invoke` for task running.

```bash
# Bootstrap environment (create venv and sync deps)
invoke bootstrap

# Activate environment
source .venv/bin/activate
```

### Key Commands (via `invoke`)

All commands should be run within the virtual environment or prefixed with `uv run`.

**Development**
*   `invoke sync`: Sync dependencies.
*   `invoke dev`: Install with dev dependencies.
*   `invoke ruff`: Run linter and formatter (auto-fix).
*   `invoke test`: Run tests with coverage.

**Data & Training**
*   `invoke dvc-pull`: Download data from remote storage.
*   `invoke preprocess-data`: Preprocess images (run once).
*   `invoke train`: Train with default CNN.
*   `invoke train --args "model=resnet18"`: Train ResNet18.
*   `invoke train --args "train.max_epochs=20"`: Custom training parameters.

**Analysis**
*   `invoke evaluate --checkpoint <path>`: Evaluate a model checkpoint.
*   `invoke compare-baselines --baseline <path> --improved <path>`: Compare two models.
*   `invoke analyze-features --checkpoint <path>`: Analyze radiomics feature importance.
*   `invoke analyze-errors --checkpoint <path>`: Analyze misclassified samples.

**Docker**
*   `invoke docker-build`: Build CPU Docker images.
*   `invoke docker-build-cuda`: Build GPU Docker image.
*   `invoke docker-train`: Run training in a Docker container.
*   `invoke docker-api`: Run the inference API in Docker.

## Project Structure

*   `src/ct_scan_mlops/`: Main source code.
    *   `data.py`: Data loading and preprocessing.
    *   `model.py`: Model architectures (CNN, ResNet18).
    *   `train.py`: Training script.
    *   `evaluate.py`: Evaluation script.
    *   `api.py`: FastAPI inference application.
    *   `sweep_train.py`: Entrypoint for W&B sweeps.
*   `configs/`: Hydra configuration files (`config.yaml`, `model/`, `data/`, `train/`).
*   `tests/`: Unit tests.
*   `tasks.py`: Definitions for all `invoke` commands.
*   `dockerfiles/`: Dockerfiles for training and API.
*   `data/`: Data directory (tracked by DVC).
*   `reports/`: Generated analysis reports.
*   `docs/`: Documentation.

## Development Conventions
*   **Dependency Management:** Always use `uv add <package>` to install new dependencies. Never use `pip install` directly.
*   **Code Style:** Run `invoke ruff` before committing to ensure formatting and linting compliance.
*   **Testing:** Run `invoke test` to verify changes.
*   **Configuration:** Use Hydra configs in `configs/` for parameters. Override them via command line args in `invoke train`.
*   **Experiment Tracking:** All training runs log to W&B by default. Use `wandb.mode=disabled` to disable.
