# CT Scan MLOps - Copilot Instructions

Chest CT scan multi-classification for lung tumor detection (4 classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal).

## IMPORTANT RULES

1. **ALWAYS activate environment first**: `source .venv/bin/activate`
2. **After code changes run**: `invoke ruff`
3. **Always use `uv run` for Python commands** (e.g., `uv run python`, `uv run pytest`)
4. **Always use `uv add` to install packages** (never `pip install`)

## Stack

- Python 3.12, PyTorch Lightning, Hydra configs
- uv for package management, invoke for tasks
- W&B for experiment tracking, DVC for data versioning

## Essential Commands

```bash
source .venv/bin/activate  # ALWAYS run first
invoke ruff                # Lint + format (run after code changes)
invoke test                # Run tests
invoke train               # Train model
invoke dvc-pull            # Get data from remote
```

## Key Paths

- `src/ct_scan_mlops/` - Main source code
- `configs/` - Hydra configs (model/, data/, train/)
- `tests/` - Unit tests
- `tasks.py` - All invoke commands

## Code Style

- Use ruff for linting and formatting
- Follow existing patterns in the codebase
- Use type hints for function signatures
- Use PyTorch Lightning for model code

## Documentation

- `docs/Structure.md` - Complete repository structure index
- `README.md` - Full project overview and all commands
- `GetStarted.md` - Setup instructions
- `COLLABORATION.md` - W&B team workflow
- `docs/` - Course materials and detailed guides
