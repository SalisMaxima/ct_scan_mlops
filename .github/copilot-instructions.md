# CT Scan MLOps - Copilot Instructions

Chest CT scan multi-classification for lung tumor detection (4 classes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal).

## IMPORTANT RULES
- **ALWAYS activate environment first: `source .venv/bin/activate`**
- **After code changes run: `invoke ruff`**
- **After changing CLAUDE.md run: `invoke sync-ai-config` to sync copilot-instructions.md (do not edit copilot-instructions.md directly; CLAUDE.md is the source of truth)**
- **Always use `uv run` for Python commands** (e.g., `uv run python`, `uv run pytest`)
- **Always use `uv add` to install packages** (never `pip install`)

## Stack
- Python 3.12, PyTorch Lightning, Hydra configs
- uv for package management, invoke for tasks
- W&B for experiment tracking, DVC for data versioning

## Essential Commands
```bash
invoke ruff          # Lint + format (run after code changes)
invoke test          # Run tests
invoke train         # Train model
invoke dvc-pull      # Get data from remote
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
