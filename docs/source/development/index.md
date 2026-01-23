# Development

Guides for contributing to and understanding the CT Scan MLOps project.

## Overview

| Guide | Description |
|-------|-------------|
| [Collaboration](collaboration.md) | W&B team workflow and experiment tracking |
| [Project Structure](structure.md) | Complete repository structure index |

## Contributing

### Workflow

1. **Pull latest changes:**

   ```bash
   git pull
   dvc pull  # If data changed
   ```

2. **Create a branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and test:**

   ```bash
   invoke ruff    # Format code
   invoke test    # Run tests
   ```

4. **Commit and push:**

   ```bash
   git add .
   git commit -m "Add your feature"
   git push -u origin feature/your-feature-name
   ```

5. **Create Pull Request** on GitHub

## Code Quality

### Linting and Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Run linter + formatter
invoke ruff

# Auto-fix issues
invoke lint --fix
```

### Testing

Tests are written with pytest:

```bash
# Run all tests
invoke test

# Run with coverage
pytest --cov=ct_scan_mlops tests/
```

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML Framework | PyTorch + PyTorch Lightning |
| Config | Hydra |
| Experiment Tracking | Weights & Biases |
| Data Versioning | DVC |
| Package Manager | uv |
| Task Runner | invoke |
| Linting | Ruff |
| Testing | pytest |
| API | FastAPI |
| Containerization | Docker |
