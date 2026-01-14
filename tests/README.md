# Tests

## Overview

This directory contains unit tests for the ct_scan_mlops project. Tests are designed to be **CI-friendly** and run without requiring actual dataset files.

## Test Structure

```
tests/
├── conftest.py          # Pytest configuration and fixtures
├── test_data.py         # Data module tests
├── test_model.py        # Model architecture tests
└── test_api.py          # API tests (placeholder)
```

## Running Tests

### Locally (basic)
```bash
pytest tests/
```

### With coverage
```bash
uv run coverage run -m pytest tests/
uv run coverage report -m
```

### Using invoke
```bash
invoke test
```

## Test Philosophy

**All tests use synthetic data** (e.g., `torch.randn`) and don't require actual dataset files from `data/raw/` or `data/processed/`. This ensures:

- ✅ Tests pass in CI without downloading large datasets
- ✅ Fast test execution
- ✅ No DVC dependency for testing
- ✅ Tests work on fresh clones

## Test Markers

Tests can be marked with custom markers for organization:

### `@pytest.mark.requires_data`
Marks tests that require actual data files. These tests will be automatically skipped in CI when data is not available.

**Example:**
```python
import pytest

@pytest.mark.requires_data
def test_load_real_dataset(skip_if_no_data):
    """Test loading actual processed dataset."""
    from ct_scan_mlops.data import ProcessedChestCTDataset

    dataset = ProcessedChestCTDataset("data/processed", split="train")
    assert len(dataset) > 0
```

### `@pytest.mark.slow`
Marks slow-running tests (e.g., training loops). Can be skipped with `pytest -m "not slow"`.

## Fixtures

### `data_available`
Boolean fixture that checks if data files exist.

### `skip_if_no_data`
Automatically skips test if data files are not available (expected in CI).

## Adding New Tests

When adding new tests:

1. **Use synthetic data when possible** - prefer `torch.randn()` over loading real files
2. **Mark data-dependent tests** - use `@pytest.mark.requires_data` if you need actual data
3. **Document clearly** - explain what the test validates
4. **Keep tests fast** - avoid expensive operations in unit tests

## CI Behavior

In GitHub Actions:
- Data files (`data/raw/`, `data/processed/`) are **not available** (.gitignore)
- Tests marked with `@pytest.mark.requires_data` are **automatically skipped**
- Coverage report generated with `-i` flag to ignore missing source warnings

## Coverage

Current coverage focuses on:
- Model initialization and forward passes
- Data utility functions
- Import checks and basic functionality

Full coverage requires integration tests with actual data, which should be run locally.
