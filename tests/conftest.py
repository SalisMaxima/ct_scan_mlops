"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def data_available() -> bool:
    """Check if data is available for testing."""
    data_dir = Path("data/processed")
    return (data_dir / "train_images.pt").exists()


@pytest.fixture
def skip_if_no_data(data_available):
    """Skip test if data is not available (expected in CI)."""
    if not data_available:
        pytest.skip("Data files not available (expected in CI)")
