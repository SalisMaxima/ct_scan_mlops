"""Tests for data module.

Note: These tests use synthetic data and don't require actual dataset files.
Tests are CI-friendly and will pass without data in .gitignore.
"""

import torch

from ct_scan_mlops.data import CLASSES, normalize


def test_normalize():
    """Test normalize function."""
    # Create dummy data
    images = torch.randn(10, 3, 224, 224) * 10 + 5  # Random data with non-zero mean

    # Normalize
    normalized = normalize(images)

    # Check mean is close to 0 and std close to 1
    assert normalized.mean(dim=(0, 2, 3)).abs().max() < 0.1
    assert (normalized.std(dim=(0, 2, 3)) - 1.0).abs().max() < 0.1


def test_classes_defined():
    """Test that CLASSES are properly defined."""
    assert len(CLASSES) == 4
    assert "adenocarcinoma" in CLASSES
    assert "large_cell_carcinoma" in CLASSES
    assert "squamous_cell_carcinoma" in CLASSES
    assert "normal" in CLASSES


def test_processed_dataset_exists():
    """Test that ProcessedChestCTDataset can be imported."""
    from ct_scan_mlops.data import ProcessedChestCTDataset

    assert ProcessedChestCTDataset is not None


def test_chest_ct_dataset_exists():
    """Test that ChestCTDataset can be imported."""
    from ct_scan_mlops.data import ChestCTDataset

    assert ChestCTDataset is not None


# Example of a test that requires actual data (automatically skipped in CI)
# Uncomment when you want to test with real data locally
#
# import pytest
#
# @pytest.mark.requires_data
# def test_load_processed_dataset(skip_if_no_data):
#     """Test loading actual processed dataset (requires data files)."""
#     from ct_scan_mlops.data import ProcessedChestCTDataset
#
#     dataset = ProcessedChestCTDataset("data/processed", split="train")
#     assert len(dataset) > 0
#     img, label = dataset[0]
#     assert img.shape == (3, 224, 224)
#     assert 0 <= label < 4
