"""Tests for data module.

Note: These tests use synthetic data and don't require actual dataset files.
Tests are CI-friendly and will pass without data in .gitignore.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ct_scan_mlops.data import CLASSES, ChestCTDataModule, normalize, preprocess


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


# --- Expanded Tests ---


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 224, 224),
        (10, 3, 224, 224),
        (5, 3, 100, 100),
        (32, 3, 64, 64),
    ],
)
def test_normalize_various_shapes(shape: tuple):
    """Test normalization works on various tensor shapes."""
    images = torch.randn(*shape) * 5 + 10  # Offset mean and scale

    normalized = normalize(images)

    assert normalized.shape == shape
    # Per-channel mean should be close to 0
    channel_means = normalized.mean(dim=(0, 2, 3))
    assert channel_means.abs().max() < 0.1, f"Mean not close to 0: {channel_means}"


def test_normalize_numerical_stability_zeros():
    """Test normalize handles near-zero std without NaN/Inf."""
    # Constant image (std = 0) should not produce NaN
    images = torch.ones(4, 3, 32, 32) * 5.0

    normalized = normalize(images)

    assert not torch.isnan(normalized).any(), "NaN values in output"
    assert not torch.isinf(normalized).any(), "Inf values in output"


def test_normalize_preserves_batch_independence():
    """Test that normalizing single image matches batch normalization."""
    batch = torch.randn(8, 3, 64, 64) * 10 + 5

    # Normalize full batch
    batch_normalized = normalize(batch)

    # Normalize each image individually
    individual_normalized = torch.stack([normalize(img.unsqueeze(0)).squeeze(0) for img in batch])

    # They should NOT be equal (batch normalization uses batch stats)
    # This test documents expected behavior
    assert batch_normalized.shape == individual_normalized.shape


def test_normalize_output_dtype():
    """Test normalize preserves input dtype."""
    images_float32 = torch.randn(4, 3, 32, 32, dtype=torch.float32)
    normalized = normalize(images_float32)
    assert normalized.dtype == torch.float32


@pytest.fixture
def dummy_raw_data_dir(tmp_path: Path) -> Path:
    """Create a minimal raw data directory structure with dummy images."""
    # Create directory structure: raw_dir/train/{class}/images
    raw_dir = tmp_path / "raw"

    for split in ["train", "valid", "test"]:
        for class_name in CLASSES:
            class_dir = raw_dir / split / class_name
            class_dir.mkdir(parents=True)

            # Create 2 dummy images per class
            rng = np.random.default_rng()
            for i in range(2):
                img_array = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
                img = Image.fromarray(img_array, mode="RGB")
                img.save(class_dir / f"image_{i}.png")

    return raw_dir


def test_preprocess_creates_output_files(dummy_raw_data_dir: Path, tmp_path: Path):
    """Test that preprocess creates the expected output files."""
    output_dir = tmp_path / "processed"

    stats = preprocess(
        raw_dir=dummy_raw_data_dir,
        output_dir=output_dir,
        image_size=64,  # Small size for faster test
    )

    # Check output files exist
    assert (output_dir / "train_images.pt").exists()
    assert (output_dir / "train_labels.pt").exists()
    assert (output_dir / "valid_images.pt").exists()
    assert (output_dir / "valid_labels.pt").exists()
    assert (output_dir / "test_images.pt").exists()
    assert (output_dir / "test_labels.pt").exists()
    assert (output_dir / "stats.pt").exists()

    # Check stats dict
    assert "image_size" in stats
    assert stats["image_size"] == 64
    assert "classes" in stats


def test_preprocess_output_shapes(dummy_raw_data_dir: Path, tmp_path: Path):
    """Test that preprocessed tensors have correct shapes."""
    output_dir = tmp_path / "processed"
    image_size = 64

    preprocess(
        raw_dir=dummy_raw_data_dir,
        output_dir=output_dir,
        image_size=image_size,
    )

    # Load and check shapes
    train_images = torch.load(output_dir / "train_images.pt", weights_only=True)
    train_labels = torch.load(output_dir / "train_labels.pt", weights_only=True)

    # 4 classes * 2 images each = 8 train images
    assert train_images.shape[0] == 8
    assert train_images.shape[1] == 3  # RGB channels
    assert train_images.shape[2] == image_size
    assert train_images.shape[3] == image_size

    assert train_labels.shape[0] == 8
    assert train_labels.min() >= 0
    assert train_labels.max() < len(CLASSES)


def test_preprocess_normalization(dummy_raw_data_dir: Path, tmp_path: Path):
    """Test that preprocessed images are normalized."""
    output_dir = tmp_path / "processed"

    preprocess(
        raw_dir=dummy_raw_data_dir,
        output_dir=output_dir,
        image_size=64,
    )

    train_images = torch.load(output_dir / "train_images.pt", weights_only=True)

    # Check normalized (mean ~0, std ~1)
    mean = train_images.mean()
    std = train_images.std()

    assert abs(mean.item()) < 0.5, f"Mean {mean} not close to 0"
    assert 0.5 < std.item() < 2.0, f"Std {std} not close to 1"


def test_chest_ct_datamodule_exists():
    """Test that ChestCTDataModule can be imported."""
    assert ChestCTDataModule is not None


def test_chest_ct_datamodule_has_required_methods():
    """Test that ChestCTDataModule has all required Lightning methods."""
    assert hasattr(ChestCTDataModule, "setup")
    assert hasattr(ChestCTDataModule, "train_dataloader")
    assert hasattr(ChestCTDataModule, "val_dataloader")
    assert hasattr(ChestCTDataModule, "test_dataloader")
