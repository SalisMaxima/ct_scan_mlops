"""Tests for model evaluation pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ct_scan_mlops.data import CLASSES
from ct_scan_mlops.evaluate import evaluate_model
from ct_scan_mlops.model import CustomCNN


@pytest.fixture
def dummy_model() -> torch.nn.Module:
    """Create a CustomCNN model with random weights."""
    return CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[16, 32],
        fc_hidden=64,
        dropout=0.0,
        image_size=224,
    )


@pytest.fixture
def dummy_test_loader():
    """Create a mock DataLoader that yields batches of dummy data."""
    # Create a small dataset: 12 samples
    images = torch.randn(12, 3, 224, 224)
    # Ensure all 4 classes are represented
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_evaluate_model_returns_metrics(dummy_model: torch.nn.Module, dummy_test_loader):
    """Test that evaluate_model returns a dictionary with required metrics."""
    device = torch.device("cpu")

    metrics = evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=False,
    )

    assert isinstance(metrics, dict)
    assert "test_accuracy" in metrics
    assert isinstance(metrics["test_accuracy"], float)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


def test_evaluate_model_returns_per_class_metrics(dummy_model: torch.nn.Module, dummy_test_loader):
    """Test that evaluate_model returns per-class precision, recall, f1."""
    device = torch.device("cpu")

    metrics = evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=False,
    )

    # Check per-class metrics for each class
    for class_name in CLASSES:
        assert f"test_{class_name}_precision" in metrics
        assert f"test_{class_name}_recall" in metrics
        assert f"test_{class_name}_f1" in metrics

        # Values should be between 0 and 1
        assert 0.0 <= metrics[f"test_{class_name}_precision"] <= 1.0
        assert 0.0 <= metrics[f"test_{class_name}_recall"] <= 1.0
        assert 0.0 <= metrics[f"test_{class_name}_f1"] <= 1.0


def test_evaluate_model_returns_aggregate_metrics(dummy_model: torch.nn.Module, dummy_test_loader):
    """Test that evaluate_model returns macro and weighted average F1."""
    device = torch.device("cpu")

    metrics = evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=False,
    )

    assert "test_macro_avg_f1" in metrics
    assert "test_weighted_avg_f1" in metrics

    assert 0.0 <= metrics["test_macro_avg_f1"] <= 1.0
    assert 0.0 <= metrics["test_weighted_avg_f1"] <= 1.0


@patch("ct_scan_mlops.evaluate.wandb")
def test_evaluate_model_logs_to_wandb(mock_wandb, dummy_model: torch.nn.Module, dummy_test_loader):
    """Test that metrics are logged to wandb when log_to_wandb=True."""
    device = torch.device("cpu")

    # Setup mock
    mock_wandb.run = MagicMock()
    mock_wandb.run.summary = {}

    evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=True,
        save_confusion_matrix=False,
    )

    # Verify wandb.log was called with metrics
    mock_wandb.log.assert_called_once()
    logged_metrics = mock_wandb.log.call_args[0][0]
    assert "test_accuracy" in logged_metrics


def test_evaluate_model_saves_confusion_matrix(dummy_model: torch.nn.Module, dummy_test_loader, tmp_path: Path):
    """Test that confusion matrix is saved when requested."""
    device = torch.device("cpu")

    evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=True,
        output_dir=tmp_path,
    )

    # Check confusion matrix file was created
    cm_path = tmp_path / "confusion_matrix.png"
    assert cm_path.exists(), "Confusion matrix image should be saved"


def test_evaluate_model_sets_eval_mode(dummy_model: torch.nn.Module, dummy_test_loader):
    """Test that model is set to eval mode during evaluation."""
    device = torch.device("cpu")

    # Set to train mode first
    dummy_model.train()
    assert dummy_model.training

    evaluate_model(
        model=dummy_model,
        test_loader=dummy_test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=False,
    )

    # Model should be in eval mode after evaluation
    assert not dummy_model.training


def test_evaluate_model_handles_single_batch():
    """Test evaluation works with a single batch."""
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[16, 32],
        fc_hidden=64,
        image_size=224,
    )

    # Single batch of 4 samples
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    metrics = evaluate_model(
        model=model,
        test_loader=loader,
        device=torch.device("cpu"),
        log_to_wandb=False,
        save_confusion_matrix=False,
    )

    assert "test_accuracy" in metrics
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
