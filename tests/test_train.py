"""Tests for training pipeline and LitModel."""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from ct_scan_mlops.train import LitModel


@pytest.fixture
def minimal_config():
    """Create a minimal config for testing LitModel."""
    return OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "name": "custom_cnn",
                "num_classes": 4,
                "input_channels": 3,
                "hidden_dims": [16, 32],
                "fc_hidden": 64,
                "dropout": 0.1,
                "batch_norm": True,
            },
            "data": {
                "image_size": 224,
            },
            "train": {
                "max_epochs": 2,
                "optimizer": {
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "eta_min": 0.00001,
                },
            },
        }
    )


@pytest.fixture
def dummy_batch():
    """Create a dummy batch of data."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 4, (4,))
    return images, labels


def test_litmodel_initialization(minimal_config):
    """Test that LitModel initializes correctly."""
    lit_model = LitModel(minimal_config)

    assert lit_model is not None
    assert lit_model.model is not None
    assert lit_model.criterion is not None
    assert isinstance(lit_model.training_history, dict)


def test_litmodel_forward_shape(minimal_config, dummy_batch):
    """Test that LitModel forward pass returns correct shape."""
    lit_model = LitModel(minimal_config)
    images, _ = dummy_batch

    output = lit_model(images)

    assert output.shape == (4, 4)  # (batch_size, num_classes)


def test_litmodel_training_step_returns_loss(minimal_config, dummy_batch):
    """Test that training_step returns a loss tensor."""
    lit_model = LitModel(minimal_config)

    # Mock the log method (Lightning calls this internally)
    lit_model.log = MagicMock()

    loss = lit_model.training_step(dummy_batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert loss.requires_grad  # Should be differentiable


def test_litmodel_validation_step_returns_loss(minimal_config, dummy_batch):
    """Test that validation_step returns a loss tensor."""
    lit_model = LitModel(minimal_config)

    # Mock the log method
    lit_model.log = MagicMock()

    loss = lit_model.validation_step(dummy_batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor


def test_litmodel_configure_optimizers(minimal_config):
    """Test that configure_optimizers returns correct structure."""
    lit_model = LitModel(minimal_config)

    opt_config = lit_model.configure_optimizers()

    assert "optimizer" in opt_config
    assert "lr_scheduler" in opt_config
    assert isinstance(opt_config["optimizer"], torch.optim.Adam)
    assert "scheduler" in opt_config["lr_scheduler"]


def test_litmodel_compute_accuracy(minimal_config):
    """Test the accuracy computation helper."""
    lit_model = LitModel(minimal_config)

    # Perfect predictions
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    targets = torch.tensor([0, 1])

    acc = lit_model._compute_accuracy(logits, targets)
    assert acc == 1.0

    # 50% accuracy
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    targets = torch.tensor([0, 2])  # Second one is wrong

    acc = lit_model._compute_accuracy(logits, targets)
    assert acc == 0.5


def test_litmodel_training_history_initialized(minimal_config):
    """Test that training history dict is properly initialized."""
    lit_model = LitModel(minimal_config)

    expected_keys = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    for key in expected_keys:
        assert key in lit_model.training_history
        assert isinstance(lit_model.training_history[key], list)
        assert len(lit_model.training_history[key]) == 0


def test_litmodel_log_called_in_training_step(minimal_config, dummy_batch):
    """Test that metrics are logged during training step."""
    lit_model = LitModel(minimal_config)
    lit_model.log = MagicMock()

    # Use batch_idx=1 to skip profiling code path (which runs on batch_idx=0, epoch=0)
    lit_model.training_step(dummy_batch, batch_idx=1)

    # Check that log was called for train_loss and train_acc
    assert lit_model.log.call_count >= 2


def test_litmodel_log_called_in_validation_step(minimal_config, dummy_batch):
    """Test that metrics are logged during validation step."""
    lit_model = LitModel(minimal_config)
    lit_model.log = MagicMock()

    lit_model.validation_step(dummy_batch, batch_idx=0)

    # Check that log was called for val_loss and val_acc
    assert lit_model.log.call_count >= 2


@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_litmodel_handles_various_batch_sizes(minimal_config, batch_size):
    """Test that LitModel works with various batch sizes."""
    lit_model = LitModel(minimal_config)
    lit_model.log = MagicMock()

    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 4, (batch_size,))
    batch = (images, labels)

    loss = lit_model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_litmodel_resnet_config():
    """Test LitModel with ResNet18 configuration."""
    config = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "name": "resnet18",
                "num_classes": 4,
                "pretrained": False,
                "freeze_backbone": False,
            },
            "train": {
                "max_epochs": 2,
                "optimizer": {
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "eta_min": 0.00001,
                },
            },
        }
    )

    lit_model = LitModel(config)
    lit_model.log = MagicMock()

    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 4, (2,))

    output = lit_model(images)
    assert output.shape == (2, 4)

    loss = lit_model.training_step((images, labels), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
