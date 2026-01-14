"""Tests for model module.

Note: These tests use synthetic data (torch.randn) and don't require actual dataset files.
Tests are CI-friendly and will pass without data in .gitignore.
"""

import torch

from ct_scan_mlops.model import CustomCNN, ResNet18


def test_custom_cnn_initialization():
    """Test CustomCNN can be initialized."""
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[32, 64, 128, 256],
        fc_hidden=512,
        dropout=0.3,
        batch_norm=True,
        image_size=224,
    )
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_custom_cnn_forward():
    """Test CustomCNN forward pass."""
    model = CustomCNN(num_classes=4, image_size=224)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 4)


def test_resnet18_initialization():
    """Test ResNet18 can be initialized."""
    model = ResNet18(num_classes=4, pretrained=False)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_resnet18_forward():
    """Test ResNet18 forward pass."""
    model = ResNet18(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 4)


def test_resnet18_freeze_unfreeze():
    """Test ResNet18 freeze and unfreeze methods."""
    model = ResNet18(num_classes=4, pretrained=False, freeze_backbone=True)

    # Check backbone is frozen
    for name, param in model.backbone.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad

    # Unfreeze and check
    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad
