"""Tests for model module.

Note: These tests use synthetic data (torch.randn) and don't require actual dataset files.
Tests are CI-friendly and will pass without data in .gitignore.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ct_scan_mlops.model import CustomCNN, DualPathwayModel, ResNet18, build_model


class AttrDict(dict):
    """Dict with attribute access and a .get method (like OmegaConf nodes)."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def make_cfg(**kwargs) -> AttrDict:
    """Convenience helper to build nested AttrDict configs."""
    return AttrDict(kwargs)


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
    with torch.no_grad():
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
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 4)


def test_resnet18_freeze_unfreeze():
    """Test ResNet18 freeze and unfreeze methods."""
    model = ResNet18(num_classes=4, pretrained=False, freeze_backbone=True)

    # Check backbone is frozen (except fc)
    for name, param in model.backbone.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad

    # Unfreeze and check
    model.unfreeze_backbone()
    for param in model.backbone.parameters():
        assert param.requires_grad


def test_custom_cnn_layer_count_with_batch_norm():
    """CustomCNN should build (Conv + BN + ReLU + Pool) per hidden dim when batch_norm=True."""
    hidden_dims = [8, 16, 32]
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=hidden_dims,
        batch_norm=True,
        image_size=224,
    )
    # Each block: Conv2d, BatchNorm2d, ReLU, MaxPool2d => 4 modules
    assert len(model.features) == 4 * len(hidden_dims)


def test_custom_cnn_layer_count_without_batch_norm():
    """CustomCNN should build (Conv + ReLU + Pool) per hidden dim when batch_norm=False."""
    hidden_dims = [8, 16, 32]
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=hidden_dims,
        batch_norm=False,
        image_size=224,
    )
    # Each block: Conv2d, ReLU, MaxPool2d => 3 modules
    assert len(model.features) == 3 * len(hidden_dims)


@pytest.mark.parametrize("image_size,hidden_dims", [(224, [16, 32, 64, 128]), (128, [8, 16, 32])])
def test_custom_cnn_forward_various_image_sizes(image_size: int, hidden_dims: list[int]):
    """Forward should work as long as image_size is compatible with pooling depth."""
    model = CustomCNN(
        num_classes=5,
        in_channels=3,
        hidden_dims=hidden_dims,
        image_size=image_size,
    )
    x = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 5)


def test_build_model_customcnn():
    """build_model should return CustomCNN configured from cfg."""
    cfg = make_cfg(
        model=AttrDict(
            name="custom_cnn",
            num_classes=4,
            input_channels=3,
            hidden_dims=[8, 16, 32],
            fc_hidden=64,
            dropout=0.1,
            batch_norm=True,
        ),
        data=AttrDict(image_size=128),
    )

    model = build_model(cfg)
    assert isinstance(model, CustomCNN)

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4)


def test_build_model_resnet18_pretrained_false():
    """build_model should build ResNet18 without downloading weights when pretrained=False."""
    cfg = make_cfg(
        model=AttrDict(
            name="resnet18",
            num_classes=4,
            pretrained=False,  # IMPORTANT for CI: avoid download
            freeze_backbone=True,
        ),
        data=AttrDict(image_size=224),
    )

    model = build_model(cfg)
    assert isinstance(model, ResNet18)

    # Ensure freeze_backbone took effect
    for name, param in model.backbone.named_parameters():
        if "fc" not in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_build_model_unknown_raises():
    cfg = make_cfg(
        model=AttrDict(name="some_weird_model", num_classes=4),
        data=AttrDict(image_size=224),
    )
    with pytest.raises(ValueError):
        build_model(cfg)


# ============================================
# DualPathwayModel Tests
# ============================================


def test_dual_pathway_initialization():
    """Test DualPathwayModel can be initialized."""
    model = DualPathwayModel(
        num_classes=4,
        radiomics_dim=50,
        radiomics_hidden=128,
        cnn_feature_dim=512,
        fusion_hidden=256,
        dropout=0.3,
        pretrained=False,
        freeze_backbone=False,
    )
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_dual_pathway_forward_with_features():
    """Test DualPathwayModel forward pass with radiomics features."""
    model = DualPathwayModel(
        num_classes=4,
        radiomics_dim=50,
        pretrained=False,
    )

    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    features = torch.randn(batch_size, 50)

    with torch.no_grad():
        output = model(images, features)

    assert output.shape == (batch_size, 4)


def test_dual_pathway_forward_without_features():
    """Test DualPathwayModel forward pass without features (backwards compatibility)."""
    model = DualPathwayModel(
        num_classes=4,
        radiomics_dim=50,
        pretrained=False,
    )

    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(images, features=None)

    # Should still work (CNN-only mode) with num_classes output
    assert output.shape == (batch_size, 4)


def test_dual_pathway_freeze_unfreeze():
    """Test DualPathwayModel freeze and unfreeze methods."""
    model = DualPathwayModel(num_classes=4, pretrained=False, freeze_backbone=True)

    # Check backbone is frozen
    for param in model.cnn_backbone.parameters():
        assert not param.requires_grad

    # Unfreeze and check
    model.unfreeze_backbone()
    for param in model.cnn_backbone.parameters():
        assert param.requires_grad


def test_build_model_dual_pathway():
    """build_model should return DualPathwayModel configured from cfg."""
    cfg = make_cfg(
        model=AttrDict(
            name="dual_pathway",
            num_classes=4,
            radiomics_dim=50,
            radiomics_hidden=128,
            cnn_feature_dim=512,
            fusion_hidden=256,
            dropout=0.3,
            pretrained=False,
            freeze_backbone=False,
        ),
        data=AttrDict(image_size=224),
        features=AttrDict(
            use_intensity=True,
            use_glcm=True,
            use_shape=True,
            use_region=True,
            use_wavelet=True,
        ),
    )

    model = build_model(cfg)
    assert isinstance(model, DualPathwayModel)

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    features = torch.randn(batch_size, 50)

    with torch.no_grad():
        output = model(images, features)

    assert output.shape == (batch_size, 4)


@pytest.mark.parametrize("model_name", ["dual_pathway", "dualpathway", "hybrid"])
def test_build_model_dual_pathway_aliases(model_name: str):
    """build_model should accept all dual pathway model aliases."""
    cfg = make_cfg(
        model=AttrDict(
            name=model_name,
            num_classes=4,
            pretrained=False,
        ),
        data=AttrDict(image_size=224),
    )

    model = build_model(cfg)
    assert isinstance(model, DualPathwayModel)
