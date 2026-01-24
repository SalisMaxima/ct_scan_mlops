from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

# Model registry for extensible model building


class ModelConfigFactory(Protocol):
    @classmethod
    def from_config(cls, cfg: DictConfig) -> nn.Module: ...


MODEL_REGISTRY: dict[str, type[ModelConfigFactory]] = {}


def register_model(*names: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Decorator to register a model class with one or more names.

    Args:
        *names: One or more names to register the model under (case-insensitive).

    Returns:
        Decorator function that registers the class and returns it unchanged.

    Example:
        @register_model("custom_cnn", "cnn")
        class CustomCNN(nn.Module):
            ...
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        for name in names:
            MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


@register_model("custom_cnn", "cnn", "customcnn")
class CustomCNN(nn.Module):
    """Configurable CNN for image classification.

    Architecture is fully configurable via constructor parameters,
    designed to work with Hydra configs.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        hidden_dims: list[int] | None = None,
        fc_hidden: int = 512,
        dropout: float = 0.3,
        batch_norm: bool = True,
        kernel_size: int = 3,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        padding = kernel_size // 2

        # Build convolutional layers dynamically
        layers: list[nn.Module] = []
        prev_channels = in_channels

        for out_channels in hidden_dims:
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))  # Halves spatial dimensions
            prev_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calculate flattened size after conv layers
        # Each MaxPool2d halves the spatial dimensions
        num_pools = len(hidden_dims)
        final_spatial = image_size // (2**num_pools)
        flatten_size = hidden_dims[-1] * final_spatial * final_spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @classmethod
    def from_config(cls, cfg: DictConfig) -> CustomCNN:
        """Create CustomCNN from Hydra config."""
        model_cfg = cfg.model
        return cls(
            num_classes=model_cfg.num_classes,
            in_channels=model_cfg.get("input_channels", 3),
            hidden_dims=list(model_cfg.hidden_dims),
            fc_hidden=model_cfg.fc_hidden,
            dropout=model_cfg.dropout,
            batch_norm=model_cfg.batch_norm,
            image_size=cfg.data.image_size,
        )


@register_model("resnet18", "resnet")
class ResNet18(nn.Module):
    """ResNet18 for transfer learning.

    Supports freezing/unfreezing backbone for fine-tuning strategies.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load pretrained model
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ResNet18:
        """Create ResNet18 from Hydra config."""
        model_cfg = cfg.model
        return cls(
            num_classes=model_cfg.num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )


@register_model("dual_pathway", "dualpathway", "hybrid")
class DualPathwayModel(nn.Module):
    """Dual-pathway model combining CNN features with hand-crafted radiomics features.

    Architecture:
        Image Input
            |
            +---> CNN Backbone (ResNet18) ---> CNN Features (512-d)
            |                                        |
            +---> Feature Extractor ---> FC Layers ---> Radiomics Features (128-d)
                                                        |
                            Concatenate <---------------+
                                |
                            Fusion FC Layers
                                |
                            Classification

    This hybrid approach typically achieves ~0.817 AUC vs 0.801 for DL alone.
    """

    def __init__(
        self,
        num_classes: int = 4,
        radiomics_dim: int = 50,
        radiomics_hidden: int = 128,
        cnn_feature_dim: int = 512,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the dual-pathway model.

        Args:
            num_classes: Number of output classes.
            radiomics_dim: Input dimension of radiomics features.
            radiomics_hidden: Hidden dimension for radiomics pathway.
            cnn_feature_dim: Output dimension of CNN feature projection.
            fusion_hidden: Hidden dimension in fusion layers.
            dropout: Dropout rate.
            pretrained: Whether to use pretrained ResNet weights.
            freeze_backbone: Whether to freeze CNN backbone.
        """
        super().__init__()

        # CNN pathway (ResNet18 backbone)
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.cnn_backbone = models.resnet18(weights=weights)
        else:
            self.cnn_backbone = models.resnet18(weights=None)

        # Remove the final FC layer, keep features
        cnn_in_features = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()

        # CNN feature projection
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_in_features, cnn_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Radiomics feature pathway
        self.radiomics_projection = nn.Sequential(
            nn.Linear(radiomics_dim, radiomics_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(radiomics_hidden, radiomics_hidden),
            nn.ReLU(inplace=True),
        )

        # Fusion layers
        combined_dim = cnn_feature_dim + radiomics_hidden
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

        # CNN-only classifier (for backwards compatibility when features=None)
        self.cnn_classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

        # Optionally freeze CNN backbone
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze CNN backbone parameters."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze CNN backbone parameters."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through dual pathways.

        Args:
            image: Image tensor of shape (B, C, H, W).
            features: Hand-crafted features of shape (B, radiomics_dim).
                     If None, only CNN pathway is used (for backwards compatibility).

        Returns:
            Classification logits of shape (B, num_classes).
        """
        # CNN pathway
        cnn_features = self.cnn_backbone(image)
        cnn_features = self.cnn_projection(cnn_features)

        if features is None:
            # CNN-only mode (backwards compatible)
            return self.cnn_classifier(cnn_features)

        # Radiomics pathway
        radiomics_features = self.radiomics_projection(features)

        # Fusion
        combined = torch.cat([cnn_features, radiomics_features], dim=1)
        return self.fusion(combined)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> DualPathwayModel:
        """Create DualPathwayModel from Hydra config."""
        model_cfg = cfg.model

        # Calculate radiomics dimension based on enabled features
        # Default: intensity(8) + glcm(13) + shape(9) + region(6) + wavelet(14) = 50
        radiomics_dim = model_cfg.get("radiomics_dim", 50)

        return cls(
            num_classes=model_cfg.num_classes,
            radiomics_dim=radiomics_dim,
            radiomics_hidden=model_cfg.get("radiomics_hidden", 128),
            cnn_feature_dim=model_cfg.get("cnn_feature_dim", 512),
            fusion_hidden=model_cfg.get("fusion_hidden", 256),
            dropout=model_cfg.get("dropout", 0.3),
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )


def build_model(cfg: DictConfig) -> nn.Module:
    """Build model from Hydra config using the model registry.

    Models are registered with the @register_model decorator and must implement
    a from_config(cfg) class method for instantiation.

    Args:
        cfg: Hydra config containing model parameters.
             Expected to have cfg.model.name specifying which model to build.

    Returns:
        Configured model instance.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    name = cfg.model.name.lower()

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    model_cls = MODEL_REGISTRY[name]
    return model_cls.from_config(cfg)


if __name__ == "__main__":
    # Test model instantiation and forward pass
    print("=" * 60)
    print("CustomCNN Model Test")
    print("=" * 60)

    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[32, 64, 128, 256],
        fc_hidden=512,
        dropout=0.3,
        batch_norm=True,
        image_size=224,
    )

    print(model)
    print()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print()

    print("=" * 60)
    print("ResNet18 Model Test")
    print("=" * 60)

    resnet = ResNet18(num_classes=4, pretrained=False, freeze_backbone=False)
    total_params = sum(p.numel() for p in resnet.parameters())
    trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    output = resnet(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
