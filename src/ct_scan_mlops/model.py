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
