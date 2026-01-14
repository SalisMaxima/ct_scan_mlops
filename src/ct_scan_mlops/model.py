from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """A small baseline CNN for 4-class classification."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Factory for models: 'cnn' or 'resnet18'."""
    name = name.lower()

    if name in {"cnn", "custom", "customcnn"}:
        return CustomCNN(num_classes=num_classes)

    if name in {"resnet18", "resnet"}:
        # torchvision API differs slightly across versions; this is a safe pattern:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=pretrained)

        # Replace final layer for our num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError("Unknown model name. Use 'cnn' or 'resnet18'.")
