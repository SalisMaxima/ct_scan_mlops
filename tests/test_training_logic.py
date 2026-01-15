"""Tests to validate models can actually learn (sanity checks)."""

import pytest
import torch

from ct_scan_mlops.model import CustomCNN, ResNet18


def test_custom_cnn_can_overfit_one_batch():
    """Test that CustomCNN can decrease loss on a single batch."""
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[32, 64],
        fc_hidden=128,
        dropout=0.0,  # Disable dropout for overfitting test
        batch_norm=True,
        image_size=224,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Create a small batch
    images = torch.randn(8, 3, 224, 224)
    targets = torch.randint(0, 4, (8,))

    model.train()
    initial_loss = criterion(model(images), targets).item()

    # Train for multiple iterations on the same batch
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert final_loss < initial_loss, "Model failed to decrease loss"
    assert final_loss < 1.0, f"Final loss {final_loss} should be < 1.0 after overfitting"


def test_resnet18_can_overfit_one_batch():
    """Test that ResNet18 can decrease loss on a single batch."""
    model = ResNet18(
        num_classes=4,
        pretrained=False,  # Don't download weights for testing
        freeze_backbone=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Create a small batch
    images = torch.randn(8, 3, 224, 224)
    targets = torch.randint(0, 4, (8,))

    model.train()
    initial_loss = criterion(model(images), targets).item()

    # Train for multiple iterations
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert final_loss < initial_loss, "ResNet18 failed to decrease loss"
    assert final_loss < 1.0, f"Final loss {final_loss} should be < 1.0 after overfitting"


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_custom_cnn_learns_with_various_batch_sizes(batch_size: int):
    """Test that model can learn regardless of batch size."""
    model = CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[16, 32],
        fc_hidden=64,
        dropout=0.0,
        image_size=224,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    images = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 4, (batch_size,))

    model.train()
    initial_loss = criterion(model(images), targets).item()

    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss, f"Failed to learn with batch_size={batch_size}"
