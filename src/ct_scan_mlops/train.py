from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from ct_scan_mlops.data import chest_ct
from ct_scan_mlops.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train(
    model: str = typer.Option("cnn", help="Model type: cnn or resnet18"),
    raw_dir: str = typer.Option("data/raw", help="Path to raw data folder"),
    image_size: int = typer.Option(224, help="Resize images to this size"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    epochs: int = typer.Option(3, help="Number of epochs"),
    save_path: str = typer.Option("models/model.pt", help="Where to save the trained weights"),
) -> None:
    # Data
    train_ds, val_ds, _test_ds = chest_ct(raw_dir=raw_dir, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = 4

    # Model
    net = build_model(model, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print(f"Training on: {DEVICE}")
    print(f"Model: {model} | train={len(train_ds)} | val={len(val_ds)}")

    # Training loop
    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        train_acc = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), y)

        train_loss /= max(1, len(train_loader))
        train_acc /= max(1, len(train_loader))

        # Validation
        net.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = net(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)

        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    # Save
    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path_p)
    print(f"Saved model weights to: {save_path_p}")


if __name__ == "__main__":
    typer.run(train)
