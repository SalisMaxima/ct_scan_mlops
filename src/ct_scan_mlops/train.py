from __future__ import annotations

import random
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.model import build_model

# Find project root for Hydra config path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")


def configure_logging(output_dir: str) -> None:
    """Configure loguru for file and console logging.

    Args:
        output_dir: Directory to save log files
    """
    logger.remove()  # Remove default handler

    # File handler with rotation
    log_path = Path(output_dir) / "training.log"
    logger.add(
        log_path,
        level="DEBUG",
        rotation="100 MB",
        retention=5,
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Console handler for INFO and above
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    logger.info(f"Logging configured. Logs saved to {log_path}")


def get_device() -> torch.device:
    """Get the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, device: torch.device) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        device: Torch device being used
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - needed for legacy library compatibility
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_model(
    cfg: DictConfig,
    output_dir: str,
    device: torch.device,
) -> str:
    """Core training logic.

    Args:
        cfg: Hydra configuration
        output_dir: Directory to save outputs
        device: Torch device to train on

    Returns:
        Path to saved model checkpoint
    """
    output_path = Path(output_dir)
    train_cfg = cfg.train

    logger.info(f"Training on {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    set_seed(cfg.seed, device)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.max_epochs,
        eta_min=train_cfg.scheduler.eta_min,
    )

    # Training statistics
    statistics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = train_cfg.early_stopping.patience if train_cfg.early_stopping.enabled else float("inf")

    # Training loop
    for epoch in range(1, train_cfg.max_epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.max_epochs} [Train]")
        for batch_idx, (x, y) in enumerate(train_pbar):
            x, y = x.to(device), y.to(device)

            # Log sample images first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                wandb.log({"examples": [wandb.Image(x[j].cpu()) for j in range(min(8, len(x)))]})

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            if train_cfg.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip_val)

            optimizer.step()

            batch_acc = accuracy(logits.detach(), y)
            train_loss += loss.item()
            train_acc += batch_acc

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

            # Log batch metrics
            wandb.log(
                {
                    "batch/train_loss": loss.item(),
                    "batch/train_acc": batch_acc,
                    "batch/lr": optimizer.param_groups[0]["lr"],
                }
            )

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{train_cfg.max_epochs} [Val]")
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                batch_acc = accuracy(logits, y)
                val_loss += loss.item()
                val_acc += batch_acc

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Update scheduler
        scheduler.step()

        # Log epoch metrics
        current_lr = optimizer.param_groups[0]["lr"]
        statistics["train_loss"].append(train_loss)
        statistics["train_acc"].append(train_acc)
        statistics["val_loss"].append(val_loss)
        statistics["val_acc"].append(val_acc)
        statistics["lr"].append(current_lr)

        wandb.log(
            {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/train_acc": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_acc": val_acc,
                "epoch/lr": current_lr,
            }
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            best_model_path = output_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            logger.info(f"New best model saved: val_acc={val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience and epoch >= train_cfg.min_epochs:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    final_model_path = output_path / "model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Log model as W&B artifact
    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}_model",
        type="model",
        description=f"CT scan classifier trained for {epoch} epochs",
        metadata={
            "epochs_trained": epoch,
            "best_val_acc": best_val_acc,
            "final_train_loss": statistics["train_loss"][-1],
            "final_val_loss": statistics["val_loss"][-1],
            "model_name": cfg.model.name,
            "num_params": total_params,
            "seed": cfg.seed,
            **OmegaConf.to_container(cfg.model),
        },
    )
    artifact.add_file(str(output_path / "best_model.pt"))
    wandb.log_artifact(artifact)
    logger.info(f"Model artifact logged to W&B: {artifact.name}")

    # Save training plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(statistics["train_loss"], label="Train")
    axes[0, 0].plot(statistics["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(statistics["train_acc"], label="Train")
    axes[0, 1].plot(statistics["val_acc"], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(statistics["lr"])
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)

    # Summary text
    axes[1, 1].axis("off")
    summary_text = (
        f"Training Summary\n"
        f"================\n"
        f"Model: {cfg.model.name}\n"
        f"Epochs: {epoch}\n"
        f"Best Val Acc: {best_val_acc:.4f}\n"
        f"Final Train Loss: {statistics['train_loss'][-1]:.4f}\n"
        f"Final Val Loss: {statistics['val_loss'][-1]:.4f}\n"
        f"Seed: {cfg.seed}"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    fig_path = output_path / "training_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    wandb.log({"training_curves": wandb.Image(str(fig_path))})
    logger.info(f"Training curves saved to {fig_path}")

    return str(final_model_path)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train a model (Hydra entry point)."""
    # Get Hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    configure_logging(output_dir)

    device = get_device()

    # Initialize W&B
    wandb_cfg = cfg.wandb
    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity"),
        job_type="train",
        name=f"{cfg.experiment_name}_{cfg.model.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(wandb_cfg.get("tags", [])),
        mode=wandb_cfg.get("mode", "online"),
    )

    logger.info(f"W&B run: {run.url}")

    try:
        model_path = train_model(cfg, output_dir, device)
        logger.info(f"Training complete. Model saved to {model_path}")

        # Log final summary to W&B
        wandb.run.summary["output_dir"] = output_dir
        wandb.run.summary["model_path"] = model_path

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
