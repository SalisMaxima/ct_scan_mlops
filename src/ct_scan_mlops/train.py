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
import pytorch_lightning as pl


from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.model import build_model

# Find project root for Hydra config path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")

import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        return torch.optim.Adam(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

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
) -> str:
    """Train using PyTorch Lightning."""
    output_path = Path(output_dir)
    train_cfg = cfg.train

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Seed for reproducibility
    set_seed(cfg.seed, get_device())
    pl.seed_everything(cfg.seed, workers=True)

    # Data
    train_loader, val_loader, _ = create_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Lightning model
    lit_model = LitModel(cfg)

    # Trainer (simple)
    trainer = pl.Trainer(
        default_root_dir=str(output_path),
        max_epochs=train_cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        enable_checkpointing=True,  # default checkpointing, no extra imports
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model weights 
    final_model_path = output_path / "model.pt"
    torch.save(lit_model.model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}_model",
        type="model",
        description="CT scan classifier (Lightning)",
        metadata={
            "model_name": cfg.model.name,
            "seed": cfg.seed,
            "epochs": train_cfg.max_epochs,
        },
    )
    artifact.add_file(str(final_model_path))
    wandb.log_artifact(artifact)

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
        model_path = train_model(cfg, output_dir)
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
