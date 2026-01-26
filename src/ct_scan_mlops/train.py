from __future__ import annotations

import random
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from ct_scan_mlops.data import ChestCTDataModule
from ct_scan_mlops.losses import build_loss
from ct_scan_mlops.model import build_model
from ct_scan_mlops.utils import get_device

# Find project root for Hydra config path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")

profiling_dir = _PROJECT_ROOT / "artifacts" / "profiling"
profiling_dir.mkdir(parents=True, exist_ok=True)


class LitModel(pl.LightningModule):
    """Lightning module wrapping the CT scan classifier."""

    def __init__(self, cfg: DictConfig, use_features: bool = False):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterion = build_loss(cfg)
        self.use_features = use_features
        self.save_hyperparameters(ignore=["model"])

        # Track training history for plotting
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Cache profiling config (parsed once, not every training step)
        profiling_cfg = cfg.get("train", {}).get("profiling", {})
        self._profiling_enabled = bool(profiling_cfg.get("enabled", True))
        self._profiling_steps = int(profiling_cfg.get("steps", 5))
        self._profiled = False

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_features and features is not None:
            return self.model(x, features)
        return self.model(x)

    def _unpack_batch(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Unpack batch handling both 2-tuple and 3-tuple formats.

        Returns:
            Tuple of (images, features_or_none, labels)
        """
        if len(batch) == 3:
            # RadiomicsChestCTDataset format: (image, features, label)
            x, features, y = batch
            return x, features, y
        # Standard format: (image, label)
        x, y = batch
        return x, None, y

    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification accuracy."""
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean()

    def training_step(self, batch, batch_idx: int):
        x, features, y = self._unpack_batch(batch)

        # Run profiling once on first batch of first epoch (if enabled)
        if self._profiling_enabled and not self._profiled and batch_idx == 0 and self.current_epoch == 0:
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True,
                on_trace_ready=tensorboard_trace_handler(str(profiling_dir)),
            ) as prof:
                for _ in range(max(1, self._profiling_steps)):
                    y_hat = self(x, features)
                    loss = self.criterion(y_hat, y)
                    prof.step()

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            self._profiled = True
        else:
            y_hat = self(x, features)
            loss = self.criterion(y_hat, y)

        acc = self._compute_accuracy(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, features, y = self._unpack_batch(batch)
        y_hat = self(x, features)
        loss = self.criterion(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx: int):
        """Test step for model evaluation."""
        x, features, y = self._unpack_batch(batch)
        y_hat = self(x, features)
        loss = self.criterion(y_hat, y)
        acc = self._compute_accuracy(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return loss

    def on_train_start(self):
        """Log sample images at training start."""
        if self.trainer.train_dataloader is None or self.logger is None:
            return

        # Only log images if using WandbLogger
        if not isinstance(self.logger, WandbLogger):
            return

        batch = next(iter(self.trainer.train_dataloader))
        x, _features, _y = self._unpack_batch(batch)
        self.logger.experiment.log({"examples": [wandb.Image(x[j].cpu()) for j in range(min(8, len(x)))]})

    def on_train_epoch_end(self):
        """Track metrics at end of each epoch for plotting."""
        metrics = self.trainer.callback_metrics
        self.training_history["train_loss"].append(
            metrics.get("train_loss", 0).item()
            if torch.is_tensor(metrics.get("train_loss", 0))
            else metrics.get("train_loss", 0)
        )
        self.training_history["train_acc"].append(
            metrics.get("train_acc", 0).item()
            if torch.is_tensor(metrics.get("train_acc", 0))
            else metrics.get("train_acc", 0)
        )
        self.training_history["val_loss"].append(
            metrics.get("val_loss", 0).item()
            if torch.is_tensor(metrics.get("val_loss", 0))
            else metrics.get("val_loss", 0)
        )
        self.training_history["val_acc"].append(
            metrics.get("val_acc", 0).item()
            if torch.is_tensor(metrics.get("val_acc", 0))
            else metrics.get("val_acc", 0)
        )

        # Get current LR from optimizer
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.training_history["lr"].append(current_lr)

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        sched_cfg = self.cfg.train.scheduler

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.max_epochs,
            eta_min=sched_cfg.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


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


def train_model(
    cfg: DictConfig,
    output_dir: str,
    wandb_logger: WandbLogger | None = None,
) -> str:
    """Train using PyTorch Lightning with full MLOps features.

    Args:
        cfg: Hydra configuration
        output_dir: Directory to save outputs
        wandb_logger: WandbLogger instance for experiment tracking

    Returns:
        Path to saved model checkpoint
    """
    output_path = Path(output_dir)
    train_cfg = cfg.train

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Seed for reproducibility
    set_seed(cfg.seed, get_device())
    pl.seed_everything(cfg.seed, workers=True)

    # Detect if using dual-pathway model (requires radiomics features)
    use_features = cfg.model.name in ("dual_pathway", "dualpathway", "hybrid")
    if use_features:
        logger.info("Dual-pathway model detected - enabling radiomics features")
        # Verify feature files exist
        processed_path = Path(cfg.data.get("processed_path", "data/processed"))
        features_file = processed_path / "train_features.pt"
        if not features_file.exists():
            raise FileNotFoundError(
                f"Radiomics features not found at {features_file}. "
                "Run 'invoke extract-features' first before training dual_pathway model."
            )

    # Data - using LightningDataModule for best practices
    # Note: Trainer.fit() calls datamodule.setup() automatically, but we call it here
    # to log dataset sizes before training starts
    datamodule = ChestCTDataModule(cfg, use_features=use_features)
    datamodule.setup(stage="fit")
    logger.info(f"Train batches: {len(datamodule.train_dataloader())}, Val batches: {len(datamodule.val_dataloader())}")

    # Lightning model
    lit_model = LitModel(cfg, use_features=use_features)

    # Log model info
    total_params = sum(p.numel() for p in lit_model.model.parameters())
    trainable_params = sum(p.numel() for p in lit_model.model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ---- Callbacks ----
    callbacks = []

    # ModelCheckpoint - saves best model based on validation metric
    ckpt_cfg = train_cfg.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename="best_model",
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_top_k=ckpt_cfg.save_top_k,
        save_last=ckpt_cfg.save_last,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # EarlyStopping - stops training when metric plateaus
    es_cfg = train_cfg.early_stopping
    if es_cfg.enabled:
        early_stop_callback = EarlyStopping(
            monitor=es_cfg.monitor,
            patience=es_cfg.patience,
            mode=es_cfg.mode,
            min_delta=0.001,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(f"Early stopping enabled: monitor={es_cfg.monitor}, patience={es_cfg.patience}")

    # LearningRateMonitor - logs LR to W&B
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # ---- Trainer ----
    trainer = pl.Trainer(
        default_root_dir=str(output_path),
        max_epochs=train_cfg.max_epochs,
        min_epochs=train_cfg.get("min_epochs", 1),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        precision=train_cfg.get("precision", 32),
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=10,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    # ---- Train ----
    trainer.fit(lit_model, datamodule=datamodule)

    # ---- Save final model weights (pure PyTorch format for easy loading) ----
    final_model_path = output_path / "model.pt"
    torch.save(lit_model.model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # ---- Generate training curves ----
    if lit_model.training_history["train_loss"]:
        _save_training_curves(lit_model.training_history, output_path, cfg, wandb_logger)

    # ---- Log model artifact with rich metadata ----
    best_val_acc = checkpoint_callback.best_model_score
    best_val_acc_value = best_val_acc.item() if best_val_acc is not None else None

    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}_model",
        type="model",
        description=f"CT scan classifier trained for {trainer.current_epoch} epochs",
        metadata={
            "epochs_trained": trainer.current_epoch,
            "best_val_acc": best_val_acc_value,
            "final_train_loss": lit_model.training_history["train_loss"][-1]
            if lit_model.training_history["train_loss"]
            else None,
            "final_val_loss": lit_model.training_history["val_loss"][-1]
            if lit_model.training_history["val_loss"]
            else None,
            "model_name": cfg.model.name,
            "num_params": total_params,
            "trainable_params": trainable_params,
            "seed": cfg.seed,
            **OmegaConf.to_container(cfg.model),
        },
    )

    # Add best model checkpoint if it exists
    best_ckpt_path = output_path / "best_model.ckpt"
    if best_ckpt_path.exists():
        artifact.add_file(str(best_ckpt_path))
    artifact.add_file(str(final_model_path))

    # Add the composed Hydra config if available
    hydra_cfg_path = output_path / ".hydra" / "config.yaml"
    if hydra_cfg_path.exists():
        artifact.add_file(str(hydra_cfg_path))

    wandb.log_artifact(artifact)
    logger.info(f"Model artifact logged to W&B: {artifact.name}")

    return str(final_model_path)


def _save_training_curves(
    history: dict,
    output_path: Path,
    cfg: DictConfig,
    wandb_logger: WandbLogger | None,
) -> None:
    """Generate and save training curves plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0, 0].plot(epochs, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    axes[0, 1].plot(epochs, history["train_acc"], label="Train")
    axes[0, 1].plot(epochs, history["val_acc"], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning rate plot
    if history["lr"]:
        axes[1, 0].plot(epochs, history["lr"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True)

    # Summary text
    axes[1, 1].axis("off")
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0
    summary_text = (
        f"Training Summary\n"
        f"================\n"
        f"Model: {cfg.model.name}\n"
        f"Epochs: {len(history['train_loss'])}\n"
        f"Best Val Acc: {best_val_acc:.4f}\n"
        f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
        f"Final Val Loss: {history['val_loss'][-1]:.4f}\n"
        f"Seed: {cfg.seed}"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family="monospace", verticalalignment="center")

    plt.tight_layout()
    fig_path = output_path / "training_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    if wandb_logger:
        wandb_logger.experiment.log({"training_curves": wandb.Image(str(fig_path))})
    logger.info(f"Training curves saved to {fig_path}")


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train a model (Hydra entry point)."""
    # Get Hydra's output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    configure_logging(output_dir)

    device = get_device()
    logger.info(f"Training on {device}")

    # Initialize W&B Logger for Lightning integration
    wandb_cfg = cfg.wandb
    wandb_logger = WandbLogger(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity"),
        name=f"{cfg.experiment_name}_{cfg.model.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=list(wandb_cfg.get("tags", [])),
        mode=wandb_cfg.get("mode", "online"),
        save_dir=output_dir,
        job_type="train",
    )

    logger.info(f"W&B run: {wandb_logger.experiment.url}")

    try:
        model_path = train_model(cfg, output_dir, wandb_logger)
        logger.info(f"Training complete. Model saved to {model_path}")

        # Log final summary to W&B
        wandb_logger.experiment.summary["output_dir"] = output_dir
        wandb_logger.experiment.summary["model_path"] = model_path

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
