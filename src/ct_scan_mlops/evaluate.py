"""Evaluate a trained model on the test set."""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import typer
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.model import build_model

app = typer.Typer()

# Get the relative path to the config directory from the project root
# Hydra requires relative paths, not absolute
_CONFIG_PATH = "../../configs"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    log_to_wandb: bool = False,
    save_confusion_matrix: bool = True,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """
    Core evaluation logic. Can be called standalone or from a sweep wrapper.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Torch device to evaluate on
        log_to_wandb: Whether to log metrics to wandb (assumes wandb is initialized)
        save_confusion_matrix: Whether to save confusion matrix plot
        output_dir: Directory to save outputs (confusion matrix, etc.)

    Returns:
        Dictionary with test metrics (accuracy, per-class metrics)
    """
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_targets = []

    logger.info("Running evaluation on test set...")

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_accuracy = correct / total
    logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")

    # Compute detailed metrics
    metrics = {"test_accuracy": test_accuracy}

    # Classification report
    report = classification_report(
        all_targets,
        all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Extract per-class metrics
    for class_name in CLASSES:
        if class_name in report:
            metrics[f"test_{class_name}_precision"] = report[class_name]["precision"]
            metrics[f"test_{class_name}_recall"] = report[class_name]["recall"]
            metrics[f"test_{class_name}_f1"] = report[class_name]["f1-score"]

    # Overall metrics
    metrics["test_macro_avg_f1"] = report["macro avg"]["f1-score"]
    metrics["test_weighted_avg_f1"] = report["weighted avg"]["f1-score"]

    # Print classification report
    logger.info("\nClassification Report:")
    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=CLASSES,
        zero_division=0,
    )
    print(report_str)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")

    # Save confusion matrix plot
    if save_confusion_matrix and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Accuracy: {test_accuracy:.4f})")
        plt.tight_layout()

        cm_path = output_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {cm_path}")

        if log_to_wandb:
            wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})

    # Log to wandb if requested
    if log_to_wandb:
        wandb.log(metrics)
        # Also set as summary metrics
        for key, value in metrics.items():
            wandb.run.summary[key] = value

    return metrics


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load model from checkpoint file.

    Handles both full checkpoint format (with optimizer state, etc.)
    and simple state_dict format.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Hydra config for building the model
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Build model from config
    model = build_model(cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint format (with optimizer, epoch, etc.)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if "val_acc" in checkpoint:
            logger.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
    else:
        # Simple state_dict format
        model.load_state_dict(checkpoint)

    return model


@app.command()
def evaluate_cli(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint (e.g., outputs/.../best_model.pt)"),
    config_path: str = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training config.yaml (default: uses default config)",
    ),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    wandb_entity: str = typer.Option(None, help="W&B entity (username or team)"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Batch size for evaluation (default: from config)"),
    output_dir: str = typer.Option(
        None, "--output", "-o", help="Output directory for plots (default: same as checkpoint)"
    ),
) -> dict[str, float]:
    """Evaluate a trained model on the test set (standalone CLI entry point).

    Examples:
        # Basic evaluation
        invoke evaluate --checkpoint outputs/2024-01-14/12-34-56/best_model.pt

        # With wandb logging
        invoke evaluate --checkpoint outputs/.../best_model.pt --wandb --wandb-entity YOUR_USERNAME

        # Custom batch size
        invoke evaluate --checkpoint models/best_model.pt --batch-size 64
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise typer.Exit(1)

    # Load config
    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        # Load default config
        logger.info("Using default config from configs/")
        with hydra.initialize(config_path=_CONFIG_PATH, version_base="1.3"):
            cfg = hydra.compose(config_name="config")

    # Override batch size if provided
    if batch_size:
        cfg.data.batch_size = batch_size

    # Set output directory
    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent

    logger.info(f"Output directory: {out_dir}")

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type="eval",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"eval_{cfg.model.name}_{checkpoint_path.stem}",
        )

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create dataloaders
    _, _, test_loader = create_dataloaders(cfg, use_processed=True)

    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_to_wandb=use_wandb,
        save_confusion_matrix=True,
        output_dir=out_dir,
    )

    if use_wandb:
        wandb.finish()

    return metrics


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def evaluate_hydra(cfg: DictConfig) -> None:
    """Evaluate with Hydra config (for integration with training pipeline).

    This can be called after training completes or as a separate step.
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Look for best_model.pt in the most recent output directory
    output_base = Path("outputs") / cfg.experiment_name
    if output_base.exists():
        # Find most recent run
        run_dirs = sorted(output_base.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            checkpoint_path = run_dirs[0] / "best_model.pt"
            if checkpoint_path.exists():
                logger.info(f"Found checkpoint: {checkpoint_path}")
            else:
                logger.error("No checkpoint found in recent outputs")
                return None
        else:
            logger.error(f"No training runs found in {output_base}")
            return None
    else:
        logger.error(f"Output directory not found: {output_base}")
        return None

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create dataloaders
    _, _, test_loader = create_dataloaders(cfg, use_processed=True)

    # Evaluate
    return evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=True,
        output_dir=checkpoint_path.parent,
    )


if __name__ == "__main__":
    app()
