"""Evaluate a trained model on the test set."""

from __future__ import annotations

from pathlib import Path

# Add OmegaConf and typing classes to safe globals for checkpoint loading
# PyTorch 2.6 requires explicit allowlisting for weights_only=True
from typing import Any

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import typer
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import Container, ContainerMetadata
from sklearn.metrics import classification_report, confusion_matrix

from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.model import build_model
from ct_scan_mlops.utils import get_device

torch.serialization.add_safe_globals([DictConfig, Container, ContainerMetadata, Any])

app = typer.Typer()

# Get the relative path to the config directory from the project root
# Hydra requires relative paths, not absolute
_CONFIG_PATH = "../../configs"


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    log_to_wandb: bool = False,
    save_confusion_matrix: bool = True,
    output_dir: Path | None = None,
    use_features: bool = False,
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
        use_features: Whether test_loader returns (image, features, label) tuples

    Returns:
        Dictionary with test metrics (accuracy, per-class metrics)
    """
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_targets = []

    logger.info("Running evaluation on test set...")

    with torch.no_grad():
        for batch in test_loader:
            # Handle both 2-tuple and 3-tuple batch formats
            if len(batch) == 3:
                images, features, targets = batch
                images = images.to(device)
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(images, features) if use_features else model(images)
            else:
                images, targets = batch
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

    Handles multiple checkpoint formats:
    - Lightning .ckpt format (state_dict key)
    - Full checkpoint format (model_state_dict key)
    - Simple state_dict format (.pt file)

    Args:
        checkpoint_path: Path to checkpoint file (.pt or .ckpt)
        cfg: Hydra config for building the model
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Build model from config
    model = build_model(cfg).to(device)

    # Load checkpoint (weights_only=False for OmegaConf/Lightning compatibility)
    # Note: These are trusted checkpoints from our own training runs
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec B614

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            # Lightning .ckpt format - model weights are under 'model.' prefix
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix from Lightning checkpoint keys
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    model_state_dict[key[6:]] = value  # Remove 'model.' prefix
                else:
                    model_state_dict[key] = value
            model.load_state_dict(model_state_dict)
            logger.info(f"Loaded Lightning checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        elif "model_state_dict" in checkpoint:
            # Full checkpoint format (with optimizer, epoch, etc.)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if "val_acc" in checkpoint:
                logger.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
        else:
            # Dict but no recognized key - treat as raw state_dict
            model.load_state_dict(checkpoint)
    else:
        # Simple state_dict format
        model.load_state_dict(checkpoint)

    return model


@app.command()
def evaluate_cli(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint (.ckpt or .pt)"),
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

    Supports both Lightning checkpoints (.ckpt) and PyTorch state dicts (.pt).

    Examples:
        # Basic evaluation (Lightning checkpoint)
        invoke evaluate --checkpoint outputs/ct_scan_classifier/2024-01-14/12-34-56/best_model.ckpt

        # Evaluate PyTorch state dict
        invoke evaluate --checkpoint outputs/.../model.pt

        # With wandb logging
        invoke evaluate --checkpoint outputs/.../best_model.ckpt --wandb --wandb-entity YOUR_USERNAME

        # Custom batch size
        invoke evaluate --checkpoint models/best_model.ckpt --batch-size 64
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

    # Detect if using dual-pathway model
    use_features = cfg.model.name in ("dual_pathway", "dualpathway", "hybrid")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create dataloaders (with features if needed)
    _, _, test_loader = create_dataloaders(cfg, use_processed=True, use_features=use_features)

    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_to_wandb=use_wandb,
        save_confusion_matrix=True,
        output_dir=out_dir,
        use_features=use_features,
    )

    if use_wandb:
        wandb.finish()

    return metrics


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def evaluate_hydra(cfg: DictConfig) -> dict[str, float] | None:
    """Evaluate with Hydra config (for integration with training pipeline).

    This can be called after training completes or as a separate step.
    Automatically finds the best checkpoint from recent training runs.
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Look for checkpoints in the most recent output directory
    output_base = Path("outputs") / cfg.experiment_name
    checkpoint_path = None

    if output_base.exists():
        # Find most recent run
        run_dirs = sorted(output_base.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            run_dir = run_dirs[0]
            # Try Lightning checkpoint first (.ckpt), then PyTorch (.pt)
            for filename in ["best_model.ckpt", "best_model.pt", "model.pt", "last.ckpt"]:
                candidate = run_dir / filename
                if candidate.exists():
                    checkpoint_path = candidate
                    logger.info(f"Found checkpoint: {checkpoint_path}")
                    break

            if checkpoint_path is None:
                logger.error(f"No checkpoint found in {run_dir}")
                return None
        else:
            logger.error(f"No training runs found in {output_base}")
            return None
    else:
        logger.error(f"Output directory not found: {output_base}")
        return None

    # Detect if using dual-pathway model
    use_features = cfg.model.name in ("dual_pathway", "dualpathway", "hybrid")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create dataloaders (with features if needed)
    _, _, test_loader = create_dataloaders(cfg, use_processed=True, use_features=use_features)

    # Evaluate
    return evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        log_to_wandb=False,
        save_confusion_matrix=True,
        output_dir=checkpoint_path.parent,
        use_features=use_features,
    )


if __name__ == "__main__":
    app()
