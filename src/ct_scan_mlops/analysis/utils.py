"""Shared utility functions for analysis modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ct_scan_mlops.evaluate import load_model_from_checkpoint

# Model names that use radiomics features
DUAL_PATHWAY_MODEL_NAMES = frozenset({"dual_pathway", "dualpathway", "hybrid"})


@dataclass
class LoadedModel:
    """Container for loaded model with metadata."""

    model: nn.Module
    config: DictConfig
    uses_features: bool
    model_name: str
    checkpoint_path: Path


class ModelLoader:
    """Centralized model loading with automatic config detection.

    Handles:
    - Finding config from checkpoint directory (.hydra/config.yaml)
    - Fallback to CLI-provided config path
    - Auto-detecting model type and feature requirements
    - Consistent model loading interface
    """

    @staticmethod
    def detect_uses_features(cfg: DictConfig) -> bool:
        """Detect if model uses radiomics features based on config.

        Args:
            cfg: Model configuration

        Returns:
            True if model uses radiomics features
        """
        model_name = cfg.model.name.lower()
        return model_name in DUAL_PATHWAY_MODEL_NAMES

    @staticmethod
    def find_config(
        checkpoint_path: Path,
        config_override: Path | str | None = None,
    ) -> DictConfig:
        """Find and load config for a checkpoint.

        Search order:
        1. CLI-provided config override
        2. .hydra/config.yaml in checkpoint's parent directory
        3. Raise FileNotFoundError with helpful message

        Args:
            checkpoint_path: Path to model checkpoint
            config_override: Optional path to config file

        Returns:
            Loaded OmegaConf DictConfig

        Raises:
            FileNotFoundError: If no config found
        """
        if config_override is not None:
            config_path = Path(config_override)
            if not config_path.exists():
                raise FileNotFoundError(f"Config override not found: {config_path}")
            return OmegaConf.load(config_path)

        # Try .hydra/config.yaml in checkpoint directory
        hydra_config = checkpoint_path.parent / ".hydra" / "config.yaml"
        if hydra_config.exists():
            return OmegaConf.load(hydra_config)

        raise FileNotFoundError(
            f"Config not found for checkpoint {checkpoint_path}. "
            f"Searched: {hydra_config}\n"
            f"Provide config via --config/-c option."
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str,
        device: torch.device,
        config_override: Path | str | None = None,
    ) -> LoadedModel:
        """Load model from checkpoint with automatic config detection.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            config_override: Optional config path override

        Returns:
            LoadedModel with model, config, and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        cfg = cls.find_config(checkpoint_path, config_override)
        uses_features = cls.detect_uses_features(cfg)
        model_name = cfg.model.name.lower()

        model = load_model_from_checkpoint(checkpoint_path, cfg, device)

        logger.info(f"Loaded {model_name} model (uses_features={uses_features}) from {checkpoint_path}")

        return LoadedModel(
            model=model,
            config=cfg,
            uses_features=uses_features,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )


def unpack_batch(
    batch: tuple,
    device: torch.device,
    use_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Unpack a batch from dataloader handling both 2-tuple and 3-tuple formats.

    Args:
        batch: Batch from dataloader (2-tuple or 3-tuple)
        device: Device to move tensors to
        use_features: Whether to expect and use features

    Returns:
        Tuple of (images, features, targets)
        features is None if batch is 2-tuple or use_features is False
    """
    if len(batch) == 3:
        images, features, targets = batch
        images = images.to(device)
        features = features.to(device) if use_features else None
        targets = targets.to(device)
    else:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        features = None

    return images, features, targets


def model_forward(
    model: nn.Module,
    images: torch.Tensor,
    features: torch.Tensor | None,
    use_features: bool,
) -> torch.Tensor:
    """Run model forward pass handling feature/no-feature cases.

    Args:
        model: Model to run
        images: Input images
        features: Radiomics features (can be None)
        use_features: Whether model expects features

    Returns:
        Model outputs
    """
    if use_features and features is not None:
        return model(images, features)
    return model(images)


def load_feature_metadata(processed_dir: Path = Path("data/processed")) -> dict:
    """Load feature_metadata.json with names and normalization stats.

    Args:
        processed_dir: Directory containing feature_metadata.json

    Returns:
        Dictionary with feature_names, normalization (mean/std), and config
    """
    metadata_path = processed_dir / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Feature metadata not found at {metadata_path}")

    with metadata_path.open() as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata for {len(metadata['feature_names'])} features")
    return metadata


def denormalize_features(features: torch.Tensor, metadata: dict) -> torch.Tensor:
    """Denormalize features using training set statistics.

    Args:
        features: Normalized features tensor (batch_size, 50)
        metadata: Feature metadata dict with normalization stats

    Returns:
        Denormalized features tensor
    """
    mean = torch.tensor(metadata["normalization"]["mean"], dtype=features.dtype, device=features.device)
    std = torch.tensor(metadata["normalization"]["std"], dtype=features.dtype, device=features.device)

    return features * std + mean


def save_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    ncols: int = 5,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Save a grid of images with titles.

    Args:
        images: List of image arrays (H, W, C) or (C, H, W)
        titles: List of titles for each image
        output_path: Path to save the grid
        ncols: Number of columns in the grid
        figsize: Figure size (width, height). Auto-calculated if None
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols  # Ceiling division

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img, title) in enumerate(zip(images, titles, strict=True)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Convert from (C, H, W) to (H, W, C) if needed
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0))

        # Remove channel dimension if grayscale
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_images, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved image grid to {output_path}")


def compute_calibration_error(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.

    Args:
        probabilities: Predicted probabilities (N, num_classes)
        predictions: Predicted class indices (N,)
        targets: True class indices (N,)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better, range [0, 1])
    """
    # Get confidence (max probability) for each prediction
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find predictions in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def log_to_wandb(metrics: dict, plots: dict[str, Path], run_name: str) -> None:
    """Log metrics and plots to W&B if wandb is active.

    Args:
        metrics: Dictionary of metrics to log
        plots: Dictionary mapping plot names to file paths
        run_name: Name prefix for the W&B run
    """
    if wandb.run is None:
        logger.warning("W&B run not active, skipping logging")
        return

    # Log metrics
    wandb.log(metrics)
    logger.info(f"Logged {len(metrics)} metrics to W&B")

    # Log plots
    for plot_name, plot_path in plots.items():
        if plot_path.exists():
            wandb.log({plot_name: wandb.Image(str(plot_path))})
            logger.info(f"Logged plot {plot_name} to W&B")
        else:
            logger.warning(f"Plot {plot_name} not found at {plot_path}")
