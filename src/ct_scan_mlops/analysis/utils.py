"""Shared utility functions for analysis modules."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger

# Import core logic from the new core module
# This ensures backward compatibility while migrating to the new architecture
from ct_scan_mlops.analysis.core import (
    DUAL_PATHWAY_MODEL_NAMES,
    LoadedModel,
    ModelLoader,
    model_forward,
    unpack_batch,
)

# Re-export for compatibility
__all__ = [
    "DUAL_PATHWAY_MODEL_NAMES",
    "LoadedModel",
    "ModelLoader",
    "model_forward",
    "unpack_batch",
    "load_feature_metadata",
    "denormalize_features",
    "save_image_grid",
    "compute_calibration_error",
    "log_to_wandb",
]


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
