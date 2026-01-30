"""Backward compatibility wrapper for evaluate_model function.

This module provides a compatibility shim for the old evaluate_model() API.
The actual functionality has been migrated to the modular analysis system:
- ct_scan_mlops.analysis.core (loading and inference)
- ct_scan_mlops.analysis.diagnostics (evaluation metrics)

DEPRECATED: This module exists only for backward compatibility.
New code should use the analysis module directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from ct_scan_mlops.analysis.core import (
    InferenceEngine,
    LoadedModel,
    PredictionResults,
    load_model_from_checkpoint,
)
from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician

# Import wandb at module level for test mocking
try:
    import wandb
except ImportError:
    wandb = None


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    log_to_wandb: bool = False,
    save_confusion_matrix: bool = True,
    output_dir: Path | str | None = None,
    use_features: bool = False,
) -> dict[str, float]:
    """Evaluate model on test data and compute metrics.

    DEPRECATED: This function exists for backward compatibility only.
    New code should use:
        from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician
        from ct_scan_mlops.analysis.core import InferenceEngine

    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        log_to_wandb: Whether to log metrics to W&B (default: False)
        save_confusion_matrix: Whether to save confusion matrix plot (default: True)
        output_dir: Directory to save outputs (default: current directory)
        use_features: Whether model uses radiomics features (default: False)

    Returns:
        Dictionary of evaluation metrics including:
        - test_accuracy: Overall accuracy
        - test_macro_avg_f1: Macro-averaged F1 score
        - test_weighted_avg_f1: Weighted-averaged F1 score
        - test_{class}_precision/recall/f1: Per-class metrics
    """
    warnings.warn(
        "evaluate_model() is deprecated. Use ct_scan_mlops.analysis.diagnostics.ModelDiagnostician instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Set output directory
    output_dir = Path() if output_dir is None else Path(output_dir)

    # Create a pseudo LoadedModel for compatibility
    from omegaconf import OmegaConf  # noqa: PLC0415

    pseudo_config = OmegaConf.create({"model": {"name": "custom"}})
    pseudo_loaded = LoadedModel(
        model=model,
        config=pseudo_config,
        uses_features=use_features,
        model_name="legacy",
        checkpoint_path=Path(),
    )

    # Run inference using new engine
    logger.info("Running inference...")
    engine = InferenceEngine(loaded_model=pseudo_loaded, device=device)
    results: PredictionResults = engine.run_inference(test_loader)

    # Evaluate using new diagnostics
    logger.info("Computing metrics...")
    diagnostician = ModelDiagnostician(results=results, output_dir=output_dir)

    # Get metrics (returns dict without "test_" prefix)
    metrics = diagnostician.evaluate_performance()

    # Add "test_" prefix to match old API
    test_metrics = {}
    test_metrics["test_accuracy"] = metrics["accuracy"]
    test_metrics["test_macro_avg_f1"] = metrics["macro_f1"]
    test_metrics["test_weighted_avg_f1"] = metrics["weighted_f1"]

    # Per-class metrics
    from ct_scan_mlops.data import CLASSES

    for cls in CLASSES:
        if f"{cls}_precision" in metrics:
            test_metrics[f"test_{cls}_precision"] = metrics[f"{cls}_precision"]
            test_metrics[f"test_{cls}_recall"] = metrics[f"{cls}_recall"]
            test_metrics[f"test_{cls}_f1"] = metrics[f"{cls}_f1"]

    # Log to W&B if requested
    if log_to_wandb:
        if wandb is None:
            logger.warning("W&B not installed, skipping logging")
        elif wandb.run is not None:
            wandb.log(test_metrics)
            logger.info("Logged metrics to W&B")
        else:
            logger.warning("W&B run not initialized, skipping logging")

    logger.info(f"Evaluation complete: Accuracy = {test_metrics['test_accuracy']:.4f}")
    return test_metrics


__all__ = ["evaluate_model", "load_model_from_checkpoint"]
