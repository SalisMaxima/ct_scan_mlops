"""Unified CLI for CT Scan Analysis Suite."""

from __future__ import annotations

from pathlib import Path

import typer
import wandb
from loguru import logger

from ct_scan_mlops.analysis.comparison import ModelComparator
from ct_scan_mlops.analysis.core import InferenceEngine, ModelLoader
from ct_scan_mlops.analysis.diagnostics import ModelDiagnostician
from ct_scan_mlops.analysis.explainability import FeatureExplainer
from ct_scan_mlops.analysis.utils import log_to_wandb
from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.utils import get_device

app = typer.Typer(help="CT Scan Analysis CLI")


@app.command()
def diagnose(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    output_dir: str = typer.Option("reports/diagnostics", "--output", "-o", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    max_images: int = typer.Option(50, help="Max images for error grids"),
):
    """Run full diagnostic suite (Metrics, Confusion Matrix, Error Analysis)."""
    checkpoint_path = Path(checkpoint)
    out_path = Path(output_dir)
    device = get_device()

    if use_wandb:
        wandb.init(project=project, name="diagnostics", job_type="analysis")

    # 1. Load
    loaded = ModelLoader.load(checkpoint_path, device, config_override=config)

    # 2. Dataloader
    logger.info("Creating dataloader...")
    _, _, test_loader = create_dataloaders(loaded.config, use_features=loaded.uses_features)

    # 3. Inference
    engine = InferenceEngine(loaded, device)
    # Collect images and features for deep analysis
    results = engine.run_inference(test_loader, collect_images=True, collect_features=True)

    # 4. Diagnostics
    diagnostician = ModelDiagnostician(results, out_path)

    # Metrics
    metrics = diagnostician.evaluate_performance()

    # Errors
    error_stats = diagnostician.analyze_errors(max_images=max_images)

    # Specific Confusion (Adeno vs Squamous)
    diagnostician.analyze_specific_confusion("adenocarcinoma", "squamous_cell_carcinoma")

    # W&B Logging
    if use_wandb:
        log_metrics = {**metrics, **error_stats}
        # Filter non-scalar stats
        scalar_metrics = {k: v for k, v in log_metrics.items() if isinstance(v, (int, float))}
        log_to_wandb(scalar_metrics, diagnostician.plots, "diagnostics")
        wandb.finish()

    logger.info(f"Diagnostics complete. Results in {out_path}")


@app.command()
def explain(
    checkpoint: str = typer.Option(..., help="Path to model checkpoint"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    output_dir: str = typer.Option("reports/feature_importance", "--output", "-o", help="Output directory"),
    method: str = typer.Option("both", help="Method: permutation, gradient, or both"),
    n_repeats: int = typer.Option(10, help="Repeats for permutation"),
):
    """Analyze feature importance (Permutation & Gradient)."""
    checkpoint_path = Path(checkpoint)
    out_path = Path(output_dir)
    device = get_device()

    # 1. Load
    loaded = ModelLoader.load(checkpoint_path, device, config_override=config)
    if not loaded.uses_features:
        logger.error("Model does not use radiomics features. Cannot run explanation.")
        raise typer.Exit(1)

    # 2. Dataloader
    _, _, test_loader = create_dataloaders(loaded.config, use_features=True)

    # 3. Explain
    explainer = FeatureExplainer(loaded, device, out_path)

    if method in ["permutation", "both"]:
        explainer.compute_permutation_importance(test_loader, n_repeats=n_repeats)

    if method in ["gradient", "both"]:
        explainer.compute_gradient_attribution(test_loader)

    logger.info(f"Explanation complete. Results in {out_path}")


@app.command()
def compare(
    baseline: str = typer.Option(..., help="Baseline checkpoint"),
    improved: str = typer.Option(..., help="Improved checkpoint"),
    output_dir: str = typer.Option("reports/comparison", "--output", "-o", help="Output directory"),
):
    """Compare two models."""
    comparator = ModelComparator(Path(baseline), Path(improved), Path(output_dir))
    comparator.run()


if __name__ == "__main__":
    app()
