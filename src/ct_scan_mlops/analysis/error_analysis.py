"""Analyze misclassified samples to identify failure patterns."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
import wandb
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_scan_mlops.analysis.utils import load_feature_metadata, log_to_wandb, save_image_grid
from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.evaluate import load_model_from_checkpoint
from ct_scan_mlops.utils import get_device

app = typer.Typer()


@dataclass
class ErrorCase:
    """Represents a single misclassified sample."""

    sample_idx: int
    true_label: str
    pred_label: str
    confidence: float
    logits: list[float]
    probabilities: list[float]
    features: list[float]
    image_array: list[list[list[float]]] | None = None  # For JSON serialization


def collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_features: bool = True,
    collect_images: bool = True,
) -> tuple[list[ErrorCase], np.ndarray, np.ndarray]:
    """Run inference and collect all predictions with metadata.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        use_features: Whether model uses features
        collect_images: Whether to collect image arrays for visualization

    Returns:
        Tuple of (error_cases, all_predictions, all_targets)
    """
    model.eval()
    error_cases = []
    all_preds = []
    all_targets = []

    sample_idx = 0

    logger.info("Collecting predictions and error cases...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Batches"):
            if len(batch) == 3:
                images, features, targets = batch
                images = images.to(device)
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(images, features) if use_features else model(images)
            else:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                features = None

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            # Process each sample in batch
            for i in range(len(targets)):
                pred_class = preds[i].item()
                true_class = targets[i].item()

                all_preds.append(pred_class)
                all_targets.append(true_class)

                # Collect error cases
                if pred_class != true_class:
                    error_case = ErrorCase(
                        sample_idx=sample_idx,
                        true_label=CLASSES[true_class],
                        pred_label=CLASSES[pred_class],
                        confidence=float(probs[i, pred_class]),
                        logits=outputs[i].cpu().numpy().tolist(),
                        probabilities=probs[i].cpu().numpy().tolist(),
                        features=features[i].cpu().numpy().tolist() if features is not None else [],
                        image_array=images[i].cpu().numpy().tolist() if collect_images else None,
                    )
                    error_cases.append(error_case)

                sample_idx += 1

    logger.info(f"Found {len(error_cases)} misclassified samples out of {sample_idx} total")
    return error_cases, np.array(all_preds), np.array(all_targets)


def analyze_error_patterns(
    error_cases: list[ErrorCase],
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Identify patterns in misclassifications.

    Args:
        error_cases: List of misclassified samples
        all_preds: All predictions
        all_targets: All true labels
        feature_names: Names of radiomics features

    Returns:
        Dictionary with error analysis results
    """
    logger.info("Analyzing error patterns...")

    # 1. Confusion pairs
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    for error in error_cases:
        pair = (error.true_label, error.pred_label)
        confusion_pairs[pair] += 1

    # 2. Confidence distribution
    confidences = [error.confidence for error in error_cases]
    hist, bins = np.histogram(confidences, bins=10)
    confidence_stats: dict[str, float | dict[str, list[float]]] = {
        "mean": float(np.mean(confidences)),
        "median": float(np.median(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "histogram": {
            "counts": hist.tolist(),
            "bins": bins.tolist(),
        },
    }

    # 3. Per-class error rate
    per_class_error_rate = {}
    for class_idx, class_name in enumerate(CLASSES):
        class_mask = all_targets == class_idx
        if class_mask.sum() > 0:
            class_errors = (all_preds[class_mask] != all_targets[class_mask]).sum()
            error_rate = float(class_errors / class_mask.sum())
            per_class_error_rate[class_name] = error_rate

    # 4. Feature differences (t-tests for errors vs correct)
    discriminative_features: list[dict[str, float | str]] = []
    if len(error_cases) > 0 and len(error_cases[0].features) > 0:
        error_features = np.array([error.features for error in error_cases])

        # Collect correct predictions' features
        # Note: We need to re-collect correct features from the dataloader
        # For simplicity, we'll compare against mean feature values

        for i, feature_name in enumerate(feature_names):
            if i < error_features.shape[1]:
                feature_values = error_features[:, i]
                # Placeholder: Would need correct predictions' features for proper t-test
                # For now, just compute descriptive stats
                discriminative_features.append(
                    {
                        "feature": feature_name,
                        "mean_value": float(np.mean(feature_values)),
                        "std_value": float(np.std(feature_values)),
                    }
                )

        # Sort by absolute mean value (proxy for importance in errors)
        discriminative_features.sort(key=lambda x: abs(float(x["mean_value"])), reverse=True)
    else:
        discriminative_features = []

    confusion_pairs_dict = {f"{k[0]}->{k[1]}": v for k, v in confusion_pairs.most_common()}

    results = {
        "confusion_pairs": confusion_pairs_dict,
        "confidence_stats": confidence_stats,
        "per_class_error_rate": per_class_error_rate,
        "discriminative_features": discriminative_features[:20],  # Top 20
        "total_errors": len(error_cases),
        "total_samples": len(all_targets),
        "overall_error_rate": float(len(error_cases) / len(all_targets)),
    }

    logger.info(f"Overall error rate: {results['overall_error_rate']:.4f}")
    logger.info(f"Top 3 confusion pairs: {list(confusion_pairs_dict.items())[:3]}")

    return results


def generate_error_visualizations(
    error_cases: list[ErrorCase],
    patterns: dict,
    output_dir: Path,
    class_filter: str | None = None,
    max_images: int = 50,
) -> dict[str, Path]:
    """Generate error analysis visualizations.

    Args:
        error_cases: List of error cases
        patterns: Error pattern analysis results
        output_dir: Directory to save plots
        class_filter: Filter errors for specific true class
        max_images: Maximum number of images to display in grid

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    # Filter errors if requested
    if class_filter:
        filtered_errors = [e for e in error_cases if e.true_label == class_filter]
        logger.info(f"Filtered to {len(filtered_errors)} errors for class {class_filter}")
    else:
        filtered_errors = error_cases

    # 1. Misclassified image grid
    if len(filtered_errors) > 0 and filtered_errors[0].image_array is not None:
        images_to_show = filtered_errors[:max_images]
        images = []
        titles = []

        for error in images_to_show:
            img_array = np.array(error.image_array)
            images.append(img_array)
            titles.append(f"True: {error.true_label}\nPred: {error.pred_label}\nConf: {error.confidence:.2f}")

        grid_name = f"misclassified_{class_filter}_grid.png" if class_filter else "all_errors_grid.png"
        grid_path = output_dir / grid_name
        save_image_grid(images, titles, grid_path, ncols=5)
        plots[grid_name.replace(".png", "")] = grid_path

    # 2. Confidence distribution
    confidences = [e.confidence for e in filtered_errors]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(
        np.mean(confidences), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(confidences):.3f}"
    )
    ax.set_xlabel("Confidence (Max Probability)")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution for Errors{' (' + class_filter + ')' if class_filter else ''}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    conf_path = output_dir / "confidence_distribution.png"
    plt.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["confidence_distribution"] = conf_path

    # 3. Confusion pairs bar chart
    confusion_pairs = patterns["confusion_pairs"]
    if len(confusion_pairs) > 0:
        top_pairs = list(confusion_pairs.items())[:10]
        pairs, counts = zip(*top_pairs, strict=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, counts, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Confusion Pairs (True -> Predicted)")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        pairs_path = output_dir / "confusion_pairs.png"
        plt.savefig(pairs_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["confusion_pairs"] = pairs_path

    # 4. Error rate heatmap (confusion matrix style)
    # Build error matrix
    n_classes = len(CLASSES)
    error_matrix = np.zeros((n_classes, n_classes))

    for error in error_cases:
        true_idx = CLASSES.index(error.true_label)
        pred_idx = CLASSES.index(error.pred_label)
        error_matrix[true_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        error_matrix,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        cbar_kws={"label": "Error Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Error Distribution Heatmap")

    plt.tight_layout()
    heatmap_path = output_dir / "error_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["error_heatmap"] = heatmap_path

    # 5. Per-class error rate bar chart
    per_class = patterns["per_class_error_rate"]
    classes = list(per_class.keys())
    error_rates = list(per_class.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(classes, error_rates, alpha=0.8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Error Rate")
    ax.set_title("Per-Class Error Rate")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (_cls, rate) in enumerate(zip(classes, error_rates, strict=True)):
        ax.text(i, rate + 0.02, f"{rate:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    rate_path = output_dir / "per_class_error_rate.png"
    plt.savefig(rate_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["per_class_error_rate"] = rate_path

    logger.info(f"Saved {len(plots)} error visualization plots to {output_dir}")
    return plots


def save_error_report(
    error_cases: list[ErrorCase],
    patterns: dict,
    output_dir: Path,
    class_filter: str | None = None,
) -> Path:
    """Save error analysis as text report.

    Args:
        error_cases: List of error cases
        patterns: Error patterns
        output_dir: Directory to save report
        class_filter: Class filter if applied

    Returns:
        Path to report file
    """
    report_path = output_dir / "error_summary.txt"

    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        if class_filter:
            f.write(f"Class Filter: {class_filter}\n")
        f.write("=" * 80 + "\n\n")

        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples:     {patterns['total_samples']}\n")
        f.write(f"Total Errors:      {patterns['total_errors']}\n")
        f.write(
            f"Overall Error Rate: {patterns['overall_error_rate']:.4f} ({patterns['overall_error_rate'] * 100:.2f}%)\n\n"
        )

        # Per-class error rates
        f.write("PER-CLASS ERROR RATES\n")
        f.write("-" * 40 + "\n")
        for class_name, rate in patterns["per_class_error_rate"].items():
            f.write(f"{class_name:25s}: {rate:.4f} ({rate * 100:.2f}%)\n")
        f.write("\n")

        # Top confusion pairs
        f.write("TOP CONFUSION PAIRS\n")
        f.write("-" * 40 + "\n")
        for pair, count in list(patterns["confusion_pairs"].items())[:10]:
            f.write(f"{pair:30s}: {count:3d}\n")
        f.write("\n")

        # Confidence stats
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-" * 40 + "\n")
        conf = patterns["confidence_stats"]
        f.write(f"Mean:   {conf['mean']:.4f}\n")
        f.write(f"Median: {conf['median']:.4f}\n")
        f.write(f"Std:    {conf['std']:.4f}\n")
        f.write(f"Min:    {conf['min']:.4f}\n")
        f.write(f"Max:    {conf['max']:.4f}\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Saved error report to {report_path}")
    return report_path


@app.command()
def main(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
    class_filter: str = typer.Option("", "--class-filter", "-c", help="Filter errors for specific class"),
    output_dir: str = typer.Option("reports/error_analysis", "--output-dir", "-o", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    max_images: int = typer.Option(50, "--max-images", help="Max images to show in grid"),
) -> None:
    """Analyze misclassified samples."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if class_filter and class_filter not in CLASSES:
        raise ValueError(f"Invalid class filter: {class_filter}. Must be one of {CLASSES}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, name="error_analysis", job_type="analysis")

    # Load feature metadata
    metadata = load_feature_metadata()
    feature_names = metadata["feature_names"]

    # Load model
    config_path = checkpoint_path.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create test dataloader
    logger.info("Creating test dataloader...")
    use_features = cfg.get("model", {}).get("name", "") in ["dual_pathway", "dualpathway", "hybrid"]
    _, _, test_loader = create_dataloaders(cfg, use_features=use_features)

    # Collect predictions and errors
    error_cases, all_preds, all_targets = collect_predictions(
        model, test_loader, device, use_features, collect_images=True
    )

    # Analyze patterns
    patterns = analyze_error_patterns(error_cases, all_preds, all_targets, feature_names)

    # Save error cases as JSON
    output_path.mkdir(parents=True, exist_ok=True)
    errors_path = output_path / "error_cases.json"

    # Convert error cases to dicts for JSON serialization
    error_dicts = []
    for error in error_cases:
        error_dict = asdict(error)
        # Remove large image arrays from JSON (keep only first few for inspection)
        if error_dict["image_array"] is not None:
            error_dict["image_array"] = None  # Too large for JSON
        error_dicts.append(error_dict)

    with errors_path.open("w") as f:
        json.dump(error_dicts, f, indent=2)
    logger.info(f"Saved error cases to {errors_path}")

    # Save patterns
    patterns_path = output_path / "error_report.json"
    with patterns_path.open("w") as f:
        json.dump(patterns, f, indent=2)
    logger.info(f"Saved error patterns to {patterns_path}")

    # Generate visualizations
    plots = generate_error_visualizations(error_cases, patterns, output_path, class_filter, max_images)

    # Generate text report
    report_path = save_error_report(error_cases, patterns, output_path, class_filter)

    # Log to W&B
    if use_wandb:
        log_metrics = {
            "total_errors": patterns["total_errors"],
            "overall_error_rate": patterns["overall_error_rate"],
            "mean_confidence": patterns["confidence_stats"]["mean"],
        }

        # Log per-class error rates
        for class_name, rate in patterns["per_class_error_rate"].items():
            log_metrics[f"error_rate_{class_name}"] = rate

        log_to_wandb(log_metrics, plots, "error_analysis")
        wandb.save(str(report_path))
        wandb.finish()

    logger.info(f"\nError analysis complete. Results saved to {output_path}")
    logger.info(
        f"Total errors: {patterns['total_errors']}/{patterns['total_samples']} ({patterns['overall_error_rate'] * 100:.2f}%)"
    )


if __name__ == "__main__":
    app()
