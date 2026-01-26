"""Deep analysis of adenocarcinoma-squamous confusion patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import wandb
from loguru import logger
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_scan_mlops.analysis.utils import load_feature_metadata, log_to_wandb, save_image_grid
from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.evaluate import load_model_from_checkpoint
from ct_scan_mlops.utils import get_device

app = typer.Typer()

# Target confusion pair indices
ADENO_IDX = CLASSES.index("adenocarcinoma")
SQUAMOUS_IDX = CLASSES.index("squamous_cell_carcinoma")


class FeatureDiff(TypedDict):
    """Type definition for feature difference statistics."""

    feature: str
    confused_mean: float
    confused_std: float
    correct_mean: float
    correct_std: float
    cohens_d: float
    abs_cohens_d: float


@dataclass
class ConfusedSample:
    """Represents a sample involved in adenocarcinoma-squamous confusion."""

    sample_idx: int
    true_label: str
    pred_label: str
    true_idx: int
    pred_idx: int
    confidence: float
    logits: np.ndarray
    probabilities: np.ndarray
    logit_margin: float  # logit[pred] - logit[true]
    features: np.ndarray | None
    image_array: np.ndarray | None


def collect_confusion_samples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_features: bool = True,
) -> tuple[list[ConfusedSample], list[dict], np.ndarray, np.ndarray]:
    """Collect samples involved in adenocarcinoma-squamous confusion.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        use_features: Whether model uses features

    Returns:
        Tuple of (confused_samples, correct_samples_data, all_preds, all_targets)
    """
    model.eval()
    confused_samples: list[ConfusedSample] = []
    correct_samples: list[dict] = []
    all_preds = []
    all_targets = []

    sample_idx = 0

    logger.info("Collecting confusion samples...")
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

            for i in range(len(targets)):
                pred_class = preds[i].item()
                true_class = targets[i].item()
                logits_np = outputs[i].cpu().numpy()
                probs_np = probs[i].cpu().numpy()

                all_preds.append(pred_class)
                all_targets.append(true_class)

                # Check for adenocarcinoma-squamous confusion
                is_adeno_squamous_confusion = (true_class == ADENO_IDX and pred_class == SQUAMOUS_IDX) or (
                    true_class == SQUAMOUS_IDX and pred_class == ADENO_IDX
                )

                if is_adeno_squamous_confusion:
                    logit_margin = logits_np[pred_class] - logits_np[true_class]
                    confused_sample = ConfusedSample(
                        sample_idx=sample_idx,
                        true_label=CLASSES[true_class],
                        pred_label=CLASSES[pred_class],
                        true_idx=true_class,
                        pred_idx=pred_class,
                        confidence=float(probs_np[pred_class]),
                        logits=logits_np,
                        probabilities=probs_np,
                        logit_margin=logit_margin,
                        features=features[i].cpu().numpy() if features is not None else None,
                        image_array=images[i].cpu().numpy(),
                    )
                    confused_samples.append(confused_sample)
                elif true_class in (ADENO_IDX, SQUAMOUS_IDX) and pred_class == true_class:
                    # Correctly classified adenocarcinoma or squamous
                    correct_samples.append(
                        {
                            "sample_idx": sample_idx,
                            "true_label": CLASSES[true_class],
                            "true_idx": true_class,
                            "confidence": float(probs_np[pred_class]),
                            "logits": logits_np,
                            "features": features[i].cpu().numpy() if features is not None else None,
                        }
                    )

                sample_idx += 1

    logger.info(
        f"Found {len(confused_samples)} adenocarcinoma-squamous confusion samples, "
        f"{len(correct_samples)} correctly classified samples"
    )
    return confused_samples, correct_samples, np.array(all_preds), np.array(all_targets)


def analyze_logit_margins(confused_samples: list[ConfusedSample]) -> dict:
    """Categorize errors by logit margin (borderline vs confident).

    Args:
        confused_samples: List of confused samples

    Returns:
        Dictionary with margin analysis
    """
    margins = [s.logit_margin for s in confused_samples]
    margins_np = np.array(margins)

    # Categorize: borderline (margin < 0.5), confident (margin > 1.5)
    borderline = [s for s in confused_samples if s.logit_margin < 0.5]
    medium = [s for s in confused_samples if 0.5 <= s.logit_margin <= 1.5]
    confident = [s for s in confused_samples if s.logit_margin > 1.5]

    # Split by direction
    adeno_to_squamous = [s for s in confused_samples if s.true_idx == ADENO_IDX]
    squamous_to_adeno = [s for s in confused_samples if s.true_idx == SQUAMOUS_IDX]

    return {
        "total_confused": len(confused_samples),
        "borderline_count": len(borderline),
        "medium_count": len(medium),
        "confident_count": len(confident),
        "borderline_pct": len(borderline) / len(confused_samples) * 100 if confused_samples else 0,
        "medium_pct": len(medium) / len(confused_samples) * 100 if confused_samples else 0,
        "confident_pct": len(confident) / len(confused_samples) * 100 if confused_samples else 0,
        "margin_mean": float(np.mean(margins_np)) if len(margins_np) > 0 else 0,
        "margin_std": float(np.std(margins_np)) if len(margins_np) > 0 else 0,
        "margin_min": float(np.min(margins_np)) if len(margins_np) > 0 else 0,
        "margin_max": float(np.max(margins_np)) if len(margins_np) > 0 else 0,
        "adeno_to_squamous_count": len(adeno_to_squamous),
        "squamous_to_adeno_count": len(squamous_to_adeno),
        "borderline_samples": [s.sample_idx for s in borderline],
        "confident_samples": [s.sample_idx for s in confident],
    }


def analyze_feature_differences(
    confused_samples: list[ConfusedSample],
    correct_samples: list[dict],
    feature_names: list[str],
) -> dict:
    """Compare feature distributions between confused and correct samples.

    Args:
        confused_samples: List of confused samples
        correct_samples: List of correctly classified samples
        feature_names: Names of radiomics features

    Returns:
        Dictionary with feature comparison
    """
    if not confused_samples or confused_samples[0].features is None:
        return {"error": "No features available"}

    if not correct_samples or correct_samples[0]["features"] is None:
        return {"error": "No correct samples with features"}

    confused_features = np.array([s.features for s in confused_samples])
    correct_features = np.array([s["features"] for s in correct_samples])

    feature_diffs: list[FeatureDiff] = []
    for i, name in enumerate(feature_names):
        if i >= confused_features.shape[1]:
            break

        confused_mean = np.mean(confused_features[:, i])
        confused_std = np.std(confused_features[:, i])
        correct_mean = np.mean(correct_features[:, i])
        correct_std = np.std(correct_features[:, i])

        # Cohen's d effect size
        pooled_std = np.sqrt((confused_std**2 + correct_std**2) / 2)
        cohens_d = (confused_mean - correct_mean) / pooled_std if pooled_std > 1e-8 else 0

        feature_diffs.append(
            FeatureDiff(
                feature=name,
                confused_mean=float(confused_mean),
                confused_std=float(confused_std),
                correct_mean=float(correct_mean),
                correct_std=float(correct_std),
                cohens_d=float(cohens_d),
                abs_cohens_d=abs(float(cohens_d)),
            )
        )

    # Sort by absolute effect size
    feature_diffs.sort(key=lambda x: x["abs_cohens_d"], reverse=True)

    return {
        "top_discriminative_features": feature_diffs[:15],
        "n_confused": len(confused_samples),
        "n_correct": len(correct_samples),
    }


def generate_tsne_visualization(
    confused_samples: list[ConfusedSample],
    correct_samples: list[dict],
    output_dir: Path,
) -> Path | None:
    """Generate t-SNE visualization of confused vs correct samples.

    Args:
        confused_samples: List of confused samples
        correct_samples: List of correctly classified samples
        output_dir: Directory to save plot

    Returns:
        Path to saved plot or None
    """
    if not confused_samples or confused_samples[0].features is None:
        logger.warning("No features available for t-SNE")
        return None

    # Combine features
    confused_features = np.array([s.features for s in confused_samples])
    correct_features = np.array([s["features"] for s in correct_samples])
    all_features = np.vstack([confused_features, correct_features])

    # Labels for plotting
    labels = ["Confused"] * len(confused_samples) + ["Correct"] * len(correct_samples)

    # True class labels
    true_classes = [s.true_label for s in confused_samples] + [s["true_label"] for s in correct_samples]

    logger.info(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
    embeddings = tsne.fit_transform(all_features)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Confused vs Correct
    colors = ["red" if label == "Confused" else "green" for label in labels]
    axes[0].scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.6, s=50)
    axes[0].set_title("t-SNE: Confused (red) vs Correct (green)")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Confused"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="Correct"),
    ]
    axes[0].legend(handles=legend_elements, loc="best")

    # Plot 2: By true class
    class_colors = {"adenocarcinoma": "blue", "squamous_cell_carcinoma": "orange"}
    colors2 = [class_colors.get(c, "gray") for c in true_classes]
    axes[1].scatter(embeddings[:, 0], embeddings[:, 1], c=colors2, alpha=0.6, s=50)
    axes[1].set_title("t-SNE: Adenocarcinoma (blue) vs Squamous (orange)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    legend_elements2 = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Adenocarcinoma"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Squamous"),
    ]
    axes[1].legend(handles=legend_elements2, loc="best")

    plt.tight_layout()
    output_path = output_dir / "tsne_confusion_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved t-SNE plot to {output_path}")
    return output_path


def generate_confusion_visualizations(
    confused_samples: list[ConfusedSample],
    margin_analysis: dict,
    output_dir: Path,
    max_images: int = 20,
) -> dict[str, Path]:
    """Generate visualizations for confusion analysis.

    Args:
        confused_samples: List of confused samples
        margin_analysis: Margin analysis results
        output_dir: Directory to save plots
        max_images: Maximum images per grid

    Returns:
        Dictionary of plot names to paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    # 1. Logit margin histogram
    margins = [s.logit_margin for s in confused_samples]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(margins, bins=15, alpha=0.7, edgecolor="black", color="coral")
    ax.axvline(0.5, color="green", linestyle="--", linewidth=2, label="Borderline threshold (0.5)")
    ax.axvline(1.5, color="red", linestyle="--", linewidth=2, label="Confident threshold (1.5)")
    ax.axvline(np.mean(margins), color="blue", linestyle="-", linewidth=2, label=f"Mean: {np.mean(margins):.2f}")
    ax.set_xlabel("Logit Margin (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Logit Margin Distribution for Adenocarcinoma-Squamous Confusion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    margin_path = output_dir / "logit_margin_distribution.png"
    plt.savefig(margin_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["logit_margin_distribution"] = margin_path

    # 2. Confidence distribution
    confidences = [s.confidence for s in confused_samples]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=15, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(np.mean(confidences), color="red", linestyle="-", linewidth=2, label=f"Mean: {np.mean(confidences):.2f}")
    ax.set_xlabel("Confidence (Max Probability)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution for Confused Samples")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    conf_path = output_dir / "confidence_distribution_confusion.png"
    plt.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["confidence_distribution_confusion"] = conf_path

    # 3. Confusion direction pie chart
    adeno_to_sq = margin_analysis["adeno_to_squamous_count"]
    sq_to_adeno = margin_analysis["squamous_to_adeno_count"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        [adeno_to_sq, sq_to_adeno],
        labels=["Adeno -> Squamous", "Squamous -> Adeno"],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff"],
        explode=(0.05, 0.05),
    )
    ax.set_title("Confusion Direction Distribution")

    direction_path = output_dir / "confusion_direction.png"
    plt.savefig(direction_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["confusion_direction"] = direction_path

    # 4. Error type breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Borderline\n(margin < 0.5)", "Medium\n(0.5 - 1.5)", "Confident\n(margin > 1.5)"]
    counts = [margin_analysis["borderline_count"], margin_analysis["medium_count"], margin_analysis["confident_count"]]
    colors = ["#90EE90", "#FFD700", "#FF6347"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Error Categories by Logit Margin")

    for bar, count in zip(bars, counts, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", fontweight="bold")

    category_path = output_dir / "error_categories.png"
    plt.savefig(category_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["error_categories"] = category_path

    # 5. Image grids for adenocarcinoma -> squamous confusion
    adeno_to_sq_samples = [s for s in confused_samples if s.true_idx == ADENO_IDX][:max_images]
    if adeno_to_sq_samples:
        images = [s.image_array for s in adeno_to_sq_samples]
        titles = [f"Conf: {s.confidence:.2f}\nMargin: {s.logit_margin:.2f}" for s in adeno_to_sq_samples]
        grid_path = output_dir / "adeno_to_squamous_grid.png"
        save_image_grid(images, titles, grid_path, ncols=5)
        plots["adeno_to_squamous_grid"] = grid_path

    # 6. Image grids for squamous -> adenocarcinoma confusion
    sq_to_adeno_samples = [s for s in confused_samples if s.true_idx == SQUAMOUS_IDX][:max_images]
    if sq_to_adeno_samples:
        images = [s.image_array for s in sq_to_adeno_samples]
        titles = [f"Conf: {s.confidence:.2f}\nMargin: {s.logit_margin:.2f}" for s in sq_to_adeno_samples]
        grid_path = output_dir / "squamous_to_adeno_grid.png"
        save_image_grid(images, titles, grid_path, ncols=5)
        plots["squamous_to_adeno_grid"] = grid_path

    logger.info(f"Saved {len(plots)} confusion visualizations to {output_dir}")
    return plots


def generate_feature_diff_plot(feature_analysis: dict, output_dir: Path) -> Path | None:
    """Generate plot showing feature differences between confused and correct samples.

    Args:
        feature_analysis: Feature analysis results
        output_dir: Directory to save plot

    Returns:
        Path to saved plot or None
    """
    if "error" in feature_analysis:
        return None

    top_features = feature_analysis["top_discriminative_features"][:10]

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [f["feature"][:30] for f in top_features]  # Truncate long names
    cohens_d = [f["cohens_d"] for f in top_features]
    colors = ["red" if d > 0 else "blue" for d in cohens_d]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, cohens_d, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (positive = higher in confused samples)")
    ax.set_title("Top Features Distinguishing Confused vs Correct Samples")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "feature_differences.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature differences plot to {output_path}")
    return output_path


def save_confusion_report(
    margin_analysis: dict,
    feature_analysis: dict,
    output_dir: Path,
) -> Path:
    """Save comprehensive confusion analysis report.

    Args:
        margin_analysis: Margin analysis results
        feature_analysis: Feature analysis results
        output_dir: Directory to save report

    Returns:
        Path to report file
    """
    report_path = output_dir / "confusion_analysis_report.txt"

    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("ADENOCARCINOMA-SQUAMOUS CONFUSION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total confused samples:      {margin_analysis['total_confused']}\n")
        f.write(f"Adeno -> Squamous:           {margin_analysis['adeno_to_squamous_count']}\n")
        f.write(f"Squamous -> Adeno:           {margin_analysis['squamous_to_adeno_count']}\n\n")

        f.write("ERROR CATEGORIES\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Borderline (margin < 0.5):   {margin_analysis['borderline_count']} ({margin_analysis['borderline_pct']:.1f}%)\n"
        )
        f.write(
            f"Medium (0.5 - 1.5):          {margin_analysis['medium_count']} ({margin_analysis['medium_pct']:.1f}%)\n"
        )
        f.write(
            f"Confident (margin > 1.5):    {margin_analysis['confident_count']} ({margin_analysis['confident_pct']:.1f}%)\n\n"
        )

        f.write("LOGIT MARGIN STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean:   {margin_analysis['margin_mean']:.4f}\n")
        f.write(f"Std:    {margin_analysis['margin_std']:.4f}\n")
        f.write(f"Min:    {margin_analysis['margin_min']:.4f}\n")
        f.write(f"Max:    {margin_analysis['margin_max']:.4f}\n\n")

        if "top_discriminative_features" in feature_analysis:
            f.write("TOP DISCRIMINATIVE FEATURES\n")
            f.write("-" * 40 + "\n")
            f.write("(Positive Cohen's d = higher in confused samples)\n\n")
            for feat in feature_analysis["top_discriminative_features"][:10]:
                f.write(f"{feat['feature'][:40]:40s}: d = {feat['cohens_d']:+.3f}\n")
            f.write("\n")

        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        if margin_analysis["confident_pct"] > 30:
            f.write("- Many confident errors: Consider focal loss to focus on hard examples\n")
        if margin_analysis["borderline_pct"] > 50:
            f.write("- Many borderline cases: Classes may be inherently similar\n")
        if margin_analysis["adeno_to_squamous_count"] > 2 * margin_analysis["squamous_to_adeno_count"]:
            f.write("- Asymmetric confusion: Increase adenocarcinoma training weight\n")
        elif margin_analysis["squamous_to_adeno_count"] > 2 * margin_analysis["adeno_to_squamous_count"]:
            f.write("- Asymmetric confusion: Increase squamous training weight\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Saved confusion report to {report_path}")
    return report_path


@app.command()
def main(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
    output_dir: str = typer.Option("reports/confusion_analysis", "--output-dir", "-o", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    max_images: int = typer.Option(20, "--max-images", help="Max images per grid"),
) -> None:
    """Analyze adenocarcinoma-squamous confusion patterns."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, name="confusion_analysis", job_type="analysis")

    # Load feature metadata
    try:
        metadata = load_feature_metadata()
        feature_names = metadata["feature_names"]
    except FileNotFoundError:
        logger.warning("Feature metadata not found, proceeding without feature analysis")
        feature_names = []

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

    # Collect confusion samples
    confused_samples, correct_samples, all_preds, all_targets = collect_confusion_samples(
        model, test_loader, device, use_features
    )

    if not confused_samples:
        logger.warning("No adenocarcinoma-squamous confusion found!")
        return

    # Analyze logit margins
    margin_analysis = analyze_logit_margins(confused_samples)

    # Analyze feature differences
    feature_analysis = analyze_feature_differences(confused_samples, correct_samples, feature_names)

    # Save analysis results
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "confusion_analysis.json"
    results = {
        "margin_analysis": margin_analysis,
        "feature_analysis": feature_analysis,
    }
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved analysis results to {results_path}")

    # Generate visualizations
    plots = generate_confusion_visualizations(confused_samples, margin_analysis, output_path, max_images)

    # Generate t-SNE
    if use_features and correct_samples:
        tsne_path = generate_tsne_visualization(confused_samples, correct_samples, output_path)
        if tsne_path:
            plots["tsne_confusion"] = tsne_path

    # Generate feature difference plot
    feat_plot = generate_feature_diff_plot(feature_analysis, output_path)
    if feat_plot:
        plots["feature_differences"] = feat_plot

    # Save text report
    report_path = save_confusion_report(margin_analysis, feature_analysis, output_path)

    # Log to W&B
    if use_wandb:
        log_metrics = {
            "confusion/total_confused": margin_analysis["total_confused"],
            "confusion/borderline_pct": margin_analysis["borderline_pct"],
            "confusion/confident_pct": margin_analysis["confident_pct"],
            "confusion/margin_mean": margin_analysis["margin_mean"],
            "confusion/adeno_to_squamous": margin_analysis["adeno_to_squamous_count"],
            "confusion/squamous_to_adeno": margin_analysis["squamous_to_adeno_count"],
        }
        log_to_wandb(log_metrics, plots, "confusion_analysis")
        wandb.save(str(report_path))
        wandb.finish()

    logger.info(f"\nConfusion analysis complete. Results saved to {output_path}")
    logger.info(f"Total confused: {margin_analysis['total_confused']}")
    logger.info(f"Borderline: {margin_analysis['borderline_count']} ({margin_analysis['borderline_pct']:.1f}%)")
    logger.info(f"Confident: {margin_analysis['confident_count']} ({margin_analysis['confident_pct']:.1f}%)")


if __name__ == "__main__":
    app()
