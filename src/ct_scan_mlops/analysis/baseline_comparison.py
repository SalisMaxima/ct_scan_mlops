"""Compare baseline CustomCNN vs improved DualPathway model performance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
import wandb
from loguru import logger
from omegaconf import OmegaConf
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ct_scan_mlops.analysis.utils import compute_calibration_error, log_to_wandb
from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.evaluate import load_model_from_checkpoint
from ct_scan_mlops.utils import get_device

app = typer.Typer()


def load_both_models(
    baseline_checkpoint: Path,
    improved_checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """Load baseline and improved models from checkpoints.

    Uses load_model_from_checkpoint() with auto-detected configs
    from .hydra/config.yaml in each checkpoint directory.

    Args:
        baseline_checkpoint: Path to baseline model checkpoint
        improved_checkpoint: Path to improved model checkpoint
        device: Device to load models on

    Returns:
        Tuple of (baseline_model, improved_model)
    """
    # Load configs from checkpoint directories
    baseline_dir = baseline_checkpoint.parent
    improved_dir = improved_checkpoint.parent

    # Try to find .hydra/config.yaml in checkpoint directory
    baseline_config_path = baseline_dir / ".hydra" / "config.yaml"
    improved_config_path = improved_dir / ".hydra" / "config.yaml"

    if not baseline_config_path.exists():
        raise FileNotFoundError(f"Baseline config not found at {baseline_config_path}")
    if not improved_config_path.exists():
        raise FileNotFoundError(f"Improved config not found at {improved_config_path}")

    baseline_cfg = OmegaConf.load(baseline_config_path)
    improved_cfg = OmegaConf.load(improved_config_path)

    logger.info(f"Loading baseline model from {baseline_checkpoint}")
    baseline_model = load_model_from_checkpoint(baseline_checkpoint, baseline_cfg, device)

    logger.info(f"Loading improved model from {improved_checkpoint}")
    improved_model = load_model_from_checkpoint(improved_checkpoint, improved_cfg, device)

    return baseline_model, improved_model


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    """Perform McNemar's test for paired model comparison.

    Tests if the two models have significantly different error rates.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2

    Returns:
        Dictionary with test statistic and p-value
    """
    # Create contingency table
    # correct1_correct2, correct1_wrong2, wrong1_correct2, wrong1_wrong2
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true

    n_correct_correct = np.sum(correct1 & correct2)
    n_correct_wrong = np.sum(correct1 & ~correct2)
    n_wrong_correct = np.sum(~correct1 & correct2)
    n_wrong_wrong = np.sum(~correct1 & ~correct2)

    # Contingency table
    table = np.array([[n_correct_correct, n_correct_wrong], [n_wrong_correct, n_wrong_wrong]])

    # McNemar's test statistic (with continuity correction)
    if n_correct_wrong + n_wrong_correct > 0:
        statistic = (abs(n_correct_wrong - n_wrong_correct) - 1) ** 2 / (n_correct_wrong + n_wrong_correct)
        # Chi-square test with 1 degree of freedom
        p_value = 1 - chi2_contingency([[n_correct_wrong], [n_wrong_correct]])[1]
    else:
        statistic = 0.0
        p_value = 1.0

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "contingency_table": table.tolist(),
        "n_correct_wrong": int(n_correct_wrong),
        "n_wrong_correct": int(n_wrong_correct),
    }


def collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect predictions and probabilities.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        use_features: Whether the model uses radiomics features

    Returns:
        Tuple of (predictions, targets, probabilities)
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
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

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
    )


def compare_models(
    baseline_model: nn.Module,
    improved_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    baseline_uses_features: bool = False,
    improved_uses_features: bool = True,
) -> dict:
    """Run inference on both models and compute comparative metrics.

    Args:
        baseline_model: Baseline model
        improved_model: Improved model
        test_loader: Test data loader
        device: Device to run on
        baseline_uses_features: Whether baseline model uses features
        improved_uses_features: Whether improved model uses features

    Returns:
        Dictionary with comparative metrics and statistical tests
    """
    logger.info("Evaluating baseline model...")
    baseline_preds, targets, baseline_probs = collect_predictions(
        baseline_model, test_loader, device, baseline_uses_features
    )

    logger.info("Evaluating improved model...")
    improved_preds, _, improved_probs = collect_predictions(improved_model, test_loader, device, improved_uses_features)

    # Compute metrics for both models
    baseline_metrics = compute_model_metrics(targets, baseline_preds, baseline_probs, "baseline")
    improved_metrics = compute_model_metrics(targets, improved_preds, improved_probs, "improved")

    # Compute delta (improvement)
    delta = {
        "accuracy_delta": improved_metrics["accuracy"] - baseline_metrics["accuracy"],
        "macro_f1_delta": improved_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "weighted_f1_delta": improved_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
    }

    # Per-class deltas
    for class_name in CLASSES:
        for metric in ["precision", "recall", "f1"]:
            key = f"{class_name}_{metric}"
            delta[f"{key}_delta"] = improved_metrics[key] - baseline_metrics[key]

    # Statistical tests
    mcnemar = mcnemar_test(targets, baseline_preds, improved_preds)
    kappa = cohen_kappa_score(baseline_preds, improved_preds)

    # Confusion matrices
    baseline_cm = confusion_matrix(targets, baseline_preds)
    improved_cm = confusion_matrix(targets, improved_preds)

    return {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "delta": delta,
        "statistical_tests": {
            "mcnemar": mcnemar,
            "cohen_kappa": float(kappa),
        },
        "confusion_matrices": {
            "baseline": baseline_cm.tolist(),
            "improved": improved_cm.tolist(),
        },
    }


def compute_model_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    prefix: str = "",
) -> dict:
    """Compute comprehensive metrics for a single model.

    Args:
        targets: True labels
        predictions: Predicted labels
        probabilities: Predicted probabilities (N, num_classes)
        prefix: Prefix for metric keys

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Overall accuracy
    accuracy = np.mean(predictions == targets)
    metrics["accuracy"] = float(accuracy)

    # Classification report for per-class metrics
    report = classification_report(targets, predictions, target_names=CLASSES, output_dict=True, zero_division=0)

    # Per-class metrics
    for class_name in CLASSES:
        if class_name in report:
            metrics[f"{class_name}_precision"] = report[class_name]["precision"]
            metrics[f"{class_name}_recall"] = report[class_name]["recall"]
            metrics[f"{class_name}_f1"] = report[class_name]["f1-score"]

    # Overall metrics
    metrics["macro_f1"] = report["macro avg"]["f1-score"]
    metrics["weighted_f1"] = report["weighted avg"]["f1-score"]

    # ROC-AUC per class (one-vs-rest)
    try:
        roc_auc_per_class = {}
        for i, class_name in enumerate(CLASSES):
            # One-vs-rest: class i vs all others
            binary_targets = (targets == i).astype(int)
            class_probs = probabilities[:, i]
            auc = roc_auc_score(binary_targets, class_probs)
            roc_auc_per_class[class_name] = float(auc)
            metrics[f"{class_name}_auc"] = float(auc)

        metrics["macro_auc"] = float(np.mean(list(roc_auc_per_class.values())))
    except ValueError as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")

    # Calibration error
    ece = compute_calibration_error(probabilities, predictions, targets)
    metrics["ece"] = float(ece)

    return metrics


def generate_comparison_plots(
    metrics: dict,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate side-by-side visualizations.

    Args:
        metrics: Dictionary with baseline, improved, and delta metrics
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    baseline_cm = np.array(metrics["confusion_matrices"]["baseline"])
    improved_cm = np.array(metrics["confusion_matrices"]["improved"])

    # 1. Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        baseline_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Baseline (Acc: {metrics['baseline']['accuracy']:.4f})")

    sns.heatmap(
        improved_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Improved (Acc: {metrics['improved']['accuracy']:.4f})")

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["confusion_matrices"] = cm_path

    # 2. Per-class metrics comparison
    metric_names = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, metric_name in enumerate(metric_names):
        baseline_vals = [metrics["baseline"][f"{c}_{metric_name}"] for c in CLASSES]
        improved_vals = [metrics["improved"][f"{c}_{metric_name}"] for c in CLASSES]

        x = np.arange(len(CLASSES))
        width = 0.35

        axes[idx].bar(x - width / 2, baseline_vals, width, label="Baseline", alpha=0.8)
        axes[idx].bar(x + width / 2, improved_vals, width, label="Improved", alpha=0.8)

        axes[idx].set_xlabel("Class")
        axes[idx].set_ylabel(metric_name.capitalize())
        axes[idx].set_title(f"Per-class {metric_name.capitalize()}")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(CLASSES, rotation=45, ha="right")
        axes[idx].legend()
        axes[idx].set_ylim([0, 1.05])
        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    metrics_path = output_dir / "per_class_metrics.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["per_class_metrics"] = metrics_path

    # 3. Improvement summary
    fig, ax = plt.subplots(figsize=(10, 6))

    improvements = {
        "Overall Accuracy": metrics["delta"]["accuracy_delta"],
        "Macro F1": metrics["delta"]["macro_f1_delta"],
        "Weighted F1": metrics["delta"]["weighted_f1_delta"],
    }

    # Add per-class F1 improvements
    for class_name in CLASSES:
        improvements[f"{class_name}\nF1"] = metrics["delta"][f"{class_name}_f1_delta"]

    x = np.arange(len(improvements))
    colors = ["green" if v > 0 else "red" for v in improvements.values()]

    bars = ax.bar(x, list(improvements.values()), color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(improvements.keys()), rotation=45, ha="right")
    ax.set_ylabel("Improvement (Improved - Baseline)")
    ax.set_title("Model Improvement Summary")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom" if height > 0 else "top",
        )

    plt.tight_layout()
    improvement_path = output_dir / "improvement_summary.png"
    plt.savefig(improvement_path, dpi=150, bbox_inches="tight")
    plt.close()
    plots["improvement_summary"] = improvement_path

    logger.info(f"Saved {len(plots)} comparison plots to {output_dir}")
    return plots


def generate_text_report(metrics: dict, output_dir: Path) -> Path:
    """Generate a text summary report.

    Args:
        metrics: Dictionary with all metrics
        output_dir: Directory to save report

    Returns:
        Path to the report file
    """
    report_path = output_dir / "comparison_report.txt"

    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline Accuracy:  {metrics['baseline']['accuracy']:.4f}\n")
        f.write(f"Improved Accuracy:  {metrics['improved']['accuracy']:.4f}\n")
        f.write(
            f"Improvement:        {metrics['delta']['accuracy_delta']:+.4f} ({metrics['delta']['accuracy_delta'] * 100:+.2f}%)\n\n"
        )

        f.write(f"Baseline Macro F1:  {metrics['baseline']['macro_f1']:.4f}\n")
        f.write(f"Improved Macro F1:  {metrics['improved']['macro_f1']:.4f}\n")
        f.write(f"Improvement:        {metrics['delta']['macro_f1_delta']:+.4f}\n\n")

        # Statistical significance
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 40 + "\n")
        mcnemar = metrics["statistical_tests"]["mcnemar"]
        f.write("McNemar's Test:\n")
        f.write(f"  Statistic: {mcnemar['statistic']:.4f}\n")
        f.write(f"  p-value:   {mcnemar['p_value']:.6f}\n")
        if mcnemar["p_value"] < 0.001:
            f.write("  Result:    Highly significant (p < 0.001) ***\n")
        elif mcnemar["p_value"] < 0.01:
            f.write("  Result:    Very significant (p < 0.01) **\n")
        elif mcnemar["p_value"] < 0.05:
            f.write("  Result:    Significant (p < 0.05) *\n")
        else:
            f.write("  Result:    Not significant (p >= 0.05)\n")
        f.write(f"\nCohen's Kappa (model agreement): {metrics['statistical_tests']['cohen_kappa']:.4f}\n\n")

        # Per-class breakdown
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 40 + "\n")
        for class_name in CLASSES:
            f.write(f"\n{class_name.upper()}:\n")
            for metric in ["precision", "recall", "f1"]:
                key = f"{class_name}_{metric}"
                baseline_val = metrics["baseline"][key]
                improved_val = metrics["improved"][key]
                delta = metrics["delta"][f"{key}_delta"]
                f.write(f"  {metric.capitalize():10s}: {baseline_val:.4f} -> {improved_val:.4f} ({delta:+.4f})\n")

        # Calibration
        f.write("\nCALIBRATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline ECE: {metrics['baseline']['ece']:.4f}\n")
        f.write(f"Improved ECE: {metrics['improved']['ece']:.4f}\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Saved text report to {report_path}")
    return report_path


@app.command()
def main(
    baseline: str = typer.Argument(..., help="Path to baseline model checkpoint"),
    improved: str = typer.Argument(..., help="Path to improved model checkpoint"),
    output_dir: str = typer.Option("reports/baseline_comparison", "--output-dir", "-o", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    baseline_uses_features: bool = typer.Option(False, "--baseline-features", help="Baseline uses features"),
    improved_uses_features: bool = typer.Option(True, "--improved-features", help="Improved uses features"),
) -> None:
    """Compare baseline CNN vs improved DualPathway model."""
    baseline_path = Path(baseline)
    improved_path = Path(improved)
    output_path = Path(output_dir)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
    if not improved_path.exists():
        raise FileNotFoundError(f"Improved checkpoint not found: {improved_path}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, name="baseline_comparison", job_type="analysis")

    # Load models
    baseline_model, improved_model = load_both_models(baseline_path, improved_path, device)

    # Load improved model config to get data settings
    improved_config_path = improved_path.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(improved_config_path)

    # Create test dataloader with features
    logger.info("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(cfg, use_features=improved_uses_features)

    # Compare models
    logger.info("Comparing models...")
    metrics = compare_models(
        baseline_model,
        improved_model,
        test_loader,
        device,
        baseline_uses_features,
        improved_uses_features,
    )

    # Save metrics
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "comparison_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Generate plots
    plots = generate_comparison_plots(metrics, output_path)

    # Generate text report
    report_path = generate_text_report(metrics, output_path)

    # Log to W&B
    if use_wandb:
        log_to_wandb(
            {
                "baseline_accuracy": metrics["baseline"]["accuracy"],
                "improved_accuracy": metrics["improved"]["accuracy"],
                "accuracy_improvement": metrics["delta"]["accuracy_delta"],
                "mcnemar_p_value": metrics["statistical_tests"]["mcnemar"]["p_value"],
            },
            plots,
            "baseline_comparison",
        )
        # Log the text report
        wandb.save(str(report_path))
        wandb.finish()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Baseline accuracy: {metrics['baseline']['accuracy']:.4f}")
    logger.info(f"Improved accuracy: {metrics['improved']['accuracy']:.4f}")
    logger.info(
        f"Improvement:       {metrics['delta']['accuracy_delta']:+.4f} ({metrics['delta']['accuracy_delta'] * 100:+.2f}%)"
    )
    logger.info(f"McNemar p-value:   {metrics['statistical_tests']['mcnemar']['p_value']:.6f}")
    logger.info("=" * 80 + "\n")
    logger.info(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    app()
