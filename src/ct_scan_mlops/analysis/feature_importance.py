"""Analyze radiomics feature importance for DualPathway model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import wandb
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_scan_mlops.analysis.utils import load_feature_metadata, log_to_wandb
from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.evaluate import load_model_from_checkpoint
from ct_scan_mlops.utils import get_device

app = typer.Typer()


def compute_permutation_importance(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_repeats: int = 10,
    feature_names: list[str] | None = None,
) -> dict:
    """Compute permutation importance for each radiomics feature.

    Algorithm:
    1. Baseline accuracy on test set
    2. For each feature i in [0..49]:
        - Shuffle feature i across batch
        - Measure accuracy drop
        - Repeat n_repeats times, average
    3. Rank features by mean importance

    Args:
        model: Trained DualPathway model
        test_loader: Test data loader (must return features)
        device: Device to run on
        n_repeats: Number of shuffle repeats for variance estimation
        feature_names: List of feature names (50 elements)

    Returns:
        Dictionary with importance scores and feature names
    """
    model.eval()

    # 1. Compute baseline accuracy
    logger.info("Computing baseline accuracy...")
    all_preds = []
    all_targets = []
    all_features_list = []

    with torch.no_grad():
        for batch in test_loader:
            images, features, targets = batch
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(images, features)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_features_list.append(features.cpu())

    baseline_accuracy = accuracy_score(all_targets, all_preds)
    logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # Concatenate all features
    all_features = torch.cat(all_features_list, dim=0)  # (N, 50)
    all_targets = np.array(all_targets)

    # 2. For each feature, compute importance via permutation
    n_features = all_features.shape[1]
    importances = np.zeros((n_features, n_repeats))

    logger.info(f"Computing permutation importance for {n_features} features...")
    for feature_idx in tqdm(range(n_features), desc="Features"):
        for repeat in range(n_repeats):
            # Create a copy and shuffle this feature
            features_permuted = all_features.clone()
            perm_idx = torch.randperm(features_permuted.shape[0])
            features_permuted[:, feature_idx] = features_permuted[perm_idx, feature_idx]

            # Evaluate with permuted features
            preds_permuted = []
            batch_size = test_loader.batch_size

            with torch.no_grad():
                for i in range(0, len(features_permuted), batch_size):
                    batch_features = features_permuted[i : i + batch_size].to(device)
                    # Need corresponding images
                    batch_idx = 0
                    images_batch = None

                    # Reconstruct images for this batch
                    for batch in test_loader:
                        images, _, _ = batch
                        batch_end = batch_idx + images.shape[0]
                        if batch_idx <= i < batch_end:
                            start_in_batch = i - batch_idx
                            end_in_batch = min(i + batch_size, batch_end) - batch_idx
                            images_batch = images[start_in_batch:end_in_batch].to(device)
                            break
                        batch_idx = batch_end

                    if images_batch is None:
                        continue

                    outputs = model(images_batch, batch_features[: images_batch.shape[0]])
                    preds = outputs.argmax(dim=1)
                    preds_permuted.extend(preds.cpu().numpy())

            # Compute accuracy drop
            acc_permuted = accuracy_score(all_targets[: len(preds_permuted)], preds_permuted)
            importances[feature_idx, repeat] = baseline_accuracy - acc_permuted

    # 3. Aggregate importance scores
    importances_mean = importances.mean(axis=1)
    importances_std = importances.std(axis=1)

    # Sort by importance
    sorted_idx = np.argsort(importances_mean)[::-1]

    results = {
        "importances_mean": importances_mean.tolist(),
        "importances_std": importances_std.tolist(),
        "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(n_features)],
        "baseline_accuracy": float(baseline_accuracy),
        "sorted_indices": sorted_idx.tolist(),
    }

    logger.info(f"Top 5 features: {[results['feature_names'][i] for i in sorted_idx[:5]]}")
    return results


def compute_gradient_attribution(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    feature_names: list[str] | None = None,
) -> dict:
    """Compute gradient-based feature attribution.

    Algorithm:
    1. For each test sample:
        - Forward pass with requires_grad=True on features
        - Backprop from predicted class logit
        - Record gradient magnitudes
    2. Average absolute gradients per feature
    3. Compute per-class attribution

    Args:
        model: Trained DualPathway model
        test_loader: Test data loader
        device: Device to run on
        feature_names: List of feature names

    Returns:
        Dictionary with global and per-class attribution
    """
    model.eval()

    n_features = 50

    # Track gradients: global and per-class
    global_gradients: list[np.ndarray] = []
    per_class_gradients: dict[str, list[np.ndarray]] = {class_name: [] for class_name in CLASSES}

    logger.info("Computing gradient attribution...")
    for batch in tqdm(test_loader, desc="Batches"):
        images, features, targets = batch
        images = images.to(device)
        features = features.to(device).requires_grad_(True)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images, features)
        preds = outputs.argmax(dim=1)

        # For each sample in batch, backprop from predicted class
        for i in range(outputs.shape[0]):
            # Zero previous gradients
            if features.grad is not None:
                features.grad.zero_()

            # Backprop from predicted class logit
            pred_class = preds[i].item()
            target_class = targets[i].item()

            outputs[i, pred_class].backward(retain_graph=True)

            # Record gradient magnitude
            grad = features.grad[i].abs().cpu().numpy()
            global_gradients.append(grad)

            # Per-class attribution
            class_name = CLASSES[target_class]
            per_class_gradients[class_name].append(grad)

    # Aggregate gradients
    global_attribution = np.mean(global_gradients, axis=0)

    per_class_attribution = {}
    for class_name, grads in per_class_gradients.items():
        if len(grads) > 0:
            per_class_attribution[class_name] = np.mean(grads, axis=0).tolist()
        else:
            per_class_attribution[class_name] = [0.0] * n_features

    results = {
        "global_attribution": global_attribution.tolist(),
        "per_class_attribution": per_class_attribution,
        "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(n_features)],
    }

    logger.info("Gradient attribution computed")
    return results


def categorize_features(feature_names: list[str]) -> dict[str, list[int]]:
    """Categorize features by type.

    Args:
        feature_names: List of 50 feature names

    Returns:
        Dictionary mapping category names to feature indices
    """
    categories: dict[str, list[int]] = {
        "Intensity": [],
        "GLCM Texture": [],
        "Shape": [],
        "Region": [],
        "Wavelet": [],
    }

    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if "glcm" in name_lower:
            categories["GLCM Texture"].append(i)
        elif "wavelet" in name_lower:
            categories["Wavelet"].append(i)
        elif any(
            x in name_lower
            for x in ["area", "perimeter", "eccentricity", "solidity", "extent", "axis", "compactness", "sphericity"]
        ):
            categories["Shape"].append(i)
        elif any(x in name_lower for x in ["ratio", "gradient", "boundary"]):
            categories["Region"].append(i)
        else:
            categories["Intensity"].append(i)

    return categories


def generate_importance_plots(
    perm_results: dict | None,
    grad_results: dict | None,
    output_dir: Path,
    top_k: int = 20,
) -> dict[str, Path]:
    """Generate feature importance visualizations.

    Args:
        perm_results: Permutation importance results (optional)
        grad_results: Gradient attribution results (optional)
        output_dir: Directory to save plots
        top_k: Number of top features to display

    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    feature_names = perm_results["feature_names"] if perm_results else grad_results["feature_names"]

    # 1. Permutation importance bar chart
    if perm_results:
        sorted_idx = np.array(perm_results["sorted_indices"][:top_k])
        importances = np.array(perm_results["importances_mean"])[sorted_idx]
        stds = np.array(perm_results["importances_std"])[sorted_idx]
        names = [feature_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, importances, xerr=stds, align="center", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Accuracy Drop)")
        ax.set_title(f"Top {top_k} Features by Permutation Importance")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        perm_path = output_dir / "top_features_permutation.png"
        plt.savefig(perm_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["top_features_permutation"] = perm_path

    # 2. Per-class gradient attribution heatmap
    if grad_results:
        per_class = grad_results["per_class_attribution"]
        df_data = []
        for _class_name, attributions in per_class.items():
            df_data.append(attributions)

        df = pd.DataFrame(df_data, index=CLASSES, columns=feature_names).T

        # Show top features based on mean attribution
        mean_attribution = df.mean(axis=1)
        top_features_idx = mean_attribution.nlargest(top_k).index

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            df.loc[top_features_idx],
            cmap="YlOrRd",
            annot=False,
            ax=ax,
            cbar_kws={"label": "Mean Absolute Gradient"},
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_k} Features: Per-Class Gradient Attribution")

        plt.tight_layout()
        grad_path = output_dir / "per_class_attribution.png"
        plt.savefig(grad_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["per_class_attribution"] = grad_path

    # 3. Category-level importance
    if perm_results:
        categories = categorize_features(feature_names)
        category_importance = {}

        importances = np.array(perm_results["importances_mean"])
        for cat_name, indices in categories.items():
            if len(indices) > 0:
                category_importance[cat_name] = np.mean(importances[indices])

        fig, ax = plt.subplots(figsize=(8, 6))
        cats = list(category_importance.keys())
        vals = list(category_importance.values())

        ax.bar(cats, vals, alpha=0.8)
        ax.set_xlabel("Feature Category")
        ax.set_ylabel("Mean Importance")
        ax.set_title("Feature Importance by Category")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        cat_path = output_dir / "feature_category_summary.png"
        plt.savefig(cat_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["feature_category_summary"] = cat_path

    logger.info(f"Saved {len(plots)} importance plots to {output_dir}")
    return plots


def save_feature_ranking(results: dict, output_dir: Path) -> Path:
    """Save ranked feature list as text file.

    Args:
        results: Permutation importance results
        output_dir: Directory to save ranking

    Returns:
        Path to ranking file
    """
    ranking_path = output_dir / "feature_ranking.txt"

    with ranking_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline Accuracy: {results['baseline_accuracy']:.4f}\n\n")

        sorted_idx = results["sorted_indices"]
        importances = results["importances_mean"]
        stds = results["importances_std"]
        names = results["feature_names"]

        f.write("Rank | Feature Name                      | Importance | Std Dev\n")
        f.write("-" * 80 + "\n")

        for rank, idx in enumerate(sorted_idx, 1):
            f.write(f"{rank:4d} | {names[idx]:33s} | {importances[idx]:10.6f} | {stds[idx]:.6f}\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Saved feature ranking to {ranking_path}")
    return ranking_path


@app.command()
def main(
    checkpoint: str = typer.Argument(..., help="Path to DualPathway model checkpoint"),
    method: str = typer.Option("permutation", "--method", "-m", help="Analysis method (permutation, gradient, both)"),
    n_repeats: int = typer.Option(10, "--n-repeats", "-n", help="Number of permutation repeats"),
    output_dir: str = typer.Option("reports/feature_importance", "--output-dir", "-o", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option("CT_Scan_MLOps", help="W&B project name"),
    top_k: int = typer.Option(20, "--top-k", help="Number of top features to display"),
) -> None:
    """Analyze radiomics feature importance for DualPathway model."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize W&B if requested
    if use_wandb:
        wandb.init(project=wandb_project, name="feature_importance", job_type="analysis")

    # Load feature metadata
    metadata = load_feature_metadata()
    feature_names = metadata["feature_names"]

    # Load model
    config_path = checkpoint_path.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create test dataloader with features
    logger.info("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(cfg, use_features=True)

    # Compute importance
    perm_results = None
    grad_results = None

    if method in ["permutation", "both"]:
        logger.info("Computing permutation importance...")
        perm_results = compute_permutation_importance(model, test_loader, device, n_repeats, feature_names)

        # Save results
        output_path.mkdir(parents=True, exist_ok=True)
        perm_path = output_path / "permutation_importance.json"
        with perm_path.open("w") as f:
            json.dump(perm_results, f, indent=2)
        logger.info(f"Saved permutation results to {perm_path}")

        # Save ranking
        save_feature_ranking(perm_results, output_path)

    if method in ["gradient", "both"]:
        logger.info("Computing gradient attribution...")
        grad_results = compute_gradient_attribution(model, test_loader, device, feature_names)

        # Save results
        output_path.mkdir(parents=True, exist_ok=True)
        grad_path = output_path / "gradient_attribution.json"
        with grad_path.open("w") as f:
            json.dump(grad_results, f, indent=2)
        logger.info(f"Saved gradient results to {grad_path}")

    # Generate plots
    plots = generate_importance_plots(perm_results, grad_results, output_path, top_k)

    # Log to W&B
    if use_wandb:
        log_metrics = {}
        if perm_results:
            log_metrics["baseline_accuracy"] = perm_results["baseline_accuracy"]
            # Log top 10 feature importances
            for i, idx in enumerate(perm_results["sorted_indices"][:10]):
                log_metrics[f"top_{i + 1}_feature"] = perm_results["importances_mean"][idx]

        log_to_wandb(log_metrics, plots, "feature_importance")
        wandb.finish()

    logger.info(f"\nFeature importance analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()
