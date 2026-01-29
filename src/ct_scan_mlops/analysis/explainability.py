"Explainability analysis: Feature importance and attribution."

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_scan_mlops.analysis.core import LoadedModel, unpack_batch
from ct_scan_mlops.analysis.utils import load_feature_metadata
from ct_scan_mlops.data import CLASSES


class FeatureExplainer:
    """Analyzes radiomics feature importance."""

    def __init__(self, loaded_model: LoadedModel, device: torch.device, output_dir: Path):
        self.model = loaded_model.model
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        try:
            metadata = load_feature_metadata()
            self.feature_names = metadata["feature_names"]
        except FileNotFoundError:
            logger.warning("Feature metadata not found. Using generic names.")
            self.feature_names = [f"feat_{i}" for i in range(50)]  # Default fallback

    def compute_permutation_importance(self, test_loader: DataLoader, n_repeats: int = 10) -> dict:
        """Compute permutation importance by shuffling features."""
        logger.info("Computing permutation importance...")
        self.model.eval()

        # 1. Cache Data (similar to original optimized logic)
        all_images = []
        all_features = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Caching data"):
                imgs, feats, targs = unpack_batch(batch, self.device, use_features=True)
                all_images.append(imgs.cpu())
                all_features.append(feats.cpu())
                all_targets.append(targs.cpu())

        # Move to device for batch processing if memory allows, or keep CPU and batch
        # For simplicity/speed on typical hardware, let's keep big tensors on CPU
        # and move batches to GPU during inference loop.
        X_img = torch.cat(all_images)
        X_feat = torch.cat(all_features)
        y_true = torch.cat(all_targets).numpy()

        # Baseline
        batch_size = test_loader.batch_size or 32
        baseline_preds = self._batched_inference(X_img, X_feat, batch_size)
        baseline_acc = accuracy_score(y_true, baseline_preds)
        logger.info(f"Baseline Accuracy: {baseline_acc:.4f}")

        # Permutation Loop
        n_features = X_feat.shape[1]
        n_samples = len(X_feat)
        importances = np.zeros((n_features, n_repeats))

        for f_idx in tqdm(range(n_features), desc="Features"):
            for r in range(n_repeats):
                # Shuffle feature column
                X_feat_perm = X_feat.clone()
                perm_idx = torch.randperm(n_samples)
                X_feat_perm[:, f_idx] = X_feat_perm[perm_idx, f_idx]

                # Inference
                preds = self._batched_inference(X_img, X_feat_perm, batch_size)
                acc = accuracy_score(y_true, preds)

                # Drop in accuracy = Importance
                importances[f_idx, r] = baseline_acc - acc

        # Aggregate
        mean_imp = importances.mean(axis=1)
        std_imp = importances.std(axis=1)
        sorted_idx = np.argsort(mean_imp)[::-1]

        results = {
            "baseline_accuracy": baseline_acc,
            "feature_names": self.feature_names,
            "importances_mean": mean_imp.tolist(),
            "importances_std": std_imp.tolist(),
            "sorted_indices": sorted_idx.tolist(),
        }

        self._plot_permutation_importance(results)
        self._save_ranking(results)

        with (self.output_dir / "permutation_importance.json").open("w") as f:
            json.dump(results, f, indent=2)

        return results

    def _batched_inference(self, images: torch.Tensor, features: torch.Tensor, batch_size: int) -> np.ndarray:
        """Helper for manual batched inference."""
        preds = []
        n_samples = len(images)
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_img = images[i:end].to(self.device)
                batch_feat = features[i:end].to(self.device)

                out = self.model(batch_img, batch_feat)
                preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(preds)

    def compute_gradient_attribution(self, test_loader: DataLoader) -> dict:
        """Compute gradient-based attribution."""
        logger.info("Computing gradient attribution...")
        self.model.eval()

        global_grads = []
        per_class_grads: dict[str, list] = {c: [] for c in CLASSES}

        for batch in tqdm(test_loader, desc="Gradient Attribution"):
            imgs, feats, targs = unpack_batch(batch, self.device, use_features=True)
            feats.requires_grad_(True)

            outputs = self.model(imgs, feats)
            preds = outputs.argmax(dim=1)

            # Backprop for each sample
            for i in range(len(preds)):
                if feats.grad is not None:
                    feats.grad.zero_()

                pred_class = preds[i].item()
                target_class = (
                    targs[i].item()
                )  # Use target for attribution analysis? Or predicted? Usually predicted for "model explanation".

                outputs[i, pred_class].backward(retain_graph=True)

                grad = feats.grad[i].abs().cpu().numpy()
                global_grads.append(grad)
                per_class_grads[CLASSES[target_class]].append(
                    grad
                )  # Group by TRUE class to see what features matter for specific classes

        # Aggregate
        global_attr = np.mean(global_grads, axis=0)
        per_class_attr = {}
        for cls, grads in per_class_grads.items():
            if grads:
                per_class_attr[cls] = np.mean(grads, axis=0).tolist()
            else:
                per_class_attr[cls] = [0.0] * len(self.feature_names)

        results = {
            "global_attribution": global_attr.tolist(),
            "per_class_attribution": per_class_attr,
            "feature_names": self.feature_names,
        }

        self._plot_gradient_attribution(results)

        with (self.output_dir / "gradient_attribution.json").open("w") as f:
            json.dump(results, f, indent=2)

        return results

    def _plot_permutation_importance(self, results: dict, top_k: int = 20):
        indices = results["sorted_indices"][:top_k]
        names = [self.feature_names[i] for i in indices]
        means = np.array(results["importances_mean"])[indices]
        stds = np.array(results["importances_std"])[indices]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stds, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Accuracy Drop)")
        ax.set_title(f"Top {top_k} Features (Permutation)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_features_permutation.png", dpi=150)
        plt.close()

    def _plot_gradient_attribution(self, results: dict, top_k: int = 20):
        # Heatmap
        df_data = []
        for _cls, attr in results["per_class_attribution"].items():
            df_data.append(attr)

        df = pd.DataFrame(df_data, index=CLASSES, columns=self.feature_names).T

        # Sort by max attribution across classes
        df["max_val"] = df.max(axis=1)
        df_sorted = df.sort_values("max_val", ascending=False).drop(columns="max_val").head(top_k)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_sorted, cmap="YlOrRd", ax=ax)
        ax.set_title("Gradient Attribution by Class")
        plt.tight_layout()
        plt.savefig(self.output_dir / "gradient_attribution_heatmap.png", dpi=150)
        plt.close()

    def _save_ranking(self, results: dict):
        path = self.output_dir / "feature_ranking.txt"
        indices = results["sorted_indices"]
        names = results["feature_names"]
        means = results["importances_mean"]

        with path.open("w") as f:
            f.write("Rank | Feature | Importance\n")
            f.write("-" * 50 + "\n")
            for rank, idx in enumerate(indices, 1):
                f.write(f"{rank:4d} | {names[idx]:30s} | {means[idx]:.6f}\n")
