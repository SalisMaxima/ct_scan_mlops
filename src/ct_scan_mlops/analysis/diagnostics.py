"""Unified diagnostics module for model evaluation and error analysis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

from ct_scan_mlops.analysis.core import PredictionResults
from ct_scan_mlops.analysis.utils import save_image_grid
from ct_scan_mlops.data import CLASSES


class ModelDiagnostician:
    """Consolidated diagnostics for model performance, errors, and confusion patterns."""

    def __init__(self, results: PredictionResults, output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots: dict[str, Path] = {}
        self.metrics: dict[str, Any] = {}

    def evaluate_performance(self) -> dict[str, float]:
        """Compute standard classification metrics and confusion matrix."""
        logger.info("Evaluating performance...")

        # Classification Report
        report = classification_report(
            self.results.targets,
            self.results.predictions,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        # Flatten metrics
        self.metrics["accuracy"] = report["accuracy"]
        self.metrics["macro_f1"] = report["macro avg"]["f1-score"]
        self.metrics["weighted_f1"] = report["weighted avg"]["f1-score"]

        for cls in CLASSES:
            if cls in report:
                self.metrics[f"{cls}_f1"] = report[cls]["f1-score"]
                self.metrics[f"{cls}_precision"] = report[cls]["precision"]
                self.metrics[f"{cls}_recall"] = report[cls]["recall"]

        # Confusion Matrix
        cm = confusion_matrix(self.results.targets, self.results.predictions)
        self._plot_confusion_matrix(cm)

        # Save classification report
        report_str = classification_report(
            self.results.targets, self.results.predictions, target_names=CLASSES, zero_division=0
        )
        (self.output_dir / "classification_report.txt").write_text(report_str)

        return self.metrics

    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Acc: {self.results.accuracy:.4f})")

        path = self.output_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        self.plots["confusion_matrix"] = path

    def analyze_errors(self, max_images: int = 50) -> dict:
        """Analyze misclassifications (confidence, heatmaps, image grids)."""
        logger.info("Analyzing errors...")

        errors_mask = self.results.predictions != self.results.targets
        error_indices = np.where(errors_mask)[0]

        if len(error_indices) == 0:
            logger.info("No errors found!")
            return {}

        # 1. Confusion Pairs
        pairs = []
        for idx in error_indices:
            true_label = CLASSES[self.results.targets[idx]]
            pred_label = CLASSES[self.results.predictions[idx]]
            pairs.append(f"{true_label}->{pred_label}")

        pair_counts = Counter(pairs)
        self._plot_confusion_pairs(pair_counts)

        # 2. Confidence Distribution of Errors
        # Get confidence of the WRONG prediction
        error_confs = []
        for idx in error_indices:
            pred_class = self.results.predictions[idx]
            error_confs.append(self.results.probabilities[idx, pred_class])

        self._plot_error_confidence(error_confs)

        # 3. Error Image Grid
        if self.results.images is not None:
            self._plot_error_grid(error_indices, max_images)

        stats = {
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / self.results.num_samples,
            "top_confusion_pairs": pair_counts.most_common(5),
        }

        # Save stats
        with (self.output_dir / "error_stats.json").open("w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def _plot_confusion_pairs(self, pair_counts: Counter):
        top_pairs = pair_counts.most_common(10)
        if not top_pairs:
            return

        pairs, counts = zip(*top_pairs, strict=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, counts, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs)
        ax.invert_yaxis()
        ax.set_title("Top Confusion Pairs (True -> Pred)")
        plt.tight_layout()

        path = self.output_dir / "confusion_pairs.png"
        plt.savefig(path, dpi=150)
        plt.close()

        self.plots["confusion_pairs"] = path

    def _plot_error_confidence(self, confidences: list[float]):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(confidences, bins=15, alpha=0.7, edgecolor="black", color="salmon")
        ax.axvline(np.mean(confidences), color="red", linestyle="--", label="Mean")
        ax.set_xlabel("Confidence of Incorrect Prediction")
        ax.set_title("Error Confidence Distribution")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / "error_confidence.png"
        plt.savefig(path, dpi=150)
        plt.close()

        self.plots["error_confidence"] = path

    def _plot_error_grid(self, error_indices: np.ndarray, max_images: int):
        # Select random subset if too many
        if len(error_indices) > max_images:
            rng = np.random.default_rng(42)
            show_indices = rng.choice(error_indices, max_images, replace=False)
        else:
            show_indices = error_indices

        images = []
        titles = []
        for idx in show_indices:
            images.append(self.results.images[idx])
            t = CLASSES[self.results.targets[idx]]
            p = CLASSES[self.results.predictions[idx]]
            c = self.results.probabilities[idx, self.results.predictions[idx]]
            titles.append(f"T:{t}\nP:{p}\nC:{c:.2f}")

        path = self.output_dir / "error_grid.png"
        save_image_grid(images, titles, path, ncols=5)
        self.plots["error_grid"] = path

    def analyze_specific_confusion(self, class_a: str, class_b: str):
        """Deep dive into confusion between two specific classes (e.g., Adeno vs Squamous)."""
        if class_a not in CLASSES or class_b not in CLASSES:
            logger.warning(f"Invalid classes for confusion analysis: {class_a}, {class_b}")
            return

        logger.info(f"Analyzing confusion: {class_a} vs {class_b}")

        idx_a = CLASSES.index(class_a)
        idx_b = CLASSES.index(class_b)

        # Find samples involved in this confusion (A->B or B->A)
        # Also include correct samples for these classes for comparison
        confused_mask = ((self.results.targets == idx_a) & (self.results.predictions == idx_b)) | (
            (self.results.targets == idx_b) & (self.results.predictions == idx_a)
        )
        confused_indices = np.where(confused_mask)[0]

        if len(confused_indices) == 0:
            logger.info(f"No confusion found between {class_a} and {class_b}")
            return

        # 1. Logit Margin Analysis
        margins = []
        for idx in confused_indices:
            pred_idx = self.results.predictions[idx]
            true_idx = self.results.targets[idx]
            margin = self.results.logits[idx, pred_idx] - self.results.logits[idx, true_idx]
            margins.append(margin)

        self._plot_logit_margins(margins, class_a, class_b)

        # 2. t-SNE Visualization (if features available)
        if self.results.features is not None:
            # Select relevant samples: Confused + Correct (A and B)
            correct_mask = ((self.results.targets == idx_a) & (self.results.predictions == idx_a)) | (
                (self.results.targets == idx_b) & (self.results.predictions == idx_b)
            )

            # Limit correct samples to avoid overcrowding
            correct_indices = np.where(correct_mask)[0]
            if len(correct_indices) > 200:
                rng = np.random.default_rng(42)
                correct_indices = rng.choice(correct_indices, 200, replace=False)

            all_indices = np.concatenate([confused_indices, correct_indices])
            self._plot_tsne(all_indices, class_a, class_b)

    def _plot_logit_margins(self, margins: list[float], class_a: str, class_b: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(margins, bins=15, alpha=0.7, edgecolor="black", color="purple")
        ax.axvline(0.5, color="green", linestyle="--", label="Borderline")
        ax.axvline(1.5, color="red", linestyle="--", label="Confident")
        ax.set_xlabel("Logit Margin (Pred - True)")
        ax.set_title(f"Confusion Margin: {class_a} <-> {class_b}")
        ax.legend()
        plt.tight_layout()

        path = self.output_dir / f"margin_{class_a}_{class_b}.png"
        plt.savefig(path, dpi=150)
        plt.close()

        self.plots[f"margin_{class_a}_{class_b}"] = path

    def _plot_tsne(self, indices: np.ndarray, class_a: str, class_b: str):
        features = self.results.features[indices]
        labels = []

        for idx in indices:
            is_error = self.results.predictions[idx] != self.results.targets[idx]
            if is_error:
                labels.append("Confused")
            else:
                labels.append("Correct")

        tsne = TSNE(n_components=2, perplexity=min(30, len(indices) - 1), random_state=42)
        embeddings = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["red" if label == "Confused" else "green" for label in labels]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.6)

        # Fake legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="Confused"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="Correct"),
        ]
        ax.legend(handles=legend_elements)
        ax.set_title(f"t-SNE: {class_a} vs {class_b}")

        path = self.output_dir / f"tsne_{class_a}_{class_b}.png"
        plt.savefig(path, dpi=150)
        plt.close()

        self.plots[f"tsne_{class_a}_{class_b}"] = path
