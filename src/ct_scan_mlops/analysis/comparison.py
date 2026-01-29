"""Model comparison logic: Baseline vs Improved."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from ct_scan_mlops.analysis.core import InferenceEngine, ModelLoader, PredictionResults
from ct_scan_mlops.data import CLASSES, create_dataloaders
from ct_scan_mlops.utils import get_device


class ModelComparator:
    """Compares two models on the same test set."""

    def __init__(self, baseline_path: Path, improved_path: Path, output_dir: Path):
        self.baseline_path = baseline_path
        self.improved_path = improved_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()

    def run(self):
        """Execute full comparison workflow."""
        # 1. Load Models
        baseline_model = ModelLoader.load(self.baseline_path, self.device)
        improved_model = ModelLoader.load(self.improved_path, self.device)

        # 2. Get Dataloader (use improved model's config as it's likely more recent/correct)
        # Note: If data configs differ significantly, this might be an issue.
        # But for model comparison, we usually assume same test set.
        logger.info("Creating dataloader based on improved model config...")
        _, _, test_loader = create_dataloaders(
            improved_model.config, use_features=improved_model.uses_features or baseline_model.uses_features
        )

        # 3. Run Inference
        logger.info("Running inference on Baseline...")
        engine_base = InferenceEngine(baseline_model, self.device)
        res_base = engine_base.run_inference(test_loader, desc="Baseline")

        logger.info("Running inference on Improved...")
        engine_imp = InferenceEngine(improved_model, self.device)
        res_imp = engine_imp.run_inference(test_loader, desc="Improved")

        # 4. Compare
        self._compare_metrics(res_base, res_imp)
        self._compare_per_class(res_base, res_imp)

        logger.info(f"Comparison complete. Results in {self.output_dir}")

    def _compare_metrics(self, base: PredictionResults, imp: PredictionResults):
        """Compare overall metrics."""
        metrics = {
            "baseline_accuracy": base.accuracy,
            "improved_accuracy": imp.accuracy,
            "improvement_absolute": imp.accuracy - base.accuracy,
            "improvement_relative_pct": ((imp.accuracy - base.accuracy) / base.accuracy) * 100,
        }

        with (self.output_dir / "comparison_summary.json").open("w") as f:
            json.dump(metrics, f, indent=2)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.bar(["Baseline", "Improved"], [base.accuracy, imp.accuracy], color=["gray", "green"])
        ax.set_ylim(0, 1.0)
        ax.set_title("Accuracy Comparison")
        ax.bar_label(bars, fmt="%.4f")
        plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=150)
        plt.close()

    def _compare_per_class(self, base: PredictionResults, imp: PredictionResults):
        """Compare per-class F1 scores."""
        from sklearn.metrics import f1_score

        f1_base = f1_score(base.targets, base.predictions, average=None)
        f1_imp = f1_score(imp.targets, imp.predictions, average=None)

        df = pd.DataFrame({"Class": CLASSES, "Baseline F1": f1_base, "Improved F1": f1_imp})

        # Plot
        df_melt = df.melt(id_vars="Class", var_name="Model", value_name="F1 Score")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Class", y="F1 Score", hue="Model", ax=ax, palette=["gray", "green"])
        ax.set_title("Per-Class F1 Score Comparison")
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "per_class_comparison.png", dpi=150)
        plt.close()
