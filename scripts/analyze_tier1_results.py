#!/usr/bin/env python
"""
Tier 1 Results Analysis
=======================

Analyzes the 7 loss function experiments from Tier 1 and identifies the best approach.

Usage:
    uv run python scripts/analyze_tier1_results.py
    uv run python scripts/analyze_tier1_results.py --detailed
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb
from loguru import logger
from tabulate import tabulate  # type: ignore[import-untyped]


def fetch_tier1_runs(project_name: str, hours_back: int = 2):
    """Fetch Tier 1 experiment runs from W&B."""
    logger.info(f"Fetching Tier 1 runs from W&B project: {project_name}")

    api = wandb.Api()
    runs = api.runs(project_name)

    # Filter for Tier 1 experiments
    tier1_runs = []
    cutoff_time = datetime.now().timestamp() - (hours_back * 3600)

    for run in runs:
        # Check if it's a Tier 1 experiment
        if "T1_" in run.name and run.state == "finished":
            # Check if it was created recently
            created_at = datetime.fromisoformat(run.created_at.replace("Z", "+00:00")).timestamp()
            if created_at > cutoff_time:
                tier1_runs.append(run)

    logger.info(f"Found {len(tier1_runs)} Tier 1 runs")
    return tier1_runs


def extract_run_data(run):
    """Extract relevant metrics from a W&B run."""
    config = run.config
    summary = run.summary

    return {
        "name": run.name,
        "run_id": run.id,
        "test_acc": summary.get("test_acc", 0),
        "test_loss": summary.get("test_loss", 0),
        "val_acc": summary.get("val_acc", 0),
        "val_loss": summary.get("val_loss", 0),
        "train_acc": summary.get("train_acc_epoch", 0),
        "train_loss": summary.get("train_loss_epoch", 0),
        "best_val_acc": summary.get("best_val_acc", 0),
        "best_epoch": summary.get("best_epoch", 0),
        "runtime_sec": summary.get("_runtime", 0),
        "loss_type": config.get("train", {}).get("loss", {}).get("type", "cross_entropy"),
        "created_at": run.created_at,
    }


def analyze_results(runs_data: list[dict]) -> dict:
    """Analyze Tier 1 results and identify best performers."""
    df = pd.DataFrame(runs_data)

    # Sort by test accuracy
    df_sorted = df.sort_values("test_acc", ascending=False)

    # Identify best run
    best_run = df_sorted.iloc[0]

    # Calculate statistics
    analysis = {
        "best_run": best_run.to_dict(),
        "top_3": df_sorted.head(3).to_dict("records"),
        "all_runs": df_sorted.to_dict("records"),
        "stats": {
            "mean_test_acc": df["test_acc"].mean(),
            "std_test_acc": df["test_acc"].std(),
            "max_test_acc": df["test_acc"].max(),
            "min_test_acc": df["test_acc"].min(),
            "baseline_acc": df[df["loss_type"] == "cross_entropy"]["test_acc"].values[0]
            if len(df[df["loss_type"] == "cross_entropy"]) > 0
            else None,
        },
    }

    # Group by loss type
    loss_type_stats = df.groupby("loss_type").agg({"test_acc": ["mean", "max", "count"]}).round(4).to_dict()
    analysis["loss_type_stats"] = loss_type_stats

    return analysis


def print_summary(analysis: dict):
    """Print a formatted summary of the results."""
    print("\n" + "=" * 80)
    print("TIER 1 LOSS FUNCTION EXPERIMENTS - RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Best run
    best = analysis["best_run"]
    print("🏆 BEST PERFORMING MODEL:")
    print("-" * 80)
    print(f"  Experiment: {best['name']}")
    print(f"  Loss Type:  {best['loss_type']}")
    print(f"  Test Acc:   {best['test_acc']:.4f} ({best['test_acc'] * 100:.2f}%)")
    print(f"  Test Loss:  {best['test_loss']:.6f}")
    print(f"  Val Acc:    {best['val_acc']:.4f} ({best['val_acc'] * 100:.2f}%)")
    print(f"  Run ID:     {best['run_id']}")
    print()

    # Baseline comparison
    baseline_acc = analysis["stats"].get("baseline_acc")
    if baseline_acc is not None:
        improvement = best["test_acc"] - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        print("📊 IMPROVEMENT OVER BASELINE:")
        print(f"  Baseline:    {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")
        print(f"  Best:        {best['test_acc']:.4f} ({best['test_acc'] * 100:.2f}%)")
        print(f"  Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        print()

    # Top 3
    print("🥇 TOP 3 EXPERIMENTS:")
    print("-" * 80)
    top3_data = []
    for i, run in enumerate(analysis["top_3"], 1):
        medal = ["🥇", "🥈", "🥉"][i - 1]
        top3_data.append(
            [
                f"{medal} {i}",
                run["name"][:40],
                run["loss_type"],
                f"{run['test_acc']:.4f}",
                f"{run['test_loss']:.6f}",
                f"{run['val_acc']:.4f}",
            ]
        )

    print(
        tabulate(
            top3_data,
            headers=["Rank", "Experiment", "Loss Type", "Test Acc", "Test Loss", "Val Acc"],
            tablefmt="simple",
        )
    )
    print()

    # All experiments comparison
    print("📋 ALL EXPERIMENTS (sorted by test accuracy):")
    print("-" * 80)
    all_data = []
    for i, run in enumerate(analysis["all_runs"], 1):
        all_data.append(
            [
                i,
                run["name"][:35],
                run["loss_type"][:20],
                f"{run['test_acc']:.4f}",
                f"{run['test_loss']:.6f}",
                f"{run['runtime_sec'] / 60:.1f}m",
            ]
        )

    print(
        tabulate(
            all_data,
            headers=["#", "Experiment", "Loss Type", "Test Acc", "Test Loss", "Runtime"],
            tablefmt="simple",
        )
    )
    print()

    # Statistics
    stats = analysis["stats"]
    print("📊 STATISTICS:")
    print("-" * 80)
    print(f"  Mean Test Accuracy: {stats['mean_test_acc']:.4f} ± {stats['std_test_acc']:.4f}")
    print(f"  Range: {stats['min_test_acc']:.4f} - {stats['max_test_acc']:.4f}")
    print()


def print_recommendations(analysis: dict):
    """Print recommendations based on the results."""
    print("=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)
    print()

    best = analysis["best_run"]
    baseline_acc = analysis["stats"].get("baseline_acc", 0.9524)

    # Determine recommendation based on improvement
    improvement = best["test_acc"] - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100

    if improvement >= 0.012:  # 1.2% improvement
        recommendation = "EXCELLENT"
        emoji = "🚀"
    elif improvement >= 0.008:  # 0.8% improvement
        recommendation = "GOOD"
        emoji = "✅"
    elif improvement >= 0.002:  # 0.2% improvement
        recommendation = "MODEST"
        emoji = "👍"
    else:
        recommendation = "MINIMAL"
        emoji = "⚠️"

    print(f"{emoji} IMPROVEMENT: {recommendation} ({improvement_pct:+.2f}%)")
    print()

    # Specific recommendations
    print("NEXT STEPS:")
    print()

    if improvement >= 0.010:
        print(f"1. {emoji} Results are promising! Proceed to Tier 2 & 3 experiments")
        print(f"   Best loss function: {best['loss_type']}")
        print()
        print("2. Run Tier 2 (Weighted Sampling) with best loss:")
        print(f"   BEST_LOSS='train.loss.type={best['loss_type']}'")
        print()
        print("3. Evaluate best model for detailed confusion matrix:")
        print("   invoke evaluate --checkpoint outputs/.../checkpoints/best_model.ckpt")
        print()
        print("4. Consider running full experiment suite:")
        print("   ./scripts/adenocarcinoma_improvement_experiments.sh")

    elif improvement >= 0.005:
        print(f"1. {emoji} Modest improvement detected")
        print(f"   Best loss function: {best['loss_type']}")
        print()
        print("2. Evaluate best model to check confusion patterns:")
        print("   invoke evaluate --checkpoint outputs/.../checkpoints/best_model.ckpt")
        print()
        print("3. If adenocarcinoma recall improved, proceed to Tier 2")
        print("   Otherwise, consider:")
        print("   - Trying different hyperparameters")
        print("   - Investigating data quality")
        print("   - Exploring feature engineering")

    else:
        print(f"1. {emoji} Minimal improvement - investigate before proceeding")
        print()
        print("2. Evaluate best model and analyze confusion matrix")
        print()
        print("3. Consider alternative approaches:")
        print("   - Different augmentation strategies")
        print("   - Class-balanced sampling")
        print("   - Feature engineering")
        print("   - Ensemble methods")
        print()
        print("4. Review training logs for issues:")
        print("   - Overfitting")
        print("   - Learning rate problems")
        print("   - Data quality issues")

    print()
    print("-" * 80)
    print()

    # Target check
    target_acc = 0.97
    print("🎯 TARGET PROGRESS:")
    print(f"  Current Best:   {best['test_acc']:.4f} ({best['test_acc'] * 100:.2f}%)")
    print(f"  Target:         {target_acc:.4f} ({target_acc * 100:.2f}%)")
    remaining = target_acc - best["test_acc"]
    print(f"  Remaining:      {remaining:.4f} ({remaining * 100:.2f}%) to reach target")
    print()

    if best["test_acc"] >= target_acc:
        print("🎉 TARGET ACHIEVED! Excellent work!")
    elif best["test_acc"] >= 0.965:
        print("✅ Close to target! Continue with Tiers 2-5 for final push")
    elif best["test_acc"] >= 0.96:
        print("👍 Good progress! Tiers 2-5 should get you to target")
    else:
        print("⚠️  Significant gap remains. May need more aggressive strategies")

    print()


def save_report(analysis: dict, output_file: Path):
    """Save detailed report to file."""
    with output_file.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("TIER 1 LOSS FUNCTION EXPERIMENTS - DETAILED REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Best run
        best = analysis["best_run"]
        f.write("BEST PERFORMING MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Experiment: {best['name']}\n")
        f.write(f"Loss Type: {best['loss_type']}\n")
        f.write(f"Test Accuracy: {best['test_acc']:.6f}\n")
        f.write(f"Test Loss: {best['test_loss']:.6f}\n")
        f.write(f"Validation Accuracy: {best['val_acc']:.6f}\n")
        f.write(f"Run ID: {best['run_id']}\n")
        f.write(
            f"W&B URL: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps/runs/{best['run_id']}\n"
        )
        f.write("\n")

        # All runs
        f.write("ALL EXPERIMENTS:\n")
        f.write("-" * 80 + "\n")
        for i, run in enumerate(analysis["all_runs"], 1):
            f.write(f"{i}. {run['name']}\n")
            f.write(f"   Loss Type: {run['loss_type']}\n")
            f.write(f"   Test Acc: {run['test_acc']:.6f}\n")
            f.write(f"   Test Loss: {run['test_loss']:.6f}\n")
            f.write(f"   Val Acc: {run['val_acc']:.6f}\n")
            f.write(f"   Runtime: {run['runtime_sec'] / 60:.1f} min\n")
            f.write("\n")

        # Statistics
        f.write("STATISTICS:\n")
        f.write("-" * 80 + "\n")
        stats = analysis["stats"]
        f.write(f"Mean Test Accuracy: {stats['mean_test_acc']:.6f}\n")
        f.write(f"Std Test Accuracy: {stats['std_test_acc']:.6f}\n")
        f.write(f"Max Test Accuracy: {stats['max_test_acc']:.6f}\n")
        f.write(f"Min Test Accuracy: {stats['min_test_acc']:.6f}\n")
        if stats.get("baseline_acc"):
            f.write(f"Baseline Accuracy: {stats['baseline_acc']:.6f}\n")
            improvement = best["test_acc"] - stats["baseline_acc"]
            f.write(f"Improvement: {improvement:+.6f} ({improvement / stats['baseline_acc'] * 100:+.2f}%)\n")

    logger.info(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Tier 1 experiment results")
    parser.add_argument(
        "--project",
        type=str,
        default="mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps",
        help="W&B project name",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=2,
        help="How many hours back to search for runs",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-run information",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/tier1_analysis.txt"),
        help="Output file for detailed report",
    )

    args = parser.parse_args()

    # Fetch runs
    runs = fetch_tier1_runs(args.project, args.hours_back)

    if not runs:
        logger.error("No Tier 1 runs found. Make sure experiments have completed.")
        return

    # Extract data
    runs_data = [extract_run_data(run) for run in runs]

    # Analyze
    analysis = analyze_results(runs_data)

    # Print summary
    print_summary(analysis)

    # Print recommendations
    print_recommendations(analysis)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(analysis, args.output)

    print("=" * 80)
    print(f"📄 Detailed report saved to: {args.output}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
