"""Generate a comprehensive analysis report for a W&B sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
import wandb
import wandb.apis.reports as wr
from loguru import logger
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def fetch_sweep_runs(sweep_id: str) -> tuple[pd.DataFrame, Any]:
    """Fetch all runs from a W&B sweep and return as a DataFrame.

    Args:
        sweep_id: Full sweep ID (entity/project/sweep_id)

    Returns:
        Tuple of (DataFrame, sweep_object)
    """
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_id)
    except Exception as e:
        logger.error(f"Failed to fetch sweep {sweep_id}: {e}")
        raise typer.Exit(code=1) from e

    data = []
    logger.info(f"Fetching runs for sweep: {sweep_id}")

    for run in tqdm(sweep.runs, desc="Fetching runs"):
        if run.state != "finished":
            continue

        # Flatten config
        row = {}
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[f"config.{k}"] = v

        # Flatten summary metrics
        for k, v in run.summary.items():
            if not k.startswith("_") and isinstance(v, (int, float)):
                row[f"summary.{k}"] = v

        row["run_id"] = run.id
        row["name"] = run.name
        data.append(row)

    if not data:
        logger.error("No finished runs found in sweep")
        raise typer.Exit(code=1)

    df = pd.DataFrame(data)
    logger.info(f"Fetched {len(df)} runs")
    return df, sweep


def get_top_correlations(df: pd.DataFrame, metric: str, top_n: int = 5) -> list[str]:
    """Identify top correlated hyperparameters."""
    config_cols = [c for c in df.columns if c.startswith("config.")]
    numeric_cols = [c for c in config_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return []

    correlations = df[numeric_cols].corrwith(df[metric]).abs().sort_values(ascending=False)
    return [c.replace("config.", "") for c in correlations.head(top_n).index.tolist()]


def generate_wandb_report(sweep: Any, metric: str, top_params: list[str]) -> str:
    """Generate a native W&B report.

    Args:
        sweep: W&B sweep object
        metric: Target metric name (without summary. prefix)
        top_params: List of top hyperparameter names

    Returns:
        URL of the generated report
    """
    logger.info("Generating native W&B report...")

    entity = sweep.entity
    project = sweep.project
    sweep_id = sweep.id

    # Initialize Report
    report = wr.Report(
        project=project,
        title=f"Sweep Analysis: {sweep_id}",
        description=f"Auto-generated analysis for sweep {sweep_id}. Optimized for {metric}.",
    )

    # 1. Parallel Coordinates Plot
    # Shows relationship between top params and metric
    pc_columns = [wr.ParallelCoordinatesPlotColumn(f"c::{p}") for p in top_params] + [
        wr.ParallelCoordinatesPlotColumn(f"s::{metric}")
    ]

    # 2. Scatter Plots
    # Create a scatter plot for each top parameter vs metric
    scatter_plots = []
    for param in top_params:
        scatter_plots.append(
            wr.ScatterPlot(
                title=f"{metric} vs {param}",
                x=f"c::{param}",
                y=f"s::{metric}",
            )
        )

    # 3. Parameter Importance (using Weave or simple Bar if available, otherwise just scatters)
    # W&B Reports API is flexible. Let's add a "Sweep Overview" section.

    report.blocks = [
        wr.H1("Sweep Analysis Report"),
        wr.P(f"Analysis of sweep `{sweep_id}` targeting `{metric}`."),
        wr.H2("Parallel Coordinates"),
        wr.P("Visualizing the relationship between high-impact hyperparameters and performance."),
        wr.PanelGrid(
            runsets=[wr.Runset(project=project, entity=entity, filters=f'Sweep == "{sweep_id}"')],
            panels=[wr.ParallelCoordinatesPlot(columns=pc_columns, layout={"h": 12, "w": 24})],
        ),
        wr.H2("Hyperparameter Correlations"),
        wr.P(f"Top {len(top_params)} hyperparameters correlated with {metric}."),
        wr.PanelGrid(
            runsets=[wr.Runset(project=project, entity=entity, filters=f'Sweep == "{sweep_id}"')],
            panels=scatter_plots,
        ),
    ]

    report.save()
    return report.url


def report(
    sweep_id: Annotated[str, typer.Argument(help="Sweep ID (entity/project/sweep_id)")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs/reports/sweep_analysis"),
    metric: Annotated[str, typer.Option(help="Target metric to analyze")] = "test_acc",
    goal: Annotated[str, typer.Option(help="Optimization goal (maximize/minimize)")] = "maximize",
    native: Annotated[bool, typer.Option(help="Generate native W&B report")] = True,
) -> None:
    """Generate a comprehensive analysis report for a W&B sweep."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Data
    df, sweep = fetch_sweep_runs(sweep_id)

    # Ensure metric exists (handle optional summary. prefix)
    target_col = None
    metric_clean = metric.replace("summary.", "")

    if metric in df.columns:
        target_col = metric
    elif f"summary.{metric}" in df.columns:
        target_col = f"summary.{metric}"
    elif metric_clean in df.columns:  # Rare case where it might be a config but treated as metric? Unlikely.
        target_col = metric_clean

    if target_col is None:
        logger.error(
            f"Metric '{metric}' not found in sweep data. Available columns: {[c for c in df.columns if 'summary' in c]}"
        )
        raise typer.Exit(code=1)

    metric_full = target_col

    # 2. Local Analysis (CSV/JSON)
    df.to_csv(output_dir / "sweep_data.csv", index=False)

    try:
        best_run = df.loc[df[metric_full].idxmax()] if goal == "maximize" else df.loc[df[metric_full].idxmin()]
    except ValueError:
        logger.warning(f"No valid data found for metric '{metric_full}'. Cannot identify best run.")
        return

    best_run_info = {
        "run_id": best_run["run_id"],
        "name": best_run["name"],
        f"best_{metric}": best_run[metric_full],
        "config": {k.replace("config.", ""): v for k, v in best_run.items() if k.startswith("config.")},
    }

    with (output_dir / "best_run.json").open("w") as f:
        json.dump(best_run_info, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Saved local data to {output_dir}")

    # 3. Native W&B Report
    if native:
        top_params = get_top_correlations(df, metric_full, top_n=5)
        if not top_params:
            logger.warning("Could not identify numeric hyperparameters for visualization.")
        else:
            try:
                url = generate_wandb_report(sweep, metric_clean, top_params)
                print(f"\n✨ W&B Report Generated: {url}\n")
            except Exception as e:
                logger.error(f"Failed to generate W&B report: {e}")
                # Fallback or just log error?
                # User specifically asked for this, so error is important.


def main() -> None:
    typer.run(report)


if __name__ == "__main__":
    main()
