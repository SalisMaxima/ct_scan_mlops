"""Model monitoring and drift detection tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def extract_stats(ctx: Context, checkpoint: str = "", output: str = "reference_stats.json") -> None:
    """Extract reference statistics for drift detection.

    Args:
        checkpoint: Path to model checkpoint
        output: Output path for reference statistics JSON

    Examples:
        invoke monitor.extract-stats --checkpoint outputs/best_model.ckpt
        invoke monitor.extract-stats --checkpoint model.pt --output prod_stats.json
    """
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return

    ctx.run(
        f"uv run python -m {PROJECT_NAME}.monitoring.extract_stats --checkpoint {checkpoint} --output {output}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def check_drift(ctx: Context, reference: str = "reference_stats.json", data_path: str = "") -> None:
    """Check for data drift against reference statistics.

    Args:
        reference: Path to reference statistics JSON
        data_path: Path to new data to check (optional)

    Examples:
        invoke monitor.check-drift
        invoke monitor.check-drift --reference prod_stats.json
        invoke monitor.check-drift --reference stats.json --data-path data/new_batch
    """
    reference_path = Path(reference)
    if not reference_path.exists():
        logger.error(f"ERROR: Reference statistics not found: {reference}")
        logger.info("Run 'invoke monitor.extract-stats' first to create reference statistics")
        return

    cmd = f"uv run python -m {PROJECT_NAME}.monitoring.drift_check --reference {reference}"

    if data_path:
        cmd += f" --data-path {data_path}"

    ctx.run(cmd, echo=True, pty=not WINDOWS)
