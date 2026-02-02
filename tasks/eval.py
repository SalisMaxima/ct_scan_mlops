"""Model evaluation and analysis tasks."""

import os

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def analyze(ctx: Context, args: str = "") -> None:
    """Run model analysis CLI (diagnose, explain, compare).

    The analysis CLI provides three commands:
    - diagnose: Full diagnostics (metrics, confusion matrix, error analysis)
    - explain: Feature importance analysis (for DualPathway models)
    - compare: Compare two models

    Args:
        args: Arguments to pass to the analysis CLI

    Examples:
        invoke eval.analyze "diagnose --checkpoint outputs/model.ckpt"
        invoke eval.analyze "explain --checkpoint outputs/model.ckpt --method permutation"
        invoke eval.analyze "compare --baseline model1.ckpt --improved model2.ckpt"
        invoke eval.analyze "--help"  # Show all available commands
    """
    if not args:
        logger.error("ERROR: args required. Use 'invoke eval.analyze \"--help\"' for usage.")
        return

    ctx.run(f"uv run python -m {PROJECT_NAME}.analysis.cli {args}", echo=True, pty=not WINDOWS)


@task
def benchmark(ctx: Context, checkpoint: str = "", dataset: str = "test", batch_size: int = 32) -> None:
    """Benchmark model inference speed and throughput.

    Args:
        checkpoint: Path to model checkpoint
        dataset: Dataset split to use (train/val/test)
        batch_size: Batch size for inference

    Examples:
        invoke eval.benchmark --checkpoint outputs/best_model.ckpt
        invoke eval.benchmark --checkpoint model.pt --batch-size 64
    """
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return

    from pathlib import Path

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return

    ctx.run(
        f"uv run python -m {PROJECT_NAME}.inference_benchmark --checkpoint {checkpoint} --dataset {dataset} --batch-size {batch_size}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile(ctx: Context, checkpoint: str = "", output: str = "profile_stats.txt") -> None:
    """Profile model and training performance using cProfile.

    Args:
        checkpoint: Path to model checkpoint (optional - will profile training if not provided)
        output: Output file for profile stats

    Examples:
        invoke eval.profile --checkpoint outputs/model.ckpt
        invoke eval.profile  # Profile training
    """
    if checkpoint:
        from pathlib import Path

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
            return

        ctx.run(
            f"uv run python -m cProfile -o {output} -m {PROJECT_NAME}.inference_benchmark --checkpoint {checkpoint}",
            echo=True,
            pty=not WINDOWS,
        )
    else:
        ctx.run(
            f"uv run python -m cProfile -o {output} -m {PROJECT_NAME}.train train.max_epochs=1",
            echo=True,
            pty=not WINDOWS,
        )

    print(f"\n✓ Profile saved to {output}")
    print("View with: uv run python -m pstats {output}")


@task
def model_info(ctx: Context, checkpoint: str) -> None:
    """Show model size, parameters, and architecture info.

    Args:
        checkpoint: Path to model checkpoint

    Examples:
        invoke eval.model-info --checkpoint outputs/model.ckpt
    """
    if not checkpoint:
        logger.error("ERROR: --checkpoint is required")
        return

    from pathlib import Path

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"ERROR: Checkpoint not found: {checkpoint}")
        return

    ctx.run(
        f"uv run python -m {PROJECT_NAME}.model_info --checkpoint {checkpoint}",
        echo=True,
        pty=not WINDOWS,
    )
