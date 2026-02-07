"""Data management tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def download(ctx: Context) -> None:
    """Download CT scan dataset from Kaggle."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data download", echo=True, pty=not WINDOWS)


@task
def preprocess(ctx: Context) -> None:
    """Preprocess data (resize, normalize, save as tensors)."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data preprocess", echo=True, pty=not WINDOWS)


@task
def extract_features(ctx: Context, n_jobs: int = -1, args: str = "") -> None:
    """Extract radiomics features from preprocessed images.

    Args:
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        args: Additional Hydra config overrides

    Examples:
        invoke data.extract-features
        invoke data.extract-features --n-jobs 4
        invoke data.extract-features --args "features.use_wavelet=false"
    """
    full_args = f"+n_jobs={n_jobs} {args}".strip()
    ctx.run(f"uv run python -m {PROJECT_NAME}.features.extract_radiomics {full_args}", echo=True, pty=not WINDOWS)


@task
def prepare_sweep_features(ctx: Context, n_jobs: int = -1) -> None:
    """Extract all feature configurations needed for sweeps.

    Extracts both default (50 features) and top_features (16 features)
    to ensure all sweep configurations can run.

    Args:
        n_jobs: Number of parallel jobs (-1 = all CPUs)

    Examples:
        invoke data.prepare-sweep-features
        invoke data.prepare-sweep-features --n-jobs 4
    """
    print("Extracting default features (50 features)...")
    ctx.run(f"uv run python -m {PROJECT_NAME}.features.extract_radiomics +n_jobs={n_jobs}", echo=True, pty=not WINDOWS)

    print("Extracting top features (16 features)...")
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.features.extract_radiomics features=top_features +n_jobs={n_jobs}",
        echo=True,
        pty=not WINDOWS,
    )

    print("✓ Feature extraction complete. Ready for dual_pathway sweeps.")


@task
def stats(ctx: Context) -> None:
    """Show dataset statistics."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.stats", echo=True, pty=not WINDOWS)


@task
def validate(ctx: Context) -> None:
    """Validate data integrity and structure."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.validate", echo=True, pty=not WINDOWS)
