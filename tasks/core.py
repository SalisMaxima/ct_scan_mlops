"""Core environment setup and maintenance tasks."""

import os
from pathlib import Path

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def bootstrap(ctx: Context, name: str = ".venv") -> None:
    """Bootstrap a UV virtual environment and install dependencies."""
    ctx.run(f"uv venv {name}", echo=True, pty=not WINDOWS)
    ctx.run("uv sync", echo=True, pty=not WINDOWS)
    print(f"\n✓ Environment created at {name}")
    print(f"To activate: source {name}/bin/activate  (or {name}\\Scripts\\activate on Windows)")


@task
def sync(ctx: Context) -> None:
    """Install/sync all dependencies."""
    ctx.run("uv sync", echo=True, pty=not WINDOWS)


@task
def dev(ctx: Context) -> None:
    """Install with dev dependencies."""
    ctx.run("uv sync --dev", echo=True, pty=not WINDOWS)


@task
def setup_dev(ctx: Context) -> None:
    """Complete development environment setup - one-command setup."""
    print("Setting up development environment...")

    print("\n1. Installing dependencies...")
    ctx.run("uv sync --dev", echo=True, pty=not WINDOWS)

    print("\n2. Installing pre-commit hooks...")
    ctx.run("uv run pre-commit install", echo=True, pty=not WINDOWS)

    print("\n3. Checking environment...")
    ctx.run("uv run python --version", echo=True, pty=not WINDOWS)
    ctx.run("uv run python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'", echo=True, pty=not WINDOWS)

    # Check if GPU is available
    print("\n4. Checking GPU availability...")
    result = ctx.run("nvidia-smi", warn=True, hide=True, pty=not WINDOWS)
    if result.ok:
        ctx.run(
            "uv run python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'",
            echo=True,
            pty=not WINDOWS,
        )
    else:
        print("   No NVIDIA GPU detected (CPU-only mode)")

    print("\n✓ Development environment ready!")
    print("Next steps:")
    print("  - Run 'source .venv/bin/activate' to activate the environment")
    print("  - Run 'invoke data.download' to download the dataset")
    print("  - Run 'invoke train.train' to start training")


@task
def python(ctx: Context) -> None:
    """Check Python path and version."""
    ctx.run("which python" if os.name != "nt" else "where python", echo=True, pty=not WINDOWS)
    ctx.run("python --version", echo=True, pty=not WINDOWS)


@task
def sync_ai_config(_ctx: Context) -> None:
    """Sync AI assistant config files from CLAUDE.md (source of truth).

    Updates .github/copilot-instructions.md to match CLAUDE.md content.

    Examples:
        invoke core.sync-ai-config
    """
    from loguru import logger

    source = Path("CLAUDE.md")
    copilot_dest = Path(".github/copilot-instructions.md")

    if not source.exists():
        logger.error("ERROR: CLAUDE.md not found")
        return

    content = source.read_text()

    # Transform for Copilot (add header, adjust title)
    copilot_content = content.replace(
        "# CT Scan MLOps",
        "# CT Scan MLOps - Copilot Instructions",
        1,
    )
    copilot_content = copilot_content.replace("## IMPORTANT", "## IMPORTANT RULES", 1)

    copilot_dest.parent.mkdir(parents=True, exist_ok=True)
    copilot_dest.write_text(copilot_content)

    # Validate sync was successful
    if not copilot_dest.exists():
        logger.error(f"ERROR: Failed to write {copilot_dest}")
        return

    logger.info(f"✓ Synced CLAUDE.md -> {copilot_dest}")
    logger.info(f"  Size: {len(copilot_content)} characters")
