import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"
PYTHON_VERSION = "3.12"


# Environment commands
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


# Check python path and version
@task
def python(ctx):
    """Check Python path and version."""
    ctx.run("which python" if os.name != "nt" else "where python")
    ctx.run("python --version")


# Project commands
@task
def download_data(ctx: Context) -> None:
    """Download CT scan dataset from Kaggle."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data download", echo=True, pty=not WINDOWS)


@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data (resize, normalize, save as tensors)."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data preprocess", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, entity: str = "", args: str = "") -> None:
    """Train model with wandb logging.

    Args:
        entity: Wandb entity (optional, defaults to team entity from config)
        args: Hydra config overrides (e.g., "train.max_epochs=100 model=resnet18")

    Examples:
        invoke train
        invoke train --args "model=resnet18"
        invoke train --args "train.max_epochs=100"
        invoke train --entity your-personal-username  # Override default team entity
        invoke train --args "wandb.mode=disabled"  # Skip wandb
    """
    # Build command with entity override if provided
    # If no entity provided, config.yaml default (team entity) will be used
    entity_override = f"wandb.entity={entity}" if entity else ""
    full_args = f"{entity_override} {args}".strip()

    ctx.run(f"uv run python -m {PROJECT_NAME}.train {full_args}", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, checkpoint: str = "", wandb: bool = False, entity: str = "") -> None:
    """Evaluate model on test set.

    Args:
        checkpoint: Path to model checkpoint (.ckpt or .pt)
        wandb: Log results to W&B
        entity: Your wandb username (for W&B logging)

    Examples:
        invoke evaluate --checkpoint outputs/ct_scan_classifier/2026-01-14/12-34-56/best_model.ckpt
        invoke evaluate --checkpoint outputs/.../model.pt --wandb --entity your-username
    """
    if not checkpoint:
        print("ERROR: --checkpoint is required")
        print("Usage: invoke evaluate --checkpoint path/to/best_model.ckpt")
        print("Example: invoke evaluate --checkpoint outputs/ct_scan_classifier/2026-01-14/12-34-56/best_model.ckpt")
        return

    cmd = f"uv run python -m {PROJECT_NAME}.evaluate {checkpoint}"
    if wandb:
        cmd += " --wandb"
        if entity:
            cmd += f" --wandb-entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


# Code quality commands
@task
def ruff(ctx: Context) -> None:
    """Run ruff check and format."""
    ctx.run("uv run ruff check .", echo=True, pty=not WINDOWS)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff linter.

    Args:
        fix: Auto-fix issues where possible
    """
    fix_flag = " --fix" if fix else ""
    ctx.run(f"uv run ruff check .{fix_flag}", echo=True, pty=not WINDOWS)


@task
def format(ctx: Context, check: bool = False) -> None:
    """Run ruff formatter.

    Args:
        check: Only check, don't modify files
    """
    check_flag = " --check" if check else ""
    ctx.run(f"uv run ruff format .{check_flag}", echo=True, pty=not WINDOWS)


# Docker commands
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images (CPU versions)."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_cuda(ctx: Context, progress: str = "plain") -> None:
    """Build CUDA-enabled training docker image."""
    ctx.run(
        f"docker build -t train-cuda:latest . -f dockerfiles/train_cuda.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_train(ctx: Context, entity: str = "", cuda: bool = True, args: str = "") -> None:
    """Run training in Docker container.

    Args:
        entity: Your wandb username (required for logging)
        cuda: Use CUDA-enabled image (default: True)
        args: Additional Hydra config overrides

    Examples:
        invoke docker-train --entity your-wandb-username
        invoke docker-train --entity your-wandb-username --args "model=resnet18"
    """
    if not entity:
        print("ERROR: --entity is required for wandb logging.")
        print("Usage: invoke docker-train --entity YOUR_WANDB_USERNAME")
        return

    image = "train-cuda:latest" if cuda else "train:latest"
    gpu_flag = "--gpus all" if cuda else ""
    train_args = f"wandb.entity={entity} {args}".strip()
    ctx.run(
        f"docker run {gpu_flag} -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models "
        f"-e WANDB_API_KEY=$WANDB_API_KEY {image} {train_args}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_api(ctx: Context, port: int = 8000) -> None:
    """Run API in Docker container."""
    ctx.run(f"docker run -p {port}:8000 -v $(pwd)/models:/app/models api:latest", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Git commands
@task
def git_status(ctx: Context) -> None:
    """Show git status."""
    ctx.run("git status", echo=True, pty=not WINDOWS)


@task
def git(ctx: Context, message: str) -> None:
    """Commit and push changes to git."""
    ctx.run("git add .", echo=True, pty=not WINDOWS)
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)
    ctx.run("git push", echo=True, pty=not WINDOWS)


# DVC commands
@task
def dvc_pull(ctx: Context) -> None:
    """Pull data from DVC remote."""
    ctx.run("dvc pull", echo=True, pty=not WINDOWS)


@task
def dvc_push(ctx: Context) -> None:
    """Push data to DVC remote."""
    ctx.run("dvc push", echo=True, pty=not WINDOWS)


@task
def dvc_add(ctx: Context, folder: str, message: str) -> None:
    """Add data to DVC and push to remote storage.

    Args:
        folder: Path to the folder or file to add to DVC
        message: Commit message for the changes

    Example:
        invoke dvc-add --folder data/raw --message "Add new training data"
    """
    print(f"Adding {folder} to DVC...")
    ctx.run(f"dvc add {folder}", echo=True, pty=not WINDOWS)

    print("\nStaging DVC files in git...")
    ctx.run(f"git add {folder}.dvc .gitignore", echo=True, pty=not WINDOWS)

    print("\nCommitting changes...")
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)

    print("\nPushing to git remote...")
    ctx.run("git push", echo=True, pty=not WINDOWS)

    print("\nPushing data to DVC remote...")
    ctx.run("dvc push", echo=True, pty=not WINDOWS)

    print(f"\n✓ Successfully added {folder} to DVC and pushed to remotes!")


# API commands
@task
def api(ctx: Context, reload: bool = True, port: int = 8000) -> None:
    """Run FastAPI development server.

    Args:
        reload: Enable auto-reload on code changes
        port: Port to run the server on
    """
    reload_flag = " --reload" if reload else ""
    ctx.run(
        f"uv run uvicorn {PROJECT_NAME}.api:app --host 0.0.0.0 --port {port}{reload_flag}", echo=True, pty=not WINDOWS
    )
