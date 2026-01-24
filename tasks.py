import os
from pathlib import Path

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
def extract_features(ctx: Context, n_jobs: int = -1, args: str = "") -> None:
    """Extract radiomics features from preprocessed images.

    Args:
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        args: Additional Hydra config overrides

    Examples:
        invoke extract-features
        invoke extract-features --n-jobs 4
        invoke extract-features --args "features.use_wavelet=false"
    """
    full_args = f"n_jobs={n_jobs} {args}".strip()
    ctx.run(f"uv run python -m {PROJECT_NAME}.features.extract_radiomics {full_args}", echo=True, pty=not WINDOWS)


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
def sweep(
    ctx: Context, sweep_config: str = "configs/sweeps/train_sweep.yaml", project: str = "", entity: str = ""
) -> None:
    """Create a W&B sweep from a sweep YAML.

    Args:
        sweep_config: Path to sweep yaml (default: configs/sweeps/train_sweep.yaml)
        project: Optional override for W&B project
        entity: Optional override for W&B entity

    Example:
        invoke sweep
        invoke sweep --entity your-username
    """
    cmd = f"uv run wandb sweep {sweep_config}"
    if project:
        cmd += f" --project {project}"
    if entity:
        cmd += f" --entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def sweep_agent(ctx: Context, sweep_id: str) -> None:
    """Run a W&B sweep agent.

    Args:
        sweep_id: The full sweep id, e.g. ENTITY/PROJECT/SWEEP_ID

    Example:
        invoke sweep-agent --sweep-id mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps/abc123
    """
    ctx.run(f"uv run wandb agent {sweep_id}", echo=True, pty=not WINDOWS)


@task
def sweep_best(ctx: Context, sweep_id: str, metric: str = "val_acc", goal: str = "maximize") -> None:
    """Print the best run (and its config) for a sweep.

    Args:
        sweep_id: The full sweep id, e.g. ENTITY/PROJECT/SWEEP_ID
        metric: Metric name to optimize (default: val_acc)
        goal: maximize | minimize

    Example:
        invoke sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
    """
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.sweep_best {sweep_id} --metric {metric} --goal {goal}",
        echo=True,
        pty=not WINDOWS,
    )


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
    ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS)
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
def docker_train(
    ctx: Context,
    entity: str = "mathiashl-danmarks-tekniske-universitet-dtu",
    cuda: bool = True,
    args: str = "",
) -> None:
    """Run training in Docker container.

    Args:
        entity: Your wandb username (default: mathiashl-danmarks-tekniske-universitet-dtu)
        cuda: Use CUDA-enabled image (default: True)
        args: Additional Hydra config overrides

    Examples:
        invoke docker-train
        invoke docker-train --no-cuda
        invoke docker-train --args "model=resnet18"
        invoke docker-train --entity other-username --args "train.max_epochs=10"
    """
    # Get absolute path to avoid Docker mount issues
    cwd = Path.cwd()

    image = "train-cuda:latest" if cuda else "train:latest"
    gpu_flag = "--gpus all" if cuda else ""
    train_args = f"wandb.entity={entity} {args}".strip()
    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    ctx.run(
        f"docker run --rm {gpu_flag} "
        f"--shm-size=2g "
        f"-v {cwd}/data:/app/data "
        f"-v {cwd}/models:/app/models "
        f"-e WANDB_API_KEY={wandb_api_key} {image} {train_args}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_api(ctx: Context, port: int = 8000) -> None:
    """Run API in Docker container."""
    ctx.run(f"docker run -p {port}:8000 -v $(pwd)/models:/app/models api:latest", echo=True, pty=not WINDOWS)


@task
def docker_api_frontend(
    ctx: Context,
    api_port: int = 8000,
    frontend_port: int = 8501,
    model_dir: str = "outputs/ct_scan_classifier/sweeps/a2mo2otc",
    config_path: str = "configs/config_production.yaml",
) -> None:
    """Build API image, run container, and launch Streamlit frontend.

    Args:
        api_port: API port to expose locally
        frontend_port: Streamlit port
        model_dir: Host path to the folder containing model.pt
        config_path: Host path to config YAML
    """
    cwd = Path.cwd()
    model_host = (cwd / model_dir).resolve()
    config_host = (cwd / config_path).resolve()

    ctx.run("docker build -t ct-scan-api . -f dockerfiles/api.dockerfile", echo=True, pty=not WINDOWS)

    model_mount = f'"{model_host}:/app/models"'
    config_mount = f'"{config_host.parent}:/app/configs"'
    api_container = "ct-scan-api-dev"

    ctx.run(
        f"docker rm -f {api_container} || exit 0",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        "docker run -d "
        f"--name {api_container} "
        f"-p {api_port}:8000 "
        f"-v {model_mount} "
        f"-v {config_mount} "
        f"-e MODEL_PATH=/app/models/model.pt "
        f"-e CONFIG_PATH=/app/configs/{config_host.name} "
        "ct-scan-api",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"uv run streamlit run src/{PROJECT_NAME}/frontend/pages/home.py --server.port {frontend_port}",
        echo=True,
        pty=not WINDOWS,
    )


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


@task
def branch(ctx: Context, name: str, message: str, files: str = ".") -> None:
    """Create a new branch, commit changes, and push to remote.

    Args:
        name: Branch name (e.g., "feature/new-feature" or "docs/readme-update")
        message: Commit message
        files: Files to add (default: "." for all changes)

    Examples:
        invoke branch --name feature/auth --message "Add authentication"
        invoke branch --name docs/readme --message "Update README" --files README.md
    """
    print("Running ruff to format and lint code...")
    ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)

    print("\nRunning pre-commit hooks to fix formatting issues...")
    ctx.run("uv run pre-commit run --all-files", echo=True, pty=not WINDOWS)

    print(f"\nCreating and switching to branch: {name}")
    ctx.run(f"git checkout -b {name}", echo=True, pty=not WINDOWS)

    print(f"\nAdding files: {files}")
    ctx.run(f"git add {files}", echo=True, pty=not WINDOWS)

    print("\nCommitting changes...")
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)

    print("\nPushing branch to remote...")
    ctx.run(f"git push -u origin {name}", echo=True, pty=not WINDOWS)

    print(f"\n✓ Branch '{name}' created and pushed!")
    print(f"   Create PR at: https://github.com/SalisMaxima/ct_scan_mlops/compare/{name}")


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


@task
def sync_ai_config(_ctx: Context) -> None:
    """Sync AI assistant config files from CLAUDE.md (source of truth).

    Updates .github/copilot-instructions.md to match CLAUDE.md content.
    """
    source = Path("CLAUDE.md")
    copilot_dest = Path(".github/copilot-instructions.md")

    if not source.exists():
        print("ERROR: CLAUDE.md not found")
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

    print(f"✓ Synced CLAUDE.md -> {copilot_dest}")
