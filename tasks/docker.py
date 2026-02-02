"""Docker container and image management tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


def check_docker_available(ctx: Context) -> bool:
    """Check if Docker is installed and running."""
    try:
        result = ctx.run("docker info", hide=True, warn=True, pty=not WINDOWS)
        return result.ok
    except Exception:
        return False


@task
def build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images (CPU versions).

    Examples:
        invoke docker.build
        invoke docker.build --progress auto
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

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
def build_cuda(ctx: Context, progress: str = "plain") -> None:
    """Build CUDA-enabled training docker image.

    Examples:
        invoke docker.build-cuda
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

    ctx.run(
        f"docker build -t train-cuda:latest . -f dockerfiles/train_cuda.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(
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
        invoke docker.train
        invoke docker.train --no-cuda
        invoke docker.train --args "model=resnet18"
        invoke docker.train --entity other-username --args "train.max_epochs=10"
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

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
def api(ctx: Context, port: int = 8000) -> None:
    """Run API in Docker container.

    Examples:
        invoke docker.api
        invoke docker.api --port 8080
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

    ctx.run(f"docker run -p {port}:8000 -v $(pwd)/models:/app/models api:latest", echo=True, pty=not WINDOWS)


@task
def api_frontend(
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

    Examples:
        invoke docker.api-frontend
        invoke docker.api-frontend --model-dir outputs/models --config-path configs/prod.yaml
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

    cwd = Path.cwd()
    model_host = (cwd / model_dir).resolve()
    config_host = (cwd / config_path).resolve()

    if not model_host.exists():
        logger.error(f"ERROR: Model directory not found: {model_host}")
        return

    if not config_host.exists():
        logger.error(f"ERROR: Config file not found: {config_host}")
        return

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
        f"-e MODEL_PATH=/app/outputs/checkpoints/model.pt "
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


@task
def clean(ctx: Context, all: bool = False) -> None:
    """Clean up Docker images and containers.

    Args:
        all: Remove all unused images (not just dangling ones)

    Examples:
        invoke docker.clean              # Remove dangling images and stopped containers
        invoke docker.clean --all         # Remove all unused images
    """
    if not check_docker_available(ctx):
        logger.error("ERROR: Docker is not running. Please start Docker and try again.")
        return

    print("Removing stopped containers...")
    ctx.run("docker container prune -f", echo=True, pty=not WINDOWS)

    if all:
        print("Removing all unused images...")
        ctx.run("docker image prune -a -f", echo=True, pty=not WINDOWS)
    else:
        print("Removing dangling images...")
        ctx.run("docker image prune -f", echo=True, pty=not WINDOWS)

    print("Removing unused volumes...")
    ctx.run("docker volume prune -f", echo=True, pty=not WINDOWS)

    print("\n✓ Docker cleanup complete!")
