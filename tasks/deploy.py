"""Deployment and serving tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def promote_model(ctx: Context, model_path: str = "", project: str = "", entity: str = "") -> None:
    """Promote model to production in W&B Model Registry.

    Args:
        model_path: Path to model checkpoint to promote
        project: W&B project name (optional, uses config default)
        entity: W&B entity name (optional, uses config default)

    Examples:
        invoke deploy.promote-model --model-path outputs/best_model.ckpt
        invoke deploy.promote-model --model-path outputs/model.pt --project my-project
    """
    if not model_path:
        logger.error("ERROR: --model-path is required")
        return

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"ERROR: Model not found: {model_path}")
        return

    cmd = f"uv run python -m {PROJECT_NAME}.promote_model --model-path {model_path}"

    if project:
        cmd += f" --wandb-project {project}"
    if entity:
        cmd += f" --wandb-entity {entity}"

    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def export_onnx(ctx: Context, run_dir: str = "", output: str = "", opset: int = 17) -> None:
    """Export trained model to ONNX format for production deployment.

    Args:
        run_dir: Path to training run directory (contains model.pt and config)
        output: Output path for ONNX file (optional)
        opset: ONNX opset version (default: 17)

    Examples:
        invoke deploy.export-onnx --run-dir outputs/ct_scan_classifier/run_123
        invoke deploy.export-onnx --run-dir outputs/model_dir --output model.onnx
    """
    if not run_dir:
        logger.error("ERROR: --run-dir is required")
        return

    run_dir_obj = Path(run_dir)
    if not run_dir_obj.exists():
        logger.error(f"ERROR: Run directory not found: {run_dir}")
        return

    cmd = f"uv run python -m {PROJECT_NAME}.onnx_export --run-dir {run_dir}"

    if output:
        cmd += f" --output {output}"
    if opset != 17:
        cmd += f" --opset {opset}"

    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def api(ctx: Context, reload: bool = True, port: int = 8000) -> None:
    """Run FastAPI development server.

    Args:
        reload: Enable auto-reload on code changes
        port: Port to run the server on

    Examples:
        invoke deploy.api
        invoke deploy.api --port 8080 --no-reload
    """
    reload_flag = " --reload" if reload else ""
    ctx.run(
        f"uv run uvicorn {PROJECT_NAME}.api:app --host 0.0.0.0 --port {port}{reload_flag}", echo=True, pty=not WINDOWS
    )


@task
def frontend(ctx: Context, port: int = 8501, api_url: str = "http://localhost:8000") -> None:
    """Run Streamlit frontend application.

    Args:
        port: Port to run Streamlit on
        api_url: URL of the backend API

    Examples:
        invoke deploy.frontend
        invoke deploy.frontend --port 8080 --api-url http://localhost:5000
    """
    import os

    os.environ["API_URL"] = api_url
    ctx.run(
        f"uv run streamlit run src/{PROJECT_NAME}/frontend/pages/home.py --server.port {port}",
        echo=True,
        pty=not WINDOWS,
    )
