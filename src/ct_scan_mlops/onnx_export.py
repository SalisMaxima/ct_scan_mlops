from pathlib import Path

import torch
import typer
from loguru import logger
from omegaconf import OmegaConf

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cpu")

app = typer.Typer()


def find_most_recent_run(experiment_name: str = "ct_scan_classifier") -> Path | None:
    """Find the most recent training run directory.

    Args:
        experiment_name: Name of the experiment (default: ct_scan_classifier)

    Returns:
        Path to most recent run directory, or None if not found
    """
    output_base = Path("outputs") / experiment_name

    if not output_base.exists():
        logger.error(f"Output directory not found: {output_base}")
        return None

    # Find most recent run
    run_dirs = sorted(output_base.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_dirs:
        return run_dirs[0]

    logger.error(f"No training runs found in {output_base}")
    return None


@app.command()
def main(
    run_dir: str = typer.Option(
        None,
        "--run-dir",
        "-r",
        envvar="RUN_DIR",
        help="Path to training run directory (default: auto-detect most recent)",
    ),
    experiment_name: str = typer.Option(
        "ct_scan_classifier",
        "--experiment",
        "-e",
        help="Experiment name for auto-detection (default: ct_scan_classifier)",
    ),
    output_path: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Custom output path for ONNX model (default: run_dir/model.onnx)",
    ),
):
    """Export a trained PyTorch model to ONNX format.

    Examples:
        # Export most recent run
        python -m ct_scan_mlops.onnx_export

        # Export specific run
        python -m ct_scan_mlops.onnx_export --run-dir outputs/ct_scan_classifier/2026-01-15/11-57-23

        # Using environment variable
        RUN_DIR=outputs/ct_scan_classifier/2026-01-15/11-57-23 python -m ct_scan_mlops.onnx_export

        # Custom output path
        python -m ct_scan_mlops.onnx_export --output models/model.onnx
    """
    # Determine run directory
    if run_dir:
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            logger.error(f"Specified run directory does not exist: {run_dir_path}")
            raise typer.Exit(1)
        logger.info(f"Using specified run directory: {run_dir_path}")
    else:
        logger.info(f"Auto-detecting most recent run from experiment: {experiment_name}")
        run_dir_path = find_most_recent_run(experiment_name)
        if run_dir_path is None:
            logger.error("Could not find run directory. Use --run-dir to specify manually.")
            raise typer.Exit(1)
        logger.info(f"Found most recent run: {run_dir_path}")

    # Set paths
    config_path = run_dir_path / ".hydra" / "config.yaml"
    model_path = run_dir_path / "model.pt"

    if output_path:
        onnx_path = Path(output_path)
        # Ensure output directory exists
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        onnx_path = run_dir_path / "model.onnx"

    # Validate paths
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise typer.Exit(1)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise typer.Exit(1)

    # Load config and model
    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Disable pretrained weights - we're loading trained weights from model.pt
    # This prevents unnecessary network calls during export
    if OmegaConf.select(cfg, "model.pretrained") is not None:
        cfg.model.pretrained = False

    logger.info(f"Loading model from: {model_path}")
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    model.to(DEVICE)

    # Export to ONNX
    image_size = cfg.data.image_size
    input_channels = OmegaConf.select(cfg, "model.input_channels", default=3)
    dummy_input = torch.randn(1, input_channels, image_size, image_size, device=DEVICE)

    logger.info(f"Exporting model to ONNX: {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
        )

    logger.success(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    app()
