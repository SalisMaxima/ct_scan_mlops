"""Test model performance from W&B Model Registry.

Following DTU MLOps course CML patterns:
https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/cml/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf

from ct_scan_mlops.model import build_model

# Performance thresholds
MAX_INFERENCE_TIME_MS = 100  # Maximum acceptable inference time in milliseconds
MIN_ACCURACY = 0.70  # Minimum acceptable accuracy (if metadata available)


def download_model_from_wandb(model_path: str, project: str) -> tuple[Path, dict]:
    """Download model artifact from W&B.

    Args:
        model_path: W&B artifact path (e.g., entity/project/model:alias)
        project: W&B project name

    Returns:
        Tuple of (local model path, artifact metadata)
    """
    logger.info(f"Downloading model from W&B: {model_path}")

    # Initialize wandb for artifact download
    run = wandb.init(project=project, job_type="model-testing", mode="online")

    try:
        artifact = run.use_artifact(model_path, type="model")
        artifact_dir = Path(artifact.download())
        logger.info(f"Model downloaded to: {artifact_dir}")

        # Get metadata
        metadata = artifact.metadata or {}
        logger.info(f"Model metadata: {metadata}")

        # Find model file (could be .pt or .ckpt)
        model_files = list(artifact_dir.glob("*.pt")) + list(artifact_dir.glob("*.ckpt"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {artifact_dir}")

        # Prefer best checkpoint if available
        best_ckpt = [f for f in model_files if "best" in f.name.lower()]
        model_file = best_ckpt[0] if best_ckpt else model_files[0]
        logger.info(f"Using model file: {model_file}")

        return model_file, metadata
    finally:
        run.finish()


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)

    # Build model
    model = build_model(cfg).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            # Lightning checkpoint format
            state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("model."):
                    state_dict[key[6:]] = value
                else:
                    state_dict[key] = value
            model.load_state_dict(state_dict)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def test_inference_speed(
    model: torch.nn.Module, device: torch.device, image_size: int = 224, n_samples: int = 100
) -> float:
    """Test model inference speed.

    Following course pattern from cml.md.

    Args:
        model: Model to test
        device: Device to run inference on
        image_size: Input image size (default: 224)
        n_samples: Number of inference runs

    Returns:
        Average inference time in milliseconds
    """
    # Create sample input (matching expected input size)
    x = torch.randn(1, 3, image_size, image_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)

    # Time inference
    start = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            model(x)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / n_samples) * 1000
    logger.info(f"Average inference time: {avg_time_ms:.2f}ms over {n_samples} samples")

    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description="Test model from W&B Model Registry")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="W&B artifact path (e.g., entity/project/model:alias)",
    )
    parser.add_argument(
        "--max-inference-time",
        type=float,
        default=MAX_INFERENCE_TIME_MS,
        help=f"Maximum acceptable inference time in ms (default: {MAX_INFERENCE_TIME_MS})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: from config.yaml)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml relative to project root)",
    )
    args = parser.parse_args()

    # Load config
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"

    cfg = OmegaConf.load(config_path)

    # Get wandb project and image size from config
    wandb_project = args.wandb_project or cfg.wandb.project
    image_size = cfg.data.image_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Download model from W&B
    model_file, metadata = download_model_from_wandb(args.model_path, wandb_project)

    # Load model
    model = load_model(model_file, device)

    # Test inference speed
    avg_time_ms = test_inference_speed(model, device, image_size)

    # Check performance threshold
    if avg_time_ms > args.max_inference_time:
        logger.error(f"Inference too slow: {avg_time_ms:.2f}ms > {args.max_inference_time}ms threshold")
        sys.exit(1)

    # Check accuracy from metadata if available
    if "val_acc" in metadata:
        val_acc = metadata["val_acc"]
        logger.info(f"Model validation accuracy: {val_acc:.4f}")
        if val_acc < MIN_ACCURACY:
            logger.error(f"Accuracy too low: {val_acc:.4f} < {MIN_ACCURACY} threshold")
            sys.exit(1)

    logger.success("Model passed all performance tests!")
    logger.info(f"  - Inference time: {avg_time_ms:.2f}ms (threshold: {args.max_inference_time}ms)")
    if "val_acc" in metadata:
        logger.info(f"  - Validation accuracy: {metadata['val_acc']:.4f}")


if __name__ == "__main__":
    main()
