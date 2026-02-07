"""Display model information (size, parameters, architecture)."""

import argparse
import sys
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(checkpoint_path):
    """Get model checkpoint file size."""
    size_bytes = Path(checkpoint_path).stat().st_size
    return size_bytes / (1024 * 1024)


def main():
    """Show model information."""
    parser = argparse.ArgumentParser(description="Display model information")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to Hydra config")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    cfg = OmegaConf.load(args.config)

    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)

    # Load model
    from ct_scan_mlops.analysis.core import load_model_from_checkpoint
    from ct_scan_mlops.utils import get_device

    logger.info(f"Loading model from {checkpoint_path}...")
    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    # Get file size
    file_size = get_model_size(checkpoint_path)

    # Print information
    logger.info(f"\nModel: {cfg.model.name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("\nParameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Non-trainable: {total_params - trainable_params:,}")

    logger.info(f"\nFile size: {file_size:.2f} MB")

    # Show model architecture
    logger.info("\nModel Architecture:")
    logger.info("-" * 60)
    logger.info(model)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
