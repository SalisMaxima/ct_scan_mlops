"""Display model information (size, parameters, architecture)."""

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from ct_scan_mlops.models import load_model


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(checkpoint_path):
    """Get model checkpoint file size."""
    size_bytes = Path(checkpoint_path).stat().st_size
    return size_bytes / (1024 * 1024)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Show model information."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_model(cfg.model, checkpoint_path)

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    # Get file size
    file_size = get_model_size(checkpoint_path)

    # Print information
    logger.info(f"\nModel: {cfg.model._target_}")
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
