"""Display dataset statistics."""

from pathlib import Path

import torch
from loguru import logger


def main():
    """Show dataset statistics."""
    data_dir = Path("data/processed")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run 'invoke data.download' and 'invoke data.preprocess' first")
        return

    logger.info("Dataset Statistics")
    logger.info("=" * 60)

    # Check each split
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"{split} split not found")
            continue

        # Count classes
        classes = {}
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                files = list(class_dir.glob("*.pt"))
                classes[class_dir.name] = len(files)

        total = sum(classes.values())
        logger.info(f"\n{split.upper()} Split ({total} samples):")
        for class_name, count in sorted(classes.items()):
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

        # Check tensor shape from first sample
        if classes:
            first_class = next(iter(classes.keys()))
            sample_files = list((split_dir / first_class).glob("*.pt"))
            if sample_files:
                sample = torch.load(sample_files[0], weights_only=True)
                logger.info(f"  Sample shape: {sample.shape}")
                logger.info(f"  Data type: {sample.dtype}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
