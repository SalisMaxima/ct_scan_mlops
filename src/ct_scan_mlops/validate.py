"""Validate data integrity and structure."""

from pathlib import Path

import torch
from loguru import logger


def main():
    """Validate dataset integrity."""
    data_dir = Path("data/processed")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run 'invoke data.download' and 'invoke data.preprocess' first")
        return

    logger.info("Validating dataset...")
    logger.info("=" * 60)

    issues = []
    total_samples = 0

    # Expected classes
    expected_classes = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]

    # Check each split
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            issues.append(f"Missing {split} split directory")
            continue

        logger.info(f"\nValidating {split} split...")

        # Check all expected classes exist
        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                issues.append(f"{split}/{class_name} directory not found")
                continue

            # Check tensor files
            tensor_files = list(class_dir.glob("*.pt"))
            if not tensor_files:
                issues.append(f"{split}/{class_name} has no tensor files")
                continue

            # Validate a few random samples
            samples_to_check = min(5, len(tensor_files))
            for tensor_file in tensor_files[:samples_to_check]:
                try:
                    tensor = torch.load(tensor_file, weights_only=True)

                    # Check shape
                    if len(tensor.shape) != 3:
                        issues.append(f"{tensor_file.name}: Invalid shape {tensor.shape}, expected 3D (C, H, W)")

                    # Check data type
                    if tensor.dtype != torch.float32:
                        issues.append(f"{tensor_file.name}: Invalid dtype {tensor.dtype}, expected float32")

                    # Check for NaN/Inf
                    if torch.isnan(tensor).any():
                        issues.append(f"{tensor_file.name}: Contains NaN values")
                    if torch.isinf(tensor).any():
                        issues.append(f"{tensor_file.name}: Contains Inf values")

                except Exception as e:
                    issues.append(f"{tensor_file.name}: Failed to load - {e!s}")

            total_samples += len(tensor_files)
            logger.info(f"  ✓ {class_name}: {len(tensor_files)} samples")

    logger.info("\n" + "=" * 60)
    logger.info(f"Total samples: {total_samples}")

    if issues:
        logger.error(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            logger.error(f"  - {issue}")
    else:
        logger.success("\n✓ All validation checks passed!")


if __name__ == "__main__":
    main()
