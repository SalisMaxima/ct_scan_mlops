"""CLI for extracting radiomics features from CT images.

This script extracts hand-crafted radiomics features from preprocessed
CT images and saves them as PyTorch tensors for use in training.

Example:
    # Extract features with default config
    python -m ct_scan_mlops.features.extract_radiomics

    # Extract features with custom config
    python -m ct_scan_mlops.features.extract_radiomics features.use_wavelet=false

    # Specify number of parallel jobs
    python -m ct_scan_mlops.features.extract_radiomics n_jobs=4
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from .extractor import FeatureConfig, FeatureExtractor


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract radiomics features from preprocessed CT images."""
    # Get paths
    processed_path = Path(cfg.data.get("processed_path", "data/processed"))
    output_path = Path(cfg.data.get("features_path", "data/processed"))

    # Get feature extraction config
    features_cfg = cfg.get("features", {})
    feature_config = FeatureConfig.from_dict(dict(features_cfg))

    # Get processing options
    n_jobs = cfg.get("n_jobs", -1)

    logger.info("Radiomics Feature Extraction")
    logger.info(f"  Input: {processed_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Parallel jobs: {n_jobs}")
    logger.info(f"  Feature config: {feature_config}")

    # Create extractor
    extractor = FeatureExtractor(config=feature_config)
    logger.info(f"  Feature dimension: {extractor.feature_dim}")
    logger.info(f"  Features: {extractor.get_feature_names()}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Track normalization stats (computed from training data)
    norm_stats: dict = {}

    # Process each split
    for split in ["train", "valid", "test"]:
        images_file = processed_path / f"{split}_images.pt"

        if not images_file.exists():
            logger.warning(f"Skipping {split}: {images_file} not found")
            continue

        logger.info(f"Processing {split} split...")

        # Load preprocessed images
        images = torch.load(images_file, weights_only=True)
        logger.info(f"  Loaded {len(images)} images of shape {images.shape[1:]}")

        # Extract features
        features = extractor.extract_batch(images, n_jobs=n_jobs, show_progress=True)

        # Compute normalization stats from training data
        if split == "train":
            mean = features.mean(dim=0)
            std = features.std(dim=0)
            # Avoid division by zero
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            norm_stats["mean"] = mean.tolist()
            norm_stats["std"] = std.tolist()
            logger.info("  Computed normalization stats from training features")

        # Save features
        torch.save(features, output_path / f"{split}_features.pt")
        logger.info(f"  Saved {split}_features.pt: shape {features.shape}")

    # Save feature metadata
    metadata = {
        "feature_names": extractor.get_feature_names(),
        "feature_dim": extractor.feature_dim,
        "normalization": norm_stats,
        "config": {
            "use_intensity": feature_config.use_intensity,
            "use_glcm": feature_config.use_glcm,
            "use_shape": feature_config.use_shape,
            "use_region": feature_config.use_region,
            "use_wavelet": feature_config.use_wavelet,
            "glcm_distances": feature_config.glcm_distances,
            "wavelet_type": feature_config.wavelet_type,
            "wavelet_level": feature_config.wavelet_level,
        },
    }

    metadata_path = output_path / "feature_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata to {metadata_path}")

    logger.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
