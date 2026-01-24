"""Main FeatureExtractor class for radiomics feature extraction.

This module provides a unified interface for extracting hand-crafted radiomics
features from CT images, combining intensity, texture, shape, region, and
wavelet features into a single feature vector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from .intensity import extract_intensity_features
from .region import extract_region_features
from .shape import extract_shape_features
from .texture import extract_glcm_features
from .wavelet import extract_wavelet_features

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    Attributes:
        use_intensity: Whether to extract intensity/histogram features.
        use_glcm: Whether to extract GLCM texture features.
        use_shape: Whether to extract shape/morphological features.
        use_region: Whether to extract region-based features.
        use_wavelet: Whether to extract wavelet features.
        glcm_distances: Pixel distances for GLCM computation.
        wavelet_type: Type of wavelet for decomposition.
        wavelet_level: Number of wavelet decomposition levels.
    """

    use_intensity: bool = True
    use_glcm: bool = True
    use_shape: bool = True
    use_region: bool = True
    use_wavelet: bool = True
    glcm_distances: list[int] = field(default_factory=lambda: [1, 2, 3])
    wavelet_type: str = "db4"
    wavelet_level: int = 2

    @classmethod
    def from_dict(cls, cfg: dict) -> FeatureConfig:
        """Create FeatureConfig from a dictionary."""
        return cls(
            use_intensity=cfg.get("use_intensity", True),
            use_glcm=cfg.get("use_glcm", True),
            use_shape=cfg.get("use_shape", True),
            use_region=cfg.get("use_region", True),
            use_wavelet=cfg.get("use_wavelet", True),
            glcm_distances=list(cfg.get("glcm_distances", [1, 2, 3])),
            wavelet_type=cfg.get("wavelet_type", "db4"),
            wavelet_level=cfg.get("wavelet_level", 2),
        )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> FeatureConfig:
        """Create FeatureConfig from Hydra config."""
        features_cfg = cfg.get("features", {})
        return cls.from_dict(dict(features_cfg))


class FeatureExtractor:
    """Extract hand-crafted radiomics features from CT images.

    This class provides a unified interface for extracting features from
    individual images or batches of images, with support for parallel
    processing and caching.

    Example:
        >>> extractor = FeatureExtractor()
        >>> image = np.random.rand(224, 224).astype(np.float32)
        >>> features = extractor.extract(image)
        >>> print(f"Extracted {len(features)} features")
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the feature extractor.

        Args:
            config: Feature extraction configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig()
        self._feature_dim: int | None = None
        self._feature_names: list[str] | None = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract all configured features from a single image.

        Args:
            image: 2D numpy array (H, W) or 3D (C, H, W).

        Returns:
            1D numpy array of extracted features.
        """
        # Handle channel dimension
        if image.ndim == 3:
            # Convert to grayscale (average channels or use first channel)
            image = np.mean(image, axis=0) if image.shape[0] == 3 else image[0]

        # Ensure float type
        image = image.astype(np.float32)

        features: dict[str, float] = {}

        if self.config.use_intensity:
            features.update(extract_intensity_features(image))

        if self.config.use_glcm:
            features.update(
                extract_glcm_features(
                    image,
                    distances=self.config.glcm_distances,
                )
            )

        if self.config.use_shape:
            features.update(extract_shape_features(image))

        if self.config.use_region:
            features.update(extract_region_features(image))

        if self.config.use_wavelet:
            features.update(
                extract_wavelet_features(
                    image,
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level,
                )
            )

        # Convert to array
        feature_array = np.array(list(features.values()), dtype=np.float32)

        # Handle NaN/Inf
        return np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_batch(
        self,
        images: torch.Tensor | np.ndarray,
        n_jobs: int = -1,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Extract features from a batch of images with parallel processing.

        Args:
            images: Tensor or array of shape (B, C, H, W) or (B, H, W).
            n_jobs: Number of parallel jobs (-1 = all CPUs).
            show_progress: Whether to show progress bar.

        Returns:
            Tensor of shape (B, num_features).
        """
        images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images

        if n_jobs == 1 or len(images_np) == 1:
            # Sequential processing
            batch_features = []
            iterator = tqdm(images_np, desc="Extracting features") if show_progress else images_np
            for img in iterator:
                features = self.extract(img)
                batch_features.append(features)
        else:
            # Parallel processing
            batch_features = Parallel(n_jobs=n_jobs)(
                delayed(self.extract)(img)
                for img in tqdm(images_np, desc="Extracting features", disable=not show_progress)
            )

        return torch.tensor(np.stack(batch_features), dtype=torch.float32)

    @property
    def feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        if self._feature_dim is None:
            # Extract from dummy image to determine dimension
            rng = np.random.default_rng(42)
            dummy = rng.random((224, 224)).astype(np.float32)
            self._feature_dim = len(self.extract(dummy))
        return self._feature_dim

    def get_feature_names(self) -> list[str]:
        """Get names of all features being extracted.

        Returns:
            List of feature names in the order they appear in the feature vector.
        """
        if self._feature_names is not None:
            return self._feature_names

        rng = np.random.default_rng(42)
        dummy = rng.random((224, 224)).astype(np.float32)
        features: dict[str, float] = {}

        if self.config.use_intensity:
            features.update(extract_intensity_features(dummy))
        if self.config.use_glcm:
            features.update(extract_glcm_features(dummy, distances=self.config.glcm_distances))
        if self.config.use_shape:
            features.update(extract_shape_features(dummy))
        if self.config.use_region:
            features.update(extract_region_features(dummy))
        if self.config.use_wavelet:
            features.update(
                extract_wavelet_features(
                    dummy,
                    wavelet=self.config.wavelet_type,
                    level=self.config.wavelet_level,
                )
            )

        self._feature_names = list(features.keys())
        return self._feature_names

    def save_metadata(self, output_path: Path) -> None:
        """Save feature metadata to JSON file.

        Args:
            output_path: Path to save the metadata JSON file.
        """
        metadata = {
            "feature_names": self.get_feature_names(),
            "feature_dim": self.feature_dim,
            "config": {
                "use_intensity": self.config.use_intensity,
                "use_glcm": self.config.use_glcm,
                "use_shape": self.config.use_shape,
                "use_region": self.config.use_region,
                "use_wavelet": self.config.use_wavelet,
                "glcm_distances": self.config.glcm_distances,
                "wavelet_type": self.config.wavelet_type,
                "wavelet_level": self.config.wavelet_level,
            },
        }
        with output_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature metadata to {output_path}")


def extract_and_save_features(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    split: str,
    extractor: FeatureExtractor | None = None,
    n_jobs: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Extract features from images and save to disk.

    Args:
        images: Image tensor of shape (N, C, H, W).
        labels: Label tensor of shape (N,).
        output_path: Directory to save features.
        split: Dataset split name ('train', 'valid', 'test').
        extractor: Feature extractor (creates new one if None).
        n_jobs: Number of parallel jobs for extraction.

    Returns:
        Tuple of (features tensor, normalization stats dict).
    """
    if extractor is None:
        extractor = FeatureExtractor()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting features for {split} split ({len(images)} images)...")
    features = extractor.extract_batch(images, n_jobs=n_jobs)

    # Compute normalization stats from training data
    stats: dict = {}
    if split == "train":
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        # Avoid division by zero
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        stats["mean"] = mean.tolist()
        stats["std"] = std.tolist()
        logger.info("Computed normalization stats from training features")

    # Save features
    torch.save(features, output_path / f"{split}_features.pt")
    torch.save(labels, output_path / f"{split}_labels.pt")
    logger.info(f"Saved {split} features: {features.shape}")

    return features, stats
