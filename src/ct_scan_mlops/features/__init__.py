"""Radiomics feature extraction module for chest CT classification.

This module provides hand-crafted radiomics features that complement CNN-based
classification for lung cancer subtype differentiation:
- Adenocarcinoma (peripheral, ground-glass opacity patterns)
- Squamous Cell Carcinoma (central, cavitation-prone)
- Large Cell Carcinoma (peripheral, necrotic, aggressive)
- Normal Lung Tissue (homogeneous, low-density parenchyma)

Features are grouped into five categories:
1. Intensity/Histogram features (8 features)
2. GLCM Texture features (13 features)
3. Shape/Morphological features (9 features)
4. Region-based features (6 features)
5. Wavelet features (14 features, optional)
"""

from __future__ import annotations

from .extractor import FeatureConfig, FeatureExtractor
from .intensity import extract_intensity_features
from .region import extract_region_features
from .shape import extract_shape_features
from .texture import extract_glcm_features
from .wavelet import extract_wavelet_features

__all__ = [
    "FeatureExtractor",
    "FeatureConfig",
    "extract_intensity_features",
    "extract_glcm_features",
    "extract_shape_features",
    "extract_region_features",
    "extract_wavelet_features",
]
