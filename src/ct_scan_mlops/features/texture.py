"""GLCM (Gray-Level Co-occurrence Matrix) texture features for CT images.

Scientific Rationale:
- GLCM captures spatial relationships between pixels
- Studies show GLCM features achieve AUC of 0.81 for ADC vs SCC differentiation
- SCC tends toward intranodular homogeneity; adenocarcinoma shows heterogeneity
- Haralick features are robust across imaging conditions
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    image: np.ndarray,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    levels: int = 256,
) -> dict[str, float]:
    """Extract GLCM texture features from CT image.

    Args:
        image: 2D numpy array (H, W) representing the CT image.
        distances: List of pixel distances for GLCM (default: [1, 2, 3]).
        angles: List of angles in radians (default: [0, pi/4, pi/2, 3pi/4]).
        levels: Number of gray levels for GLCM (default: 256).

    Returns:
        Dictionary with 13 GLCM features:
        - glcm_contrast_mean/std: Local intensity variation
        - glcm_homogeneity_mean/std: Closeness to diagonal
        - glcm_energy_mean/std: Uniformity measure
        - glcm_correlation_mean/std: Linear dependency
        - glcm_dissimilarity_mean/std: Edge/boundary detection
        - glcm_ASM_mean/std: Angular Second Moment (orderliness)
        - glcm_entropy: Randomness measure
    """
    if distances is None:
        distances = [1, 2, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Normalize to 0-255 range for GLCM
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-10:
        # Constant image - return zeros
        return {
            "glcm_contrast_mean": 0.0,
            "glcm_contrast_std": 0.0,
            "glcm_homogeneity_mean": 1.0,
            "glcm_homogeneity_std": 0.0,
            "glcm_energy_mean": 1.0,
            "glcm_energy_std": 0.0,
            "glcm_correlation_mean": 0.0,
            "glcm_correlation_std": 0.0,
            "glcm_dissimilarity_mean": 0.0,
            "glcm_dissimilarity_std": 0.0,
            "glcm_ASM_mean": 1.0,
            "glcm_ASM_std": 0.0,
            "glcm_entropy": 0.0,
        }

    image_uint8 = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        image_uint8,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    features: dict[str, float] = {}
    properties = ["contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"]

    for prop in properties:
        values = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))

    # Compute entropy separately
    glcm_normalized = glcm.astype(np.float64) / (glcm.sum() + 1e-10)
    # Avoid log(0) by adding small epsilon
    glcm_nonzero = glcm_normalized + 1e-10
    entropy = -np.sum(glcm_normalized * np.log2(glcm_nonzero))
    features["glcm_entropy"] = float(entropy)

    return features
