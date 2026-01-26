"""Region-based features for CT images.

Scientific Rationale:
- Ground-glass opacity ratio is critical for adenocarcinoma subtyping
- Cavitation is present in up to 82% of squamous cell carcinomas
- Solid component ratio correlates with invasiveness in adenocarcinoma
- Low-density central regions indicate necrosis (LCC, advanced SCC)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def extract_region_features(
    image: np.ndarray,
    low_threshold: float = 0.2,
    ggo_threshold: float = 0.5,
    solid_threshold: float = 0.7,
) -> dict[str, float]:
    """Extract region-based features for GGO/solid/cavity detection.

    Args:
        image: 2D numpy array (H, W) representing the CT image.
        low_threshold: Threshold for low-density (air/cavity) regions.
        ggo_threshold: Threshold for ground-glass opacity regions.
        solid_threshold: Threshold for solid tissue regions.

    Returns:
        Dictionary with 6 region features:
        - solid_ratio: High-intensity pixels / total (solid vs GGO balance)
        - ggo_ratio: Mid-intensity pixels / total (ground-glass component)
        - low_density_ratio: Low-intensity pixels (cavitation/necrosis)
        - mean_gradient: Average edge gradient magnitude
        - max_gradient: Maximum edge gradient magnitude
        - boundary_irregularity: Std of edge gradients (spiculation proxy)
    """
    # Normalize image to 0-1
    img_min, img_max = image.min(), image.max()
    if img_max - img_min < 1e-10:
        # Constant image
        return {
            "solid_ratio": 0.0,
            "ggo_ratio": 0.0,
            "low_density_ratio": 1.0,
            "mean_gradient": 0.0,
            "max_gradient": 0.0,
            "boundary_irregularity": 0.0,
        }

    img_norm = (image - img_min) / (img_max - img_min)
    total_pixels = img_norm.size

    # Count pixels in different density regions
    low_density_pixels = np.sum(img_norm < low_threshold)
    ggo_pixels = np.sum((img_norm >= low_threshold) & (img_norm < ggo_threshold))
    solid_pixels = np.sum(img_norm >= solid_threshold)

    # Compute gradient for margin analysis
    gradient_x = ndimage.sobel(img_norm, axis=0)
    gradient_y = ndimage.sobel(img_norm, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Boundary irregularity (standard deviation of edge gradients)
    edge_threshold = np.percentile(gradient_magnitude, 90)
    edge_mask = gradient_magnitude > edge_threshold
    boundary_irregularity = float(np.std(gradient_magnitude[edge_mask])) if edge_mask.any() else 0.0

    return {
        "solid_ratio": float(solid_pixels / total_pixels),
        "ggo_ratio": float(ggo_pixels / total_pixels),
        "low_density_ratio": float(low_density_pixels / total_pixels),
        "mean_gradient": float(np.mean(gradient_magnitude)),
        "max_gradient": float(np.max(gradient_magnitude)),
        "boundary_irregularity": boundary_irregularity,
    }
