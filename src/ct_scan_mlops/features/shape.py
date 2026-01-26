"""Shape/morphological features for CT images.

Scientific Rationale:
- Sphericity correlates with malignancy (irregular shapes = higher malignancy)
- Spiculated margins are characteristic of adenocarcinoma
- Compactness helps identify infiltrative growth patterns
- Shape features combined with other features achieved AUC of 0.92 in studies
"""

from __future__ import annotations

import numpy as np
from skimage import filters, measure


def extract_shape_features(
    image: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Extract shape/morphological features from CT image.

    Since we don't have segmentation masks, we use Otsu thresholding to
    identify the primary region of interest.

    Args:
        image: 2D numpy array (H, W) representing the CT image.
        threshold: Manual threshold for binarization (default: Otsu's method).

    Returns:
        Dictionary with 9 shape features:
        - area: Pixel count (size measure)
        - perimeter: Boundary length (edge complexity)
        - eccentricity: Elongation from fitted ellipse
        - solidity: Area / ConvexHullArea (fill ratio)
        - extent: Area / BoundingBoxArea (spread measure)
        - major_axis_length: Fitted ellipse major axis
        - minor_axis_length: Fitted ellipse minor axis
        - compactness: P^2 / (4*pi*A) (circularity measure)
        - sphericity: (4*pi*A) / P^2 (roundness)
    """
    # Default features for empty/degenerate cases
    default_features = {
        "area": 0.0,
        "perimeter": 0.0,
        "eccentricity": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "major_axis_length": 0.0,
        "minor_axis_length": 0.0,
        "compactness": 0.0,
        "sphericity": 0.0,
    }

    # Handle edge case of constant image
    if np.std(image) < 1e-10:
        return default_features

    # Binarize image using Otsu's threshold
    if threshold is None:
        threshold = filters.threshold_otsu(image)

    binary = image > threshold

    # Label connected components
    labeled = measure.label(binary)

    if labeled.max() == 0:
        # No regions found
        return default_features

    # Get region properties
    regions = measure.regionprops(labeled)
    if not regions:
        return default_features

    # Get largest region
    largest = max(regions, key=lambda r: r.area)

    area = float(largest.area)
    perimeter = float(largest.perimeter)

    # Compute derived features (avoid division by zero)
    if area > 0 and perimeter > 0:
        compactness = (perimeter**2) / (4 * np.pi * area)
        sphericity = (4 * np.pi * area) / (perimeter**2)
    else:
        compactness = 0.0
        sphericity = 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "eccentricity": float(largest.eccentricity),
        "solidity": float(largest.solidity),
        "extent": float(largest.extent),
        "major_axis_length": float(largest.axis_major_length),
        "minor_axis_length": float(largest.axis_minor_length),
        "compactness": compactness,
        "sphericity": sphericity,
    }
