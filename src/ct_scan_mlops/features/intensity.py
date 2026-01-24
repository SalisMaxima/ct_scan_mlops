"""First-order intensity/histogram features for CT images.

Scientific Rationale:
- Hounsfield Unit (HU) statistics directly reflect tissue density
- Ground-glass opacity (adenocarcinoma) has lower HU than solid tumors (SCC)
- Normal lung parenchyma has characteristic low HU values (-700 to -900)
- Entropy measures tissue heterogeneity, elevated in malignant lesions
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def extract_intensity_features(image: np.ndarray) -> dict[str, float]:
    """Extract intensity-based features from CT image.

    Args:
        image: 2D numpy array (H, W) representing the CT image.

    Returns:
        Dictionary with 8 intensity features:
        - mean_intensity: Mean pixel value (overall tissue density)
        - std_intensity: Standard deviation (density variation)
        - skewness: Third standardized moment (asymmetry of distribution)
        - kurtosis: Fourth standardized moment (outlier intensities)
        - percentile_10: 10th percentile (low-density component: air/GGO)
        - percentile_90: 90th percentile (high-density component: solid tissue)
        - intensity_range: P90 - P10 (dynamic range of densities)
        - entropy: -sum(p * log(p)) (heterogeneity/randomness)
    """
    flat = image.flatten().astype(np.float64)

    # Handle edge case of constant image
    if np.std(flat) < 1e-10:
        return {
            "mean_intensity": float(np.mean(flat)),
            "std_intensity": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "percentile_10": float(np.percentile(flat, 10)),
            "percentile_90": float(np.percentile(flat, 90)),
            "intensity_range": 0.0,
            "entropy": 0.0,
        }

    p10 = np.percentile(flat, 10)
    p90 = np.percentile(flat, 90)

    # Compute histogram for entropy
    hist, _ = np.histogram(flat, bins=256, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))

    return {
        "mean_intensity": float(np.mean(flat)),
        "std_intensity": float(np.std(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "percentile_10": float(p10),
        "percentile_90": float(p90),
        "intensity_range": float(p90 - p10),
        "entropy": float(entropy),
    }
