"""Wavelet-based texture features for CT images.

Scientific Rationale:
- Multi-resolution analysis captures texture at different scales
- Wavelet coefficients encode frequency information lost in spatial analysis
- Effective for detecting subtle tissue patterns in medical imaging
"""

from __future__ import annotations

import numpy as np
import pywt


def extract_wavelet_features(
    image: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
) -> dict[str, float]:
    """Extract wavelet-based texture features.

    Args:
        image: 2D numpy array (H, W) representing the CT image.
        wavelet: Wavelet type (default: 'db4' - Daubechies 4).
        level: Number of decomposition levels (default: 2).

    Returns:
        Dictionary with wavelet features (14 features for level=2):
        - wavelet_L{i}_LH_energy: Energy in LH subband (horizontal edges)
        - wavelet_L{i}_HL_energy: Energy in HL subband (vertical edges)
        - wavelet_L{i}_HH_energy: Energy in HH subband (diagonal features)
        - wavelet_L{i}_LH_mean: Mean absolute value of LH subband
        - wavelet_L{i}_HL_mean: Mean absolute value of HL subband
        - wavelet_L{i}_HH_mean: Mean absolute value of HH subband
        - wavelet_approx_energy: Energy in approximation coefficients
        - wavelet_approx_mean: Mean of approximation coefficients
    """
    features: dict[str, float] = {}

    # Handle edge case of constant image
    if np.std(image) < 1e-10:
        for i in range(1, level + 1):
            features[f"wavelet_L{i}_LH_energy"] = 0.0
            features[f"wavelet_L{i}_HL_energy"] = 0.0
            features[f"wavelet_L{i}_HH_energy"] = 0.0
            features[f"wavelet_L{i}_LH_mean"] = 0.0
            features[f"wavelet_L{i}_HL_mean"] = 0.0
            features[f"wavelet_L{i}_HH_mean"] = 0.0
        features["wavelet_approx_energy"] = float(np.sum(image**2))
        features["wavelet_approx_mean"] = float(np.mean(image))
        return features

    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, level=level)

    # Extract features from detail coefficients at each level
    for i, detail_coeffs in enumerate(coeffs[1:], 1):
        lh, hl, hh = detail_coeffs

        # Energy in each subband
        features[f"wavelet_L{i}_LH_energy"] = float(np.sum(lh**2))
        features[f"wavelet_L{i}_HL_energy"] = float(np.sum(hl**2))
        features[f"wavelet_L{i}_HH_energy"] = float(np.sum(hh**2))

        # Mean absolute values
        features[f"wavelet_L{i}_LH_mean"] = float(np.mean(np.abs(lh)))
        features[f"wavelet_L{i}_HL_mean"] = float(np.mean(np.abs(hl)))
        features[f"wavelet_L{i}_HH_mean"] = float(np.mean(np.abs(hh)))

    # Approximation coefficients
    approx = coeffs[0]
    features["wavelet_approx_energy"] = float(np.sum(approx**2))
    features["wavelet_approx_mean"] = float(np.mean(approx))

    return features
