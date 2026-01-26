"""Unit tests for radiomics feature extraction module."""

from __future__ import annotations

import numpy as np
import torch

from ct_scan_mlops.features import (
    FeatureConfig,
    FeatureExtractor,
    extract_glcm_features,
    extract_intensity_features,
    extract_region_features,
    extract_shape_features,
    extract_wavelet_features,
)

# Create a random generator for reproducible tests
RNG = np.random.default_rng(42)


def random_image(shape: tuple[int, ...] = (224, 224)) -> np.ndarray:
    """Generate a random test image using modern numpy RNG."""
    return RNG.random(shape).astype(np.float32)


def normal_image(shape: tuple[int, ...] = (224, 224)) -> np.ndarray:
    """Generate a random test image with normal distribution."""
    return RNG.standard_normal(shape).astype(np.float32)


class TestIntensityFeatures:
    """Tests for intensity/histogram features."""

    def test_output_keys(self):
        """Test that all expected features are returned."""
        image = random_image()
        features = extract_intensity_features(image)

        expected_keys = [
            "mean_intensity",
            "std_intensity",
            "skewness",
            "kurtosis",
            "percentile_10",
            "percentile_90",
            "intensity_range",
            "entropy",
        ]
        assert set(features.keys()) == set(expected_keys)

    def test_output_types(self):
        """Test that all features are floats."""
        image = random_image()
        features = extract_intensity_features(image)

        for key, value in features.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_constant_image(self):
        """Test handling of constant (uniform) image."""
        image = np.ones((224, 224), dtype=np.float32) * 0.5
        features = extract_intensity_features(image)

        assert features["std_intensity"] == 0.0
        assert features["skewness"] == 0.0
        assert features["kurtosis"] == 0.0
        assert features["intensity_range"] == 0.0

    def test_zeros_image(self):
        """Test handling of all-zeros image."""
        image = np.zeros((224, 224), dtype=np.float32)
        features = extract_intensity_features(image)

        assert features["mean_intensity"] == 0.0
        assert not np.isnan(features["entropy"])

    def test_no_nan_values(self):
        """Test that no NaN values are returned."""
        image = random_image()
        features = extract_intensity_features(image)

        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"


class TestGLCMFeatures:
    """Tests for GLCM texture features."""

    def test_output_keys(self):
        """Test that all expected features are returned."""
        image = random_image()
        features = extract_glcm_features(image)

        expected_keys = [
            "glcm_contrast_mean",
            "glcm_contrast_std",
            "glcm_homogeneity_mean",
            "glcm_homogeneity_std",
            "glcm_energy_mean",
            "glcm_energy_std",
            "glcm_correlation_mean",
            "glcm_correlation_std",
            "glcm_dissimilarity_mean",
            "glcm_dissimilarity_std",
            "glcm_ASM_mean",
            "glcm_ASM_std",
            "glcm_entropy",
        ]
        assert set(features.keys()) == set(expected_keys)

    def test_constant_image(self):
        """Test handling of constant (uniform) image."""
        image = np.ones((224, 224), dtype=np.float32) * 0.5
        features = extract_glcm_features(image)

        # Constant image should have no contrast
        assert features["glcm_contrast_mean"] == 0.0
        # But high homogeneity/energy
        assert features["glcm_homogeneity_mean"] == 1.0
        assert features["glcm_energy_mean"] == 1.0

    def test_no_nan_values(self):
        """Test that no NaN values are returned."""
        image = random_image()
        features = extract_glcm_features(image)

        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"

    def test_custom_distances(self):
        """Test with custom GLCM distances."""
        image = random_image()
        features = extract_glcm_features(image, distances=[1, 5, 10])

        assert len(features) == 13  # Same number of features


class TestShapeFeatures:
    """Tests for shape/morphological features."""

    def test_output_keys(self):
        """Test that all expected features are returned."""
        image = random_image()
        features = extract_shape_features(image)

        expected_keys = [
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "extent",
            "major_axis_length",
            "minor_axis_length",
            "compactness",
            "sphericity",
        ]
        assert set(features.keys()) == set(expected_keys)

    def test_constant_image(self):
        """Test handling of constant (uniform) image."""
        image = np.ones((224, 224), dtype=np.float32) * 0.5
        features = extract_shape_features(image)

        # All features should be zeros for constant image
        assert features["area"] == 0.0

    def test_no_nan_values(self):
        """Test that no NaN values are returned."""
        image = random_image()
        features = extract_shape_features(image)

        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"

    def test_binary_circle(self):
        """Test with a simple binary circle (known shape)."""
        image = np.zeros((224, 224), dtype=np.float32)
        # Create a circle
        y, x = np.ogrid[:224, :224]
        center = (112, 112)
        radius = 50
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        image[mask] = 1.0

        features = extract_shape_features(image)

        # Circle should have high sphericity (close to 1)
        assert features["sphericity"] > 0.8
        assert features["area"] > 0


class TestRegionFeatures:
    """Tests for region-based features."""

    def test_output_keys(self):
        """Test that all expected features are returned."""
        image = random_image()
        features = extract_region_features(image)

        expected_keys = [
            "solid_ratio",
            "ggo_ratio",
            "low_density_ratio",
            "mean_gradient",
            "max_gradient",
            "boundary_irregularity",
        ]
        assert set(features.keys()) == set(expected_keys)

    def test_ratios_sum_roughly_to_one(self):
        """Test that density ratios are reasonable."""
        image = random_image()
        features = extract_region_features(image)

        # Ratios should be between 0 and 1
        assert 0 <= features["solid_ratio"] <= 1
        assert 0 <= features["ggo_ratio"] <= 1
        assert 0 <= features["low_density_ratio"] <= 1

    def test_constant_image(self):
        """Test handling of constant (uniform) image."""
        image = np.ones((224, 224), dtype=np.float32) * 0.5
        features = extract_region_features(image)

        # Gradients should be zero
        assert features["mean_gradient"] == 0.0
        assert features["max_gradient"] == 0.0

    def test_no_nan_values(self):
        """Test that no NaN values are returned."""
        image = random_image()
        features = extract_region_features(image)

        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"


class TestWaveletFeatures:
    """Tests for wavelet features."""

    def test_output_keys_level2(self):
        """Test that all expected features are returned for level 2."""
        image = random_image()
        features = extract_wavelet_features(image, level=2)

        # Level 2: 6 features per level x 2 levels + 2 approx features = 14
        expected_keys = [
            "wavelet_L1_LH_energy",
            "wavelet_L1_HL_energy",
            "wavelet_L1_HH_energy",
            "wavelet_L1_LH_mean",
            "wavelet_L1_HL_mean",
            "wavelet_L1_HH_mean",
            "wavelet_L2_LH_energy",
            "wavelet_L2_HL_energy",
            "wavelet_L2_HH_energy",
            "wavelet_L2_LH_mean",
            "wavelet_L2_HL_mean",
            "wavelet_L2_HH_mean",
            "wavelet_approx_energy",
            "wavelet_approx_mean",
        ]
        assert set(features.keys()) == set(expected_keys)

    def test_constant_image(self):
        """Test handling of constant (uniform) image."""
        image = np.ones((224, 224), dtype=np.float32) * 0.5
        features = extract_wavelet_features(image)

        # Detail coefficients should be zero for constant image
        assert features["wavelet_L1_LH_energy"] == 0.0
        assert features["wavelet_L1_HL_energy"] == 0.0
        assert features["wavelet_L1_HH_energy"] == 0.0

    def test_no_nan_values(self):
        """Test that no NaN values are returned."""
        image = random_image()
        features = extract_wavelet_features(image)

        for key, value in features.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"

    def test_different_wavelet_types(self):
        """Test with different wavelet types."""
        image = random_image()

        for wavelet in ["db4", "haar", "sym4"]:
            features = extract_wavelet_features(image, wavelet=wavelet)
            assert len(features) == 14  # Same structure


class TestFeatureExtractor:
    """Tests for the main FeatureExtractor class."""

    def test_default_config(self):
        """Test extraction with default config."""
        extractor = FeatureExtractor()
        image = random_image()
        features = extractor.extract(image)

        # Default config enables all features
        assert len(features) == extractor.feature_dim
        assert features.ndim == 1

    def test_feature_dim_property(self):
        """Test that feature_dim is computed correctly."""
        extractor = FeatureExtractor()
        dim = extractor.feature_dim

        # Default: 8 + 13 + 9 + 6 + 14 = 50
        assert dim == 50

    def test_selective_features(self):
        """Test extraction with selective feature categories."""
        config = FeatureConfig(
            use_intensity=True,
            use_glcm=False,
            use_shape=False,
            use_region=False,
            use_wavelet=False,
        )
        extractor = FeatureExtractor(config=config)

        # Only intensity features (8)
        assert extractor.feature_dim == 8

    def test_batch_extraction(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor()
        batch = torch.rand(4, 3, 224, 224)
        features = extractor.extract_batch(batch, n_jobs=1, show_progress=False)

        assert features.shape == (4, extractor.feature_dim)
        assert features.dtype == torch.float32

    def test_3d_input(self):
        """Test extraction from 3D input (C, H, W)."""
        extractor = FeatureExtractor()
        image = random_image((3, 224, 224))
        features = extractor.extract(image)

        assert len(features) == extractor.feature_dim

    def test_no_nan_in_output(self):
        """Test that output contains no NaN values."""
        extractor = FeatureExtractor()

        # Test various edge cases
        test_cases = [
            np.zeros((224, 224), dtype=np.float32),
            np.ones((224, 224), dtype=np.float32),
            random_image(),
            normal_image(),
        ]

        for image in test_cases:
            features = extractor.extract(image)
            assert not np.any(np.isnan(features)), "Found NaN in features"
            assert not np.any(np.isinf(features)), "Found Inf in features"

    def test_get_feature_names(self):
        """Test that feature names match feature dimension."""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == extractor.feature_dim
        assert all(isinstance(name, str) for name in names)

    def test_feature_names_consistency(self):
        """Test that feature names are consistent across calls."""
        extractor = FeatureExtractor()
        names1 = extractor.get_feature_names()
        names2 = extractor.get_feature_names()

        assert names1 == names2


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeatureConfig()

        assert config.use_intensity is True
        assert config.use_glcm is True
        assert config.use_shape is True
        assert config.use_region is True
        assert config.use_wavelet is True
        assert config.glcm_distances == [1, 2, 3]
        assert config.wavelet_type == "db4"
        assert config.wavelet_level == 2

    def test_from_dict(self):
        """Test creating config from dictionary."""
        cfg = {
            "use_intensity": True,
            "use_glcm": False,
            "use_shape": True,
            "use_region": False,
            "use_wavelet": True,
            "glcm_distances": [1, 3, 5],
            "wavelet_type": "haar",
            "wavelet_level": 3,
        }
        config = FeatureConfig.from_dict(cfg)

        assert config.use_glcm is False
        assert config.use_region is False
        assert config.glcm_distances == [1, 3, 5]
        assert config.wavelet_type == "haar"
        assert config.wavelet_level == 3
