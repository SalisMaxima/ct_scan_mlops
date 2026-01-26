"""Tests for analysis utilities."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from ct_scan_mlops.analysis.utils import (
    DUAL_PATHWAY_MODEL_NAMES,
    LoadedModel,
    ModelLoader,
    model_forward,
    unpack_batch,
)


class TestDualPathwayModelNames:
    """Tests for DUAL_PATHWAY_MODEL_NAMES constant."""

    def test_contains_dual_pathway(self):
        """Should contain dual_pathway."""
        assert "dual_pathway" in DUAL_PATHWAY_MODEL_NAMES

    def test_contains_dualpathway(self):
        """Should contain dualpathway (no underscore)."""
        assert "dualpathway" in DUAL_PATHWAY_MODEL_NAMES

    def test_contains_hybrid(self):
        """Should contain hybrid."""
        assert "hybrid" in DUAL_PATHWAY_MODEL_NAMES

    def test_does_not_contain_custom_cnn(self):
        """Should not contain custom_cnn."""
        assert "custom_cnn" not in DUAL_PATHWAY_MODEL_NAMES


class TestModelLoaderDetectUsesFeatures:
    """Tests for ModelLoader.detect_uses_features method."""

    def test_detect_dual_pathway(self):
        """Should detect dual_pathway as using features."""
        cfg = OmegaConf.create({"model": {"name": "dual_pathway"}})
        assert ModelLoader.detect_uses_features(cfg) is True

    def test_detect_dualpathway(self):
        """Should detect dualpathway as using features."""
        cfg = OmegaConf.create({"model": {"name": "dualpathway"}})
        assert ModelLoader.detect_uses_features(cfg) is True

    def test_detect_hybrid(self):
        """Should detect hybrid as using features."""
        cfg = OmegaConf.create({"model": {"name": "hybrid"}})
        assert ModelLoader.detect_uses_features(cfg) is True

    def test_detect_custom_cnn(self):
        """Should detect custom_cnn as NOT using features."""
        cfg = OmegaConf.create({"model": {"name": "custom_cnn"}})
        assert ModelLoader.detect_uses_features(cfg) is False

    def test_detect_resnet18(self):
        """Should detect resnet18 as NOT using features."""
        cfg = OmegaConf.create({"model": {"name": "resnet18"}})
        assert ModelLoader.detect_uses_features(cfg) is False

    def test_case_insensitive(self):
        """Should be case insensitive."""
        cfg = OmegaConf.create({"model": {"name": "DUAL_PATHWAY"}})
        assert ModelLoader.detect_uses_features(cfg) is True


class TestModelLoaderFindConfig:
    """Tests for ModelLoader.find_config method."""

    def test_find_config_with_override(self, tmp_path):
        """Should use config override when provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  name: test_model")

        cfg = ModelLoader.find_config(
            checkpoint_path=tmp_path / "model.ckpt",
            config_override=config_file,
        )
        assert cfg.model.name == "test_model"

    def test_find_config_from_hydra_dir(self, tmp_path):
        """Should find config from .hydra directory."""
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir()
        config_file = hydra_dir / "config.yaml"
        config_file.write_text("model:\n  name: from_hydra")

        cfg = ModelLoader.find_config(
            checkpoint_path=tmp_path / "model.ckpt",
            config_override=None,
        )
        assert cfg.model.name == "from_hydra"

    def test_find_config_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError when no config found."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ModelLoader.find_config(
                checkpoint_path=tmp_path / "model.ckpt",
                config_override=None,
            )
        assert "Config not found" in str(exc_info.value)

    def test_find_config_override_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError when config override doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ModelLoader.find_config(
                checkpoint_path=tmp_path / "model.ckpt",
                config_override=tmp_path / "nonexistent.yaml",
            )
        assert "Config override not found" in str(exc_info.value)


class TestUnpackBatch:
    """Tests for unpack_batch utility."""

    def test_unpack_3tuple_with_features(self):
        """Should unpack 3-tuple batch correctly with features."""
        device = torch.device("cpu")
        images = torch.randn(2, 3, 224, 224)
        features = torch.randn(2, 50)
        targets = torch.tensor([0, 1])

        batch = (images, features, targets)
        img, feat, tgt = unpack_batch(batch, device, use_features=True)

        assert img.shape == (2, 3, 224, 224)
        assert feat.shape == (2, 50)
        assert tgt.shape == (2,)

    def test_unpack_3tuple_without_features(self):
        """Should return None for features when use_features=False."""
        device = torch.device("cpu")
        batch = (torch.randn(2, 3, 224, 224), torch.randn(2, 50), torch.tensor([0, 1]))

        img, feat, tgt = unpack_batch(batch, device, use_features=False)

        assert img.shape == (2, 3, 224, 224)
        assert feat is None
        assert tgt.shape == (2,)

    def test_unpack_2tuple(self):
        """Should unpack 2-tuple batch correctly."""
        device = torch.device("cpu")
        images = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 1])

        batch = (images, targets)
        img, feat, tgt = unpack_batch(batch, device, use_features=False)

        assert img.shape == (2, 3, 224, 224)
        assert feat is None
        assert tgt.shape == (2,)

    def test_unpack_2tuple_with_use_features_true(self):
        """Should handle 2-tuple batch even when use_features=True."""
        device = torch.device("cpu")
        batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))

        img, feat, tgt = unpack_batch(batch, device, use_features=True)

        assert img.shape == (2, 3, 224, 224)
        assert feat is None  # No features in batch
        assert tgt.shape == (2,)


class TestModelForward:
    """Tests for model_forward utility."""

    def test_forward_with_features(self):
        """Should call model with features when use_features=True."""
        model = Mock()
        model.return_value = torch.randn(2, 4)

        images = torch.randn(2, 3, 224, 224)
        features = torch.randn(2, 50)

        _output = model_forward(model, images, features, use_features=True)

        model.assert_called_once()
        args = model.call_args[0]
        assert len(args) == 2  # images and features

    def test_forward_without_features(self):
        """Should call model without features when use_features=False."""
        model = Mock()
        model.return_value = torch.randn(2, 4)

        images = torch.randn(2, 3, 224, 224)

        _output = model_forward(model, images, None, use_features=False)

        model.assert_called_once()
        args = model.call_args[0]
        assert len(args) == 1  # only images

    def test_forward_with_use_features_but_none_features(self):
        """Should handle use_features=True but features=None gracefully."""
        model = Mock()
        model.return_value = torch.randn(2, 4)

        images = torch.randn(2, 3, 224, 224)

        _output = model_forward(model, images, None, use_features=True)

        # Should fall back to calling without features
        model.assert_called_once()
        args = model.call_args[0]
        assert len(args) == 1  # only images


class TestLoadedModel:
    """Tests for LoadedModel dataclass."""

    def test_create_loaded_model(self):
        """Should create LoadedModel with all fields."""
        model = Mock()
        config = OmegaConf.create({"model": {"name": "test"}})
        checkpoint_path = Path("/path/to/model.ckpt")

        loaded = LoadedModel(
            model=model,
            config=config,
            uses_features=True,
            model_name="test",
            checkpoint_path=checkpoint_path,
        )

        assert loaded.model is model
        assert loaded.config == config
        assert loaded.uses_features is True
        assert loaded.model_name == "test"
        assert loaded.checkpoint_path == checkpoint_path
