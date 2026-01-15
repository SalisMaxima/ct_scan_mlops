"""Tests for Hydra configuration validation."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Get absolute path to configs directory
CONFIGS_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.fixture(scope="module")
def default_config() -> DictConfig:
    """Load the default configuration."""
    with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR):
        return compose(config_name="config")


def test_config_loads_successfully(default_config: DictConfig):
    """Test that the default config can be loaded without errors."""
    assert default_config is not None
    assert isinstance(default_config, DictConfig)


def test_config_has_required_sections(default_config: DictConfig):
    """Test that all required config sections exist."""
    required_sections = ["model", "data", "train", "seed", "wandb", "paths"]
    for section in required_sections:
        assert section in default_config, f"Missing required section: {section}"


def test_model_config_structure(default_config: DictConfig):
    """Test model configuration has required fields."""
    model_cfg = default_config.model

    assert "name" in model_cfg
    assert "num_classes" in model_cfg
    assert model_cfg.num_classes == 4  # CT scan has 4 classes


def test_data_config_structure(default_config: DictConfig):
    """Test data configuration has required fields."""
    data_cfg = default_config.data

    assert "batch_size" in data_cfg
    assert isinstance(data_cfg.batch_size, int)
    assert data_cfg.batch_size > 0

    assert "image_size" in data_cfg
    assert data_cfg.image_size > 0

    assert "num_workers" in data_cfg


def test_train_config_structure(default_config: DictConfig):
    """Test training configuration has required fields."""
    train_cfg = default_config.train

    assert "max_epochs" in train_cfg
    assert train_cfg.max_epochs > 0

    assert "optimizer" in train_cfg
    assert "lr" in train_cfg.optimizer
    assert train_cfg.optimizer.lr > 0

    assert "checkpoint" in train_cfg


def test_cnn_model_config():
    """Test CNN model configuration specifically."""
    with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR):
        cfg = compose(config_name="config", overrides=["model=cnn"])

        assert cfg.model.name == "custom_cnn"
        assert cfg.model.num_classes == 4
        assert "hidden_dims" in cfg.model
        assert len(cfg.model.hidden_dims) > 0


def test_resnet18_model_config():
    """Test ResNet18 model configuration specifically."""
    with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR):
        cfg = compose(config_name="config", overrides=["model=resnet18"])

        assert cfg.model.name == "resnet18"
        assert cfg.model.num_classes == 4
        assert "pretrained" in cfg.model


def test_config_override_works():
    """Test that command-line style overrides work correctly."""
    with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR):
        cfg = compose(config_name="config", overrides=["train.max_epochs=10", "data.batch_size=16"])

        assert cfg.train.max_epochs == 10
        assert cfg.data.batch_size == 16


def test_wandb_config_structure(default_config: DictConfig):
    """Test W&B configuration has required fields."""
    wandb_cfg = default_config.wandb

    assert "project" in wandb_cfg
    assert "mode" in wandb_cfg
    assert wandb_cfg.mode in ["online", "offline", "disabled"]


def test_config_can_be_converted_to_dict(default_config: DictConfig):
    """Test that config can be serialized (needed for logging)."""
    config_dict = OmegaConf.to_container(default_config, resolve=True)
    assert isinstance(config_dict, dict)
    assert "model" in config_dict
