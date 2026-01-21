"""Tests for promote_model module.

Tests the convert_ckpt_to_pt function that converts PyTorch Lightning
checkpoints to production-ready state dict files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ct_scan_mlops.model import CustomCNN
from ct_scan_mlops.promote_model import convert_ckpt_to_pt


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model files."""
    return tmp_path / "models"


@pytest.fixture
def sample_model() -> CustomCNN:
    """Create a sample model for testing."""
    return CustomCNN(
        num_classes=4,
        in_channels=3,
        hidden_dims=[8, 16],
        fc_hidden=32,
        dropout=0.1,
        batch_norm=True,
        image_size=64,
    )


@pytest.fixture
def ckpt_file(temp_model_dir: Path, sample_model: CustomCNN) -> Path:
    """Create a sample .ckpt file mimicking PyTorch Lightning format."""
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = temp_model_dir / "model.ckpt"

    # Mimic Lightning checkpoint structure with "model." prefix on keys
    state_dict_with_prefix = {f"model.{k}": v for k, v in sample_model.state_dict().items()}

    checkpoint = {
        "epoch": 10,
        "global_step": 1000,
        "state_dict": state_dict_with_prefix,
        "optimizer_states": [{"state": {}, "param_groups": []}],
        "lr_schedulers": [],
        "callbacks": {},
        "hyper_parameters": {"num_classes": 4},
    }

    torch.save(checkpoint, ckpt_path)
    return ckpt_path


class TestConvertCkptToPt:
    """Tests for convert_ckpt_to_pt function."""

    def test_converts_ckpt_to_pt_successfully(self, ckpt_file: Path, temp_model_dir: Path, sample_model: CustomCNN):
        """Test basic conversion from .ckpt to .pt."""
        pt_path = temp_model_dir / "model.pt"
        convert_ckpt_to_pt(ckpt_file, pt_path)

        assert pt_path.exists()

        # Load the converted weights securely
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)

        # Verify the model can load the converted weights
        sample_model.load_state_dict(state_dict)

    def test_removes_model_prefix(self, ckpt_file: Path, temp_model_dir: Path):
        """Test that 'model.' prefix is removed from state_dict keys."""
        pt_path = temp_model_dir / "model.pt"
        convert_ckpt_to_pt(ckpt_file, pt_path)

        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

        # None of the keys should start with "model."
        for key in state_dict:
            assert not key.startswith("model."), f"Key '{key}' still has 'model.' prefix"

    def test_output_file_is_smaller(self, ckpt_file: Path, temp_model_dir: Path):
        """Test that .pt file is smaller than .ckpt (no optimizer states)."""
        pt_path = temp_model_dir / "model.pt"
        convert_ckpt_to_pt(ckpt_file, pt_path)

        ckpt_size = ckpt_file.stat().st_size
        pt_size = pt_path.stat().st_size

        # .pt should be smaller (no optimizer states, callbacks, etc.)
        assert pt_size <= ckpt_size

    def test_creates_parent_directories(self, ckpt_file: Path, temp_model_dir: Path):
        """Test that parent directories are created if they don't exist."""
        pt_path = temp_model_dir / "nested" / "dir" / "model.pt"
        convert_ckpt_to_pt(ckpt_file, pt_path)

        assert pt_path.exists()
        assert pt_path.parent.exists()

    def test_raises_on_missing_ckpt(self, temp_model_dir: Path):
        """Test that FileNotFoundError is raised if .ckpt doesn't exist."""
        ckpt_path = temp_model_dir / "nonexistent.ckpt"
        pt_path = temp_model_dir / "model.pt"

        with pytest.raises(FileNotFoundError):
            convert_ckpt_to_pt(ckpt_path, pt_path)

    def test_handles_checkpoint_without_state_dict_key(self, temp_model_dir: Path, sample_model: CustomCNN):
        """Test handling of checkpoint that is directly a state_dict."""
        temp_model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = temp_model_dir / "simple.ckpt"
        pt_path = temp_model_dir / "model.pt"

        # Save state_dict directly (not as Lightning checkpoint)
        torch.save(sample_model.state_dict(), ckpt_path)

        convert_ckpt_to_pt(ckpt_path, pt_path)
        assert pt_path.exists()

        # Should still work
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        sample_model.load_state_dict(state_dict)

    def test_converted_model_produces_same_output(self, ckpt_file: Path, temp_model_dir: Path, sample_model: CustomCNN):
        """Test that model with converted weights produces same output as original."""
        pt_path = temp_model_dir / "model.pt"
        convert_ckpt_to_pt(ckpt_file, pt_path)

        # Load original weights (with prefix handling)
        original_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        original_state = original_ckpt["state_dict"]
        original_state_clean = {k.replace("model.", "", 1): v for k, v in original_state.items()}

        model_original = CustomCNN(num_classes=4, in_channels=3, hidden_dims=[8, 16], fc_hidden=32, image_size=64)
        model_original.load_state_dict(original_state_clean)

        # Load converted weights
        converted_state = torch.load(pt_path, map_location="cpu", weights_only=True)
        model_converted = CustomCNN(num_classes=4, in_channels=3, hidden_dims=[8, 16], fc_hidden=32, image_size=64)
        model_converted.load_state_dict(converted_state)

        # Both should produce identical outputs
        model_original.eval()
        model_converted.eval()
        test_input = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            out_original = model_original(test_input)
            out_converted = model_converted(test_input)

        assert torch.allclose(out_original, out_converted)
