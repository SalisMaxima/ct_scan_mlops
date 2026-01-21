"""Tests for CT Scan API endpoints."""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ct_scan_mlops import api


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable outputs."""
    model = MagicMock()
    model.eval = MagicMock(return_value=None)
    model.to = MagicMock(return_value=model)
    # Return logits where class 2 (normal) has highest score
    model.return_value = torch.tensor([[0.1, 0.2, 0.9, 0.1]])
    return model


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    # Patch the model loading
    api.model = mock_model
    return TestClient(api.app, raise_server_exceptions=False)


@pytest.fixture
def sample_image() -> bytes:
    """Create a sample test image."""
    img = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "device" in data
        assert "model_loaded" in data

    def test_health_shows_model_loaded(self, client):
        """Test health endpoint shows model is loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_image(self, client, sample_image):
        """Test prediction with valid image."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "pred_index" in data
        assert "pred_class" in data
        assert isinstance(data["pred_index"], int)
        assert data["pred_index"] in range(4)
        assert data["pred_class"] in api.CLASS_NAMES

    def test_predict_returns_class_name(self, client, sample_image):
        """Test prediction includes human-readable class name."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")},
        )
        data = response.json()
        # Mock model returns highest logit for index 2
        assert data["pred_index"] == 2
        assert data["pred_class"] == "normal"

    def test_predict_invalid_image(self, client):
        """Test prediction with invalid image data."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_predict_model_not_loaded(self, client):
        """Test prediction when model is not loaded."""
        api.model = None
        response = client.post(
            "/predict",
            files={"file": ("test.png", b"fake", "image/png")},
        )
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


class TestClassNames:
    """Tests for class name configuration."""

    def test_class_names_defined(self):
        """Test that class names are properly defined."""
        assert len(api.CLASS_NAMES) == 4
        assert "adenocarcinoma" in api.CLASS_NAMES
        assert "large_cell_carcinoma" in api.CLASS_NAMES
        assert "normal" in api.CLASS_NAMES
        assert "squamous_cell_carcinoma" in api.CLASS_NAMES


class TestStartupLogging:
    """Tests for startup logging behavior."""

    def test_startup_logs_config_check(self, caplog, tmp_path):
        """Test that startup logs config file check."""
        import logging
        from contextlib import asynccontextmanager

        from ct_scan_mlops import api

        # Create a temporary config and model
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: resnet18\n  num_classes: 4\n")
        model_path = tmp_path / "model.pt"

        # Create a dummy model state
        dummy_state = {"weight": torch.tensor([1.0])}
        torch.save(dummy_state, model_path)

        with (
            patch.object(api, "CONFIG_PATH", config_path),
            patch.object(api, "MODEL_PATH", model_path),
            patch("ct_scan_mlops.api.build_model") as mock_build,
        ):
            mock_model = MagicMock()
            mock_model.eval = MagicMock(return_value=None)
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.load_state_dict = MagicMock()
            mock_build.return_value = mock_model

            with caplog.at_level(logging.INFO):
                # Create a new app instance with patched paths
                @asynccontextmanager
                async def test_lifespan(_app):
                    async with api.lifespan(_app):
                        yield

                from fastapi import FastAPI

                test_app = FastAPI(lifespan=test_lifespan)

                with TestClient(test_app):
                    # Check that startup logs were generated
                    assert any("Startup: Checking config" in record.message for record in caplog.records)
                    assert any("Startup: Checking model" in record.message for record in caplog.records)
                    assert any("Model loaded successfully" in record.message for record in caplog.records)

    def test_startup_logs_missing_config_debug_info(self, caplog, tmp_path):
        """Test that startup logs debug info when config is missing."""
        import os

        from ct_scan_mlops import api

        # Set DEBUG environment variable
        os.environ["DEBUG"] = "1"

        # Use a non-existent config path
        config_path = tmp_path / "nonexistent" / "config.yaml"

        with patch.object(api, "CONFIG_PATH", config_path), caplog.at_level(logging.DEBUG):
            try:
                # Create a new app instance with patched paths
                from fastapi import FastAPI

                test_app = FastAPI(lifespan=api.lifespan)
                with TestClient(test_app, raise_server_exceptions=False):
                    pass
            except RuntimeError:
                pass  # Expected to fail

            # Check that error was logged
            assert any("Config not found" in record.message for record in caplog.records)
            # Debug logs should be available
            assert any(record.levelname == "DEBUG" for record in caplog.records)

        # Clean up
        os.environ.pop("DEBUG", None)

    def test_startup_logs_missing_model_debug_info(self, caplog, tmp_path):
        """Test that startup logs debug info when model is missing."""
        import os

        from ct_scan_mlops import api

        # Set DEBUG environment variable
        os.environ["DEBUG"] = "1"

        # Create a temporary config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  name: resnet18\n  num_classes: 4\n")

        # Use a non-existent model path
        model_path = tmp_path / "nonexistent" / "model.pt"

        with (
            patch.object(api, "CONFIG_PATH", config_path),
            patch.object(api, "MODEL_PATH", model_path),
            caplog.at_level(logging.DEBUG),
        ):
            try:
                # Create a new app instance with patched paths
                from fastapi import FastAPI

                test_app = FastAPI(lifespan=api.lifespan)
                with TestClient(test_app, raise_server_exceptions=False):
                    pass
            except RuntimeError:
                pass  # Expected to fail

            # Check that error was logged
            assert any("Model not found" in record.message for record in caplog.records)
            # Debug logs should be available
            assert any(record.levelname == "DEBUG" for record in caplog.records)

        # Clean up
        os.environ.pop("DEBUG", None)
