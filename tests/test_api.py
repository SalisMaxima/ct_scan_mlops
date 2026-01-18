"""Tests for CT Scan API endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

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
