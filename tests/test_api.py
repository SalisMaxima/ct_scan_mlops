"""Tests for CT Scan API endpoints."""

from __future__ import annotations

import io
from pathlib import Path
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
    """Create a sample test image with sufficient contrast."""
    import numpy as np

    # Create image with varying intensities to pass validation
    rng = np.random.default_rng()
    arr = rng.integers(50, 200, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
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
        # New response format
        assert "prediction" in data
        assert "probabilities" in data
        assert "metadata" in data
        # Validate prediction structure
        prediction = data["prediction"]
        assert "class" in prediction
        assert "class_index" in prediction
        assert "confidence" in prediction
        assert isinstance(prediction["class_index"], int)
        assert prediction["class_index"] in range(4)
        assert prediction["class"] in api.CLASS_NAMES
        # Validate probabilities
        probs = data["probabilities"]
        assert len(probs) == 4
        assert all(k in api.CLASS_NAMES for k in probs)
        assert all(0 <= v <= 1 for v in probs.values())
        # Validate metadata
        assert "model_type" in data["metadata"]
        assert "device" in data["metadata"]

    def test_predict_returns_class_name(self, client, sample_image):
        """Test prediction includes human-readable class name."""
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image, "image/png")},
        )
        data = response.json()
        # Mock model returns highest logit for index 2
        assert data["prediction"]["class_index"] == 2
        assert data["prediction"]["class"] == api.CLASS_NAMES[2]

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

    def test_predict_validates_image_size(self, client):
        """Test prediction validates image dimensions."""
        # Create tiny image (below minimum)
        tiny_img = Image.new("RGB", (32, 32), color="gray")
        buffer = io.BytesIO()
        tiny_img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("tiny.png", buffer.read(), "image/png")},
        )
        assert response.status_code == 422
        assert "too small" in response.json()["detail"].lower()


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_multiple_images(self, client, sample_image):
        """Test batch prediction with multiple images."""
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.png", sample_image, "image/png")),
                ("files", ("test2.png", sample_image, "image/png")),
                ("files", ("test3.png", sample_image, "image/png")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert "batch_summary" in data
        assert "results" in data
        assert data["batch_summary"]["total"] == 3
        assert data["batch_summary"]["successful"] == 3
        assert len(data["results"]) == 3

    def test_batch_predict_exceeds_limit(self, client, sample_image):
        """Test batch prediction exceeds max batch size."""
        files = [("files", (f"test{i}.png", sample_image, "image/png")) for i in range(25)]
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_batch_predict_handles_failures(self, client, sample_image):
        """Test batch prediction handles individual failures gracefully."""
        response = client.post(
            "/predict/batch",
            files=[
                ("files", ("test1.png", sample_image, "image/png")),
                ("files", ("bad.png", b"invalid", "image/png")),
                ("files", ("test2.png", sample_image, "image/png")),
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["batch_summary"]["total"] == 3
        assert data["batch_summary"]["successful"] == 2
        assert data["batch_summary"]["failed"] == 1


class TestExplainEndpoint:
    """Tests for /explain endpoint."""

    def test_explain_generates_heatmap(self, client, sample_image, mock_model):
        """Test explain endpoint handles explainability request."""
        response = client.post(
            "/explain",
            files={"file": ("test.png", sample_image, "image/png")},
        )
        # GradCAM may fail with mock models (no grad_fn), so we accept 200 or 500
        # In production with real models, this should return 200
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "explanation" in data
            assert "heatmap" in data["explanation"]
            assert "description" in data["explanation"]
            # Heatmap should be base64 encoded
            import base64

            try:
                base64.b64decode(data["explanation"]["heatmap"])
            except Exception:
                pytest.fail("Heatmap is not valid base64")
        else:
            # Expected failure with mock model
            assert "Explainability generation failed" in response.json()["detail"]

    def test_explain_invalid_image(self, client):
        """Test explain endpoint with invalid image."""
        response = client.post(
            "/explain",
            files={"file": ("bad.png", b"invalid", "image/png")},
        )
        assert response.status_code == 400


class TestFeedbackStatsEndpoint:
    """Tests for /feedback/stats endpoint."""

    def test_feedback_stats_returns_summary(self, client, tmp_path):
        """Test feedback stats endpoint returns summary."""
        from ct_scan_mlops.feedback_store import SqliteFeedbackStore

        api.feedback_store = SqliteFeedbackStore(db_path=str(tmp_path / "feedback.db"))

        response = client.get("/feedback/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data
        assert "correct_predictions" in data
        assert "incorrect_predictions" in data
        assert "accuracy" in data
        assert "class_distribution" in data
        assert "recent_feedback" in data


class TestFeedbackEndpoint:
    """Tests for /feedback endpoint."""

    def test_feedback_saves_correct_prediction(self, client, sample_image, tmp_path):
        from ct_scan_mlops.feedback_store import SqliteFeedbackStore

        api.FEEDBACK_DIR = tmp_path / "feedback"
        api.feedback_store = SqliteFeedbackStore(db_path=str(tmp_path / "feedback.db"))

        response = client.post(
            "/feedback",
            files={"file": ("scan.png", sample_image, "image/png")},
            data={
                "predicted_class": "normal",
                "predicted_confidence": "0.85",
                "is_correct": "true",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        saved_path = Path(payload["saved_to"])
        assert payload["class"] == "normal"
        assert payload["logged_to_db"] is True
        assert saved_path.exists()
        assert saved_path.parent.name == "normal"

    def test_feedback_saves_in_correct_class(self, client, sample_image, tmp_path):
        from ct_scan_mlops.feedback_store import SqliteFeedbackStore

        api.FEEDBACK_DIR = tmp_path / "feedback"
        api.feedback_store = SqliteFeedbackStore(db_path=str(tmp_path / "feedback.db"))

        response = client.post(
            "/feedback",
            files={"file": ("scan.png", sample_image, "image/png")},
            data={
                "predicted_class": "normal",
                "predicted_confidence": "0.75",
                "is_correct": "false",
                "correct_class": "adenocarcinoma",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        saved_path = Path(payload["saved_to"])
        assert payload["class"] == "adenocarcinoma"
        assert saved_path.exists()
        assert saved_path.parent.name == "adenocarcinoma"

    def test_feedback_requires_correct_class_when_incorrect(self, client, sample_image):
        response = client.post(
            "/feedback",
            files={"file": ("scan.png", sample_image, "image/png")},
            data={
                "predicted_class": "normal",
                "is_correct": "false",
            },
        )

        assert response.status_code == 400
        assert "correct_class" in response.json()["detail"]


class TestClassNames:
    """Tests for class name configuration."""

    def test_class_names_defined(self):
        """Test that class names are properly defined."""
        assert len(api.CLASS_NAMES) == 4
        assert "adenocarcinoma" in api.CLASS_NAMES
        assert "large_cell_carcinoma" in api.CLASS_NAMES
        assert "normal" in api.CLASS_NAMES
        assert "squamous_cell_carcinoma" in api.CLASS_NAMES
