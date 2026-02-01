from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from prometheus_client import Counter, Gauge, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from torchvision import transforms

from ct_scan_mlops.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from ct_scan_mlops.features.extractor import FeatureConfig, FeatureExtractor
from ct_scan_mlops.model import DualPathwayModel, build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"
DEFAULT_CKPT_PATH = Path("outputs/checkpoints") / "best_model.ckpt"
DEFAULT_PT_PATH = Path("outputs/checkpoints") / "model.pt"
DEFAULT_FEATURE_METADATA_PATH = Path("outputs/checkpoints") / "feature_metadata.json"

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_PT_PATH)))
FEATURE_METADATA_PATH = Path(os.environ.get("FEATURE_METADATA_PATH", str(DEFAULT_FEATURE_METADATA_PATH)))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "feedback"))
FEEDBACK_DB = Path(os.environ.get("FEEDBACK_DB", "feedback/feedback.db"))
MODEL_PATH_ENV = os.environ.get("MODEL_PATH")


def resolve_model_path() -> Path:
    """Resolve model path with .pt preferred for security, .ckpt as fallback.

    Production deployments should use .pt files which can be loaded securely
    with weights_only=True. The .ckpt format is kept as fallback for development.

    Env var MODEL_PATH takes precedence when set.
    """
    if MODEL_PATH_ENV:
        return Path(MODEL_PATH_ENV)
    # Prefer .pt for secure loading (production)
    if DEFAULT_PT_PATH.exists():
        return DEFAULT_PT_PATH
    # Fallback to .ckpt (development/legacy)
    return DEFAULT_CKPT_PATH


def load_config(cfg_path: Path) -> DictConfig:
    """Load a Hydra config, composing if needed.

    Raises:
        RuntimeError: If the config cannot be loaded or composed.
    """
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception as exc:  # pragma: no cover - defensive, depends on filesystem / Hydra
        msg = f"Failed to load config from '{cfg_path}': {exc}"
        raise RuntimeError(msg) from exc

    if "model" in cfg:
        return cfg

    if "defaults" in cfg:
        try:
            with initialize_config_dir(
                version_base=None,
                config_dir=str(cfg_path.parent.resolve()),
            ):
                return compose(config_name=cfg_path.stem)
        except Exception as exc:  # pragma: no cover - defensive, depends on Hydra internals
            msg = f"Failed to compose Hydra config from '{cfg_path}': {exc}"
            raise RuntimeError(msg) from exc

    return cfg


# Use canonical class labels from data module to ensure consistency with training
CLASS_NAMES = CLASSES

LOAD_MODEL = os.environ.get("LOAD_MODEL", "1") == "1"


model: torch.nn.Module | None = None
feature_extractor: FeatureExtractor | None = None
norm_stats: dict[str, Any] | None = None

tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# ----------------------------
# System metrics (Prometheus)
# ----------------------------
PROC = psutil.Process()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_cpu_percent = Gauge("system_cpu_percent", "System CPU utilization percent")
system_memory_percent = Gauge("system_memory_percent", "System memory utilization percent")
process_rss_bytes = Gauge("process_rss_bytes", "Process resident set size in bytes")


async def _metrics_loop(stop_event: asyncio.Event, interval_s: float = 5.0) -> None:
    """Background loop updating system/process gauges every `interval_s` seconds."""
    # Prime CPU calculation to avoid a misleading first sample
    psutil.cpu_percent(interval=None)

    while not stop_event.is_set():
        system_cpu_percent.set(psutil.cpu_percent(interval=None))
        system_memory_percent.set(psutil.virtual_memory().percent)
        process_rss_bytes.set(PROC.memory_info().rss)

        # Sleep or exit early if stop_event is set
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


DRIFT_CURRENT_PATH = Path(os.environ.get("DRIFT_CURRENT_PATH", "data/drift/current.csv"))
_write_lock = Lock()


def _img_to_np_for_stats(img_rgb: Image.Image) -> np.ndarray:
    return np.asarray(img_rgb.convert("L"), dtype=np.float32)


def _compute_stats(arr: np.ndarray) -> dict:
    flat = arr.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
    }


def _append_row_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    with _write_lock:
        if not path.exists() or path.stat().st_size == 0:
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, mode="a", header=False, index=False)


def init_feedback_db() -> None:
    """Initialize feedback database on startup."""
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            is_correct BOOLEAN NOT NULL,
            correct_class TEXT,
            user_note TEXT,
            confidence_rating TEXT,
            image_stats TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Feedback database initialized at %s", FEEDBACK_DB)


def validate_medical_image(img: Image.Image) -> tuple[bool, str | None]:
    """Validate image meets medical imaging requirements.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check dimensions
    if img.size[0] < 64 or img.size[1] < 64:
        return False, "Image too small (minimum 64x64 pixels required)"

    if img.size[0] > 4096 or img.size[1] > 4096:
        return False, "Image too large (maximum 4096x4096 pixels)"

    # Check if image is likely a CT scan (grayscale intensity check)
    gray = img.convert("L")
    arr = np.array(gray)

    # CT scans typically have good contrast
    if arr.std() < 5:
        return False, "Image appears to have insufficient contrast for CT analysis"

    # Check for blank/uniform images
    unique_values = len(np.unique(arr))
    if unique_values < 10:
        return False, "Image appears to be blank or uniform"

    return True, None


def generate_gradcam_heatmap(
    model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int | None = None
) -> Image.Image:
    """Generate GradCAM heatmap for model explanation.

    Args:
        model: The neural network model
        input_tensor: Preprocessed input tensor [1, 3, 224, 224]
        target_class: Target class index (if None, uses predicted class)

    Returns:
        PIL Image of the heatmap overlay
    """
    model.eval()

    # Get the last convolutional layer
    # For ResNet18/SimpleCNN, typically the last conv layer before pooling
    target_layer = None
    if hasattr(model, "features"):
        # SimpleCNN or similar architecture
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break
    elif hasattr(model, "cnn_pathway") and hasattr(model.cnn_pathway, "features"):
        # DualPathway model
        for layer in reversed(list(model.cnn_pathway.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break

    if target_layer is None:
        # Fallback: create a simple attention map based on gradient magnitude
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Use gradient magnitude as attention
        grads = input_tensor.grad.data.abs()
        attention = grads.mean(dim=1).squeeze().cpu().numpy()

        # Normalize
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    else:
        # Proper GradCAM implementation
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            output = model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()

            # GradCAM calculation
            pooled_gradients = gradients[0].mean(dim=[2, 3], keepdim=True)
            activation = activations[0]
            weighted_activation = (pooled_gradients * activation).sum(dim=1).squeeze()

            # ReLU and normalize
            attention = F.relu(weighted_activation).cpu().numpy()
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        finally:
            forward_handle.remove()
            backward_handle.remove()

    # Resize attention map to input size
    from scipy.ndimage import zoom

    attention_resized = zoom(attention, (224 / attention.shape[0], 224 / attention.shape[1]))

    # Create heatmap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.imshow(attention_resized, cmap="jet", alpha=0.8)
    plt.axis("off")

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)

    return Image.open(buf)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, feature_extractor, norm_stats

    logger.info("Startup config_path=%s model_path_env=%s", CONFIG_PATH, MODEL_PATH_ENV)
    logger.info("Startup cwd=%s", Path.cwd())
    logger.info("Config exists=%s", CONFIG_PATH.exists())
    logger.info("Mount /models exists=%s is_dir=%s", Path("/models").exists(), Path("/models").is_dir())
    logger.info("Mount /gcs exists=%s is_dir=%s", Path("/gcs").exists(), Path("/gcs").is_dir())

    models_path = Path("/models")
    try:
        models_entries = [p.name for p in models_path.glob("*")] if models_path.is_dir() else []
    except OSError as exc:
        logger.warning("Unable to list /models entries: %s", exc)
        models_entries = []
    logger.info("/models entries=%s", models_entries)

    model_path = resolve_model_path()
    logger.info("Resolved model_path=%s exists=%s", model_path, model_path.exists())

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config: {CONFIG_PATH}. Set CONFIG_PATH or add the default file.")
    if not model_path.exists():
        raise RuntimeError(f"Missing model weights: {model_path}. Set MODEL_PATH or include weights in the image.")

    # Initialize feedback database
    init_feedback_db()

    # Load feature metadata if available
    metadata_path = FEATURE_METADATA_PATH
    if not metadata_path.exists():
        # Fallback to local processed data path for development
        fallback_path = Path("data/processed/feature_metadata.json")
        if fallback_path.exists():
            logger.info("Feature metadata not found at %s, using fallback: %s", metadata_path, fallback_path)
            metadata_path = fallback_path

    if metadata_path.exists():
        logger.info("Loading feature metadata from %s", metadata_path)
        try:
            with metadata_path.open("r") as f:
                metadata = json.load(f)

            # Initialize feature extractor with config from metadata
            if "config" in metadata:
                # Use from_dict which handles validation and defaults
                feature_config = FeatureConfig.from_dict(metadata["config"])
                # If selected_features is in metadata (top-level) or config, ensure it's set
                # The metadata['feature_names'] is the ground truth of what was trained on
                if "feature_names" in metadata:
                    feature_config.selected_features = metadata["feature_names"]

                feature_extractor = FeatureExtractor(config=feature_config)
                logger.info("Initialized FeatureExtractor with %d features", feature_extractor.feature_dim)

            if "normalization" in metadata:
                norm_stats = metadata["normalization"]
                logger.info("Loaded normalization stats")
        except Exception as e:
            logger.error("Failed to load feature metadata: %s", e)
    else:
        logger.warning(
            "Feature metadata not found at %s. Radiomics features will not be available.", FEATURE_METADATA_PATH
        )

    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg)

    # Load model weights securely with weights_only=True
    try:
        if model_path.suffix == ".pt":
            logger.info("Loading inference-ready .pt weights (secure mode)")
        else:
            logger.info("Loading .ckpt checkpoint with secure loading (weights_only=True)")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.error(
            "Failed to load model weights with weights_only=True: %s. "
            "Please convert checkpoint to .pt format using promote_model.convert_ckpt_to_pt()",
            e,
        )
        raise RuntimeError(
            f"Cannot load model from {model_path}. Use convert_ckpt_to_pt() to create a .pt file."
        ) from e

    # Handle PyTorch Lightning checkpoint structure
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Remove "model." prefix if present (added by LightningModule wrapper)
    if isinstance(state, dict) and all(k.startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    # Start system metrics background loop
    stop_event = asyncio.Event()
    task = asyncio.create_task(_metrics_loop(stop_event))

    yield

    # Shutdown: stop background task + cleanup model
    stop_event.set()
    task.cancel()
    with suppress(Exception):
        await task

    model = None
    feature_extractor = None
    norm_stats = None


error_counter = Counter("prediction_error", "Number of prediction errors")
prediction_counter = Counter("prediction_total", "Total number of predictions")
prediction_histogram = Gauge("prediction_confidence", "Prediction confidence distribution")

app = FastAPI(
    title="CT Scan Classification API",
    description="""
    ## Medical Imaging AI Service

    Classify chest CT scans for lung cancer detection using dual-pathway deep learning.

    ### Model Classes
    - **adenocarcinoma**: Glandular tissue cancer
    - **large_cell_carcinoma**: Undifferentiated large cell cancer
    - **squamous_cell_carcinoma**: Squamous epithelium cancer
    - **normal**: No cancer detected

    ### Workflow
    1. Upload CT scan via `/predict` endpoint
    2. Review confidence scores and probability distribution
    3. Request explanation via `/explain` endpoint (optional)
    4. Submit feedback for continuous improvement via `/feedback`

    ### Batch Processing
    Use `/predict/batch` to process multiple images efficiently.

    ⚠️ **Medical Disclaimer**: This is a research tool for educational and experimental purposes.
    All predictions must be validated by qualified medical professionals.
    This system should not be used for clinical diagnosis without proper validation.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.mount("/metrics", make_asgi_app())

# HTTP metrics + /metrics endpoint (Prometheus)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health() -> dict:
    """Check API health status and model availability.

    Returns comprehensive system information including model configuration,
    device availability, and feature extraction capabilities.
    """
    return {
        "status": "healthy",
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "model_type": "dual_pathway" if isinstance(model, DualPathwayModel) else "single_pathway",
        "features_loaded": feature_extractor is not None,
        "feature_dim": feature_extractor.feature_dim if feature_extractor else 0,
        "config_path": str(CONFIG_PATH),
        "model_path": str(resolve_model_path()),
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def _process_single_image(img_bytes: bytes, filename: str = "upload") -> dict[str, Any]:
    """Internal helper to process a single image and return prediction results."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image '{filename}': {e}") from e

    # Validate image
    is_valid, error_msg = validate_medical_image(img)
    if not is_valid:
        raise HTTPException(status_code=422, detail=f"Image validation failed: {error_msg}")

    arr = _img_to_np_for_stats(img)
    stats = _compute_stats(arr)

    x = tfm(img).unsqueeze(0).to(DEVICE)

    # Radiomics feature extraction (if enabled/required)
    features_tensor = None
    if isinstance(model, DualPathwayModel) and feature_extractor is not None and norm_stats is not None:
        try:
            # Extract features from preprocessed tensor (same as training)
            img_preprocessed = x.squeeze(0).cpu().numpy()  # [3, 224, 224]
            features = feature_extractor.extract(img_preprocessed)

            # Normalize
            mean = np.array(norm_stats["mean"], dtype=np.float32)
            std = np.array(norm_stats["std"], dtype=np.float32)
            # Avoid division by zero if std is 0
            std = np.where(std > 1e-8, std, 1.0)

            features = (features - mean) / std

            # Prepare tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}") from e

    with torch.no_grad():
        if features_tensor is not None and isinstance(model, DualPathwayModel):
            logits = model(x, features=features_tensor)
        else:
            logits = model(x)

        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        pred_conf = float(probs[pred].item())

    if not (0 <= pred < len(CLASS_NAMES)):
        raise HTTPException(
            status_code=500,
            detail=f"Model predicted invalid class index {pred}; expected 0-{len(CLASS_NAMES) - 1}",
        )

    pred_class = CLASS_NAMES[pred]

    # Log to drift monitoring CSV
    try:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            **stats,
            "pred_index": pred,
            "pred_class": pred_class,
            "pred_conf": pred_conf,
        }
        _append_row_csv(DRIFT_CURRENT_PATH, row)
    except Exception:
        pass

    # Update metrics
    prediction_counter.inc()
    prediction_histogram.set(pred_conf)

    return {
        "prediction": {
            "class": pred_class,
            "class_index": pred,
            "confidence": pred_conf,
        },
        "probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))},
        "metadata": {
            "model_type": "dual_pathway" if isinstance(model, DualPathwayModel) else "single_pathway",
            "features_used": features_tensor is not None,
            "device": str(DEVICE),
            "timestamp": datetime.now(UTC).isoformat(),
            "image_stats": stats,
        },
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict:
    """Classify a CT scan image with enhanced response including confidence scores.

    Returns detailed prediction results including:
    - Primary prediction with confidence
    - Probability distribution across all classes
    - Model metadata and processing information
    - Image statistics for drift monitoring

    Args:
        file: CT scan image file (PNG, JPG, JPEG)

    Returns:
        Dictionary with prediction, probabilities, and metadata

    Raises:
        HTTPException: If model not loaded, invalid image, or validation fails
    """
    img_bytes = await file.read()
    return await _process_single_image(img_bytes, file.filename or "upload")


@app.post("/predict/batch")
async def predict_batch(
    files: Annotated[list[UploadFile], File(...)],
    max_batch_size: int = 20,
) -> dict:
    """Process multiple CT scans in a single batch request.

    Efficiently processes multiple images and returns individual results.
    Failed images are reported but don't stop batch processing.

    Args:
        files: List of CT scan image files
        max_batch_size: Maximum number of images per batch (default: 20)

    Returns:
        Dictionary with batch summary and individual results

    Raises:
        HTTPException: If batch size exceeds maximum
    """
    if len(files) > max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size {len(files)} exceeds maximum {max_batch_size}")

    results = []
    for file in files:
        try:
            img_bytes = await file.read()
            result = await _process_single_image(img_bytes, file.filename or "upload")
            results.append({"filename": file.filename, "success": True, **result})
        except HTTPException as e:
            results.append({"filename": file.filename, "success": False, "error": e.detail})
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    successful = sum(1 for r in results if r.get("success", False))

    return {
        "batch_summary": {
            "total": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "results": results,
    }


@app.post("/explain")
async def explain_prediction(file: Annotated[UploadFile, File(...)]) -> dict:
    """Generate explainability heatmap (GradCAM) for a prediction.

    Creates a visual explanation showing which regions of the image
    the model focused on when making its prediction.

    Args:
        file: CT scan image file

    Returns:
        Dictionary with prediction, confidence, and base64-encoded heatmap

    Raises:
        HTTPException: If model not loaded or explainability generation fails
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # Validate image
    is_valid, error_msg = validate_medical_image(img)
    if not is_valid:
        raise HTTPException(status_code=422, detail=f"Image validation failed: {error_msg}")

    x = tfm(img).unsqueeze(0).to(DEVICE)

    try:
        # Get prediction first
        with torch.no_grad():
            if isinstance(model, DualPathwayModel) and feature_extractor is not None and norm_stats is not None:
                # Extract features
                img_preprocessed = x.squeeze(0).cpu().numpy()
                features = feature_extractor.extract(img_preprocessed)
                mean = np.array(norm_stats["mean"], dtype=np.float32)
                std = np.array(norm_stats["std"], dtype=np.float32)
                std = np.where(std > 1e-8, std, 1.0)
                features = (features - mean) / std
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits = model(x, features=features_tensor)
            else:
                logits = model(x)

            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            pred_conf = float(probs[pred_idx].item())

        # Generate GradCAM heatmap
        heatmap_img = generate_gradcam_heatmap(model, x, target_class=pred_idx)

        # Encode heatmap as base64
        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "prediction": {
                "class": CLASS_NAMES[pred_idx],
                "class_index": pred_idx,
                "confidence": pred_conf,
            },
            "explanation": {
                "heatmap": heatmap_b64,
                "description": "Highlighted regions show areas the model focused on for this prediction. "
                "Warmer colors (red/yellow) indicate higher importance.",
                "method": "GradCAM",
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
    except Exception as e:
        logger.error("Explainability generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Explainability generation failed: {e}") from e


@app.post("/feedback")
async def feedback(
    file: Annotated[UploadFile, File(...)],
    predicted_class: Annotated[str, Form(...)],
    predicted_confidence: Annotated[float, Form(...)] = 0.0,
    is_correct: Annotated[bool, Form(...)] = True,
    correct_class: Annotated[str | None, Form()] = None,
    user_note: Annotated[str | None, Form()] = None,
    confidence_rating: Annotated[str | None, Form()] = None,
) -> dict:
    """Save user feedback for continuous model improvement.

    Stores both the image and structured metadata in a database for future retraining.

    Args:
        file: The CT scan image
        predicted_class: Class predicted by the model
        predicted_confidence: Model's confidence score
        is_correct: Whether the prediction was correct
        correct_class: Actual class if prediction was incorrect
        user_note: Optional user comments
        confidence_rating: User's confidence in their assessment

    Returns:
        Confirmation with save location and database logging status
    """
    if predicted_class not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid predicted_class")

    if not is_correct:
        error_counter.inc()
        if correct_class is None:
            raise HTTPException(status_code=400, detail="correct_class is required when is_correct is false")
        if correct_class not in CLASS_NAMES:
            raise HTTPException(status_code=400, detail="Invalid correct_class")
        target_class = correct_class
    else:
        target_class = predicted_class

    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # Compute image statistics
    arr = _img_to_np_for_stats(img)
    stats = _compute_stats(arr)

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = FEEDBACK_DIR / target_class
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "upload").stem
    safe_name = "".join(ch for ch in original_name if ch.isalnum() or ch in {"-", "_"}) or "image"
    filename = f"{safe_name}-{uuid.uuid4().hex}.png"
    save_path = target_dir / filename
    img.save(save_path, format="PNG")

    # Log to database
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        conn.execute(
            """INSERT INTO feedback
               (timestamp, image_path, predicted_class, predicted_confidence,
                is_correct, correct_class, user_note, confidence_rating, image_stats)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                str(save_path),
                predicted_class,
                predicted_confidence,
                is_correct,
                correct_class,
                user_note,
                confidence_rating,
                json.dumps(stats),
            ),
        )
        conn.commit()
        conn.close()
        db_logged = True
    except Exception as e:
        logger.error("Failed to log feedback to database: %s", e)
        db_logged = False

    return {
        "saved_to": str(save_path),
        "class": target_class,
        "logged_to_db": db_logged,
        "feedback_id": None,  # Could return last_insert_rowid if needed
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/feedback/stats")
def feedback_stats() -> dict:
    """Get statistics about collected feedback.

    Returns summary of feedback data including total count, accuracy metrics,
    and class distribution.
    """
    try:
        conn = sqlite3.connect(FEEDBACK_DB)
        cursor = conn.cursor()

        # Total feedback count
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_count = cursor.fetchone()[0]

        # Correct vs incorrect
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_correct = 1")
        correct_count = cursor.fetchone()[0]

        # Class distribution
        cursor.execute("SELECT predicted_class, COUNT(*) FROM feedback GROUP BY predicted_class")
        class_distribution = dict(cursor.fetchall())

        # Recent feedback
        cursor.execute("SELECT timestamp, predicted_class, is_correct FROM feedback ORDER BY timestamp DESC LIMIT 10")
        recent_feedback = [
            {"timestamp": row[0], "predicted_class": row[1], "is_correct": bool(row[2])} for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "total_feedback": total_count,
            "correct_predictions": correct_count,
            "incorrect_predictions": total_count - correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "class_distribution": class_distribution,
            "recent_feedback": recent_feedback,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error("Failed to retrieve feedback stats: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback stats: {e}") from e
