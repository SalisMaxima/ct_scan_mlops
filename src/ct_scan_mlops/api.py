from __future__ import annotations

import asyncio
import io
import json
import logging
import os
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

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"  # change if your file name differs
DEFAULT_CKPT_PATH = Path("models") / "best_model.ckpt"
DEFAULT_PT_PATH = Path("models") / "model.pt"
DEFAULT_FEATURE_METADATA_PATH = Path("models") / "feature_metadata.json"

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_PT_PATH)))
FEATURE_METADATA_PATH = Path(os.environ.get("FEATURE_METADATA_PATH", str(DEFAULT_FEATURE_METADATA_PATH)))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "feedback"))
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

app = FastAPI(title="CT Scan Inference API", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

# HTTP metrics + /metrics endpoint (Prometheus)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health() -> dict:
    """Check API health status and model availability."""
    return {
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "features_loaded": feature_extractor is not None,
        "config_path": str(CONFIG_PATH),
        "model_path": str(resolve_model_path()),
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict:
    """Classify a CT scan image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    arr = _img_to_np_for_stats(img)
    stats = _compute_stats(arr)

    x = tfm(img).unsqueeze(0).to(DEVICE)

    # Radiomics feature extraction (if enabled/required)
    features_tensor = None
    if isinstance(model, DualPathwayModel) and feature_extractor is not None and norm_stats is not None:
        try:
            # Extract features from raw numpy image
            # Convert PIL to numpy (use L/grayscale as used in training typically)
            img_np = np.array(img.convert("L"))
            features = feature_extractor.extract(img_np)

            # Normalize
            mean = np.array(norm_stats["mean"], dtype=np.float32)
            std = np.array(norm_stats["std"], dtype=np.float32)
            # Avoid division by zero if std is 0 (should be handled in training/extraction)
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

    return {
        "pred_index": pred,
        "pred_class": CLASS_NAMES[pred],
    }


@app.post("/feedback")
async def feedback(
    file: Annotated[UploadFile, File(...)],
    predicted_class: Annotated[str, Form(...)],
    is_correct: Annotated[bool, Form(...)],
    correct_class: Annotated[str | None, Form()] = None,
) -> dict:
    """Save feedback image into a class-named folder."""
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

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = FEEDBACK_DIR / target_class
    target_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "upload").stem
    safe_name = "".join(ch for ch in original_name if ch.isalnum() or ch in {"-", "_"}) or "image"
    filename = f"{safe_name}-{uuid.uuid4().hex}.png"
    save_path = target_dir / filename
    img.save(save_path, format="PNG")

    return {
        "saved_to": str(save_path),
        "class": target_class,
    }
