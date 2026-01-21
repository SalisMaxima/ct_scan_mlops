# src/ct_scan_mlops/api.py
from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ct_scan_mlops.model import build_model

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Deployment-friendly paths (no dependence on outputs/<date>/<time>) ---
# You can override these in Cloud Run via env vars: CONFIG_PATH, MODEL_PATH
DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"  # change if your file name differs
DEFAULT_MODEL_PATH = Path("models") / "model.pt"  # change if your weights live elsewhere

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma",
]

model: torch.nn.Module | None = None
tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model

    # Configure logging to stdout so Cloud Run captures it
    # Only configure if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # If already configured, just set the level
        log_level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
        logging.getLogger().setLevel(log_level)

    logger.info(f"Startup: Checking config at {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        logger.error(f"Config not found at {CONFIG_PATH}")
        # List files in the parent directory to help debug (only at debug level)
        parent = CONFIG_PATH.parent
        if parent.exists():
            try:
                logger.debug(f"Contents of {parent}: {[p.name for p in parent.iterdir()]}")
            except Exception as e:
                logger.debug(f"Failed to list {parent}: {e}")
        raise RuntimeError(f"Missing config: {CONFIG_PATH}. Set CONFIG_PATH or add the default file.")

    logger.info(f"Startup: Checking model at {MODEL_PATH}")
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        # List contents of the mount point to debug GCS fuse (only at debug level)
        mount_point = MODEL_PATH.parent
        if mount_point.exists():
            try:
                logger.debug(f"Contents of {mount_point}: {[p.name for p in mount_point.iterdir()]}")
            except Exception as e:
                logger.debug(f"Failed to list {mount_point}: {e}")
        else:
            logger.debug(f"Mount point {mount_point} does not exist!")
        raise RuntimeError(f"Missing model weights: {MODEL_PATH}. Set MODEL_PATH or include weights in the image.")

    logger.info("Startup: Loading configuration...")
    cfg = OmegaConf.load(CONFIG_PATH)

    logger.info("Startup: Building model...")
    model = build_model(cfg)

    logger.info(f"Startup: Loading weights from {MODEL_PATH}...")
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        raise

    model.to(DEVICE)
    model.eval()
    logger.info("Startup: Model loaded successfully.")

    yield

    # Cleanup
    del model


app = FastAPI(title="CT Scan Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Check API health status and model availability."""
    return {
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "config_path": str(CONFIG_PATH),
        "model_path": str(MODEL_PATH),
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

    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    if not (0 <= pred < len(CLASS_NAMES)):
        raise HTTPException(
            status_code=500,
            detail=f"Model predicted invalid class index {pred}; expected 0-{len(CLASS_NAMES) - 1}",
        )

    return {
        "pred_index": pred,
        "pred_class": CLASS_NAMES[pred],
    }
