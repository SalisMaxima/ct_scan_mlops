from __future__ import annotations

import io
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Deployment-friendly paths (no dependence on outputs/<date>/<time>) ---
# You can override these in Cloud Run via env vars: CONFIG_PATH, MODEL_PATH
DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"  # change if your file name differs
DEFAULT_MODEL_PATH = Path("models") / "model.pt"  # change if your weights live elsewhere

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "feedback"))

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

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config: {CONFIG_PATH}. Set CONFIG_PATH or add the default file.")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model weights: {MODEL_PATH}. Set MODEL_PATH or include weights in the image.")

    cfg = OmegaConf.load(CONFIG_PATH)
    model = build_model(cfg)

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

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
