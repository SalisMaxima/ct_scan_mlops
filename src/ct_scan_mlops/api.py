from __future__ import annotations
import io
from contextlib import asynccontextmanager
from typing import Annotated
from datetime import datetime, timezone
from pathlib import Path
import os
from threading import Lock

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml" 
DEFAULT_MODEL_PATH = Path("models") / "model.pt"  

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma",
]

LOAD_MODEL = os.environ.get("LOAD_MODEL", "1") == "1"


model: torch.nn.Module | None = None
tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


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
    global model

    if not LOAD_MODEL:
        model = None
        yield
        return

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
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    arr = _img_to_np_for_stats(img)
    stats = _compute_stats(arr)

    if model is None:
        pred = 0
        pred_conf = 1.0
        pred_class = CLASS_NAMES[pred]
    else:
        x = tfm(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        "pred_class": pred_class,
        "pred_conf": pred_conf,
    }


