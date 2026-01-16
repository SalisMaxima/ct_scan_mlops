# src/ct_scan_mlops/api.py
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ct_scan_mlops.model import build_model

app = FastAPI(title="CT Scan Inference API", version="0.1.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Point this to the training run folder that contains:
#   - .hydra/config.yaml
#   - model.pt   (or change MODEL_FILENAME below)
RUN_DIR = Path(os.environ.get("RUN_DIR", "outputs/2026-01-15/11-57-23"))
CONFIG_PATH = RUN_DIR / ".hydra" / "config.yaml"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.pt")
MODEL_PATH = RUN_DIR / MODEL_FILENAME

model: torch.nn.Module | None = None
tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


@app.on_event("startup")
def load() -> None:
    global model

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing Hydra config: {CONFIG_PATH}")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model weights: {MODEL_PATH}")

    cfg = OmegaConf.load(CONFIG_PATH)
    model = build_model(cfg)

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "run_dir": str(RUN_DIR),
        "config_path": str(CONFIG_PATH),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
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

    return {"pred_index": pred}
