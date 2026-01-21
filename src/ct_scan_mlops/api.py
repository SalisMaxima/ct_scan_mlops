# src/ct_scan_mlops/api.py
from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from omegaconf import OmegaConf
from PIL import Image
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from torchvision import transforms

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model/config location
# ----------------------------
# Default local path (works locally after training)
RUN_DIR = Path(os.environ.get("RUN_DIR", "outputs/2026-01-15/11-57-23"))
CONFIG_PATH = RUN_DIR / ".hydra" / "config.yaml"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.pt")
MODEL_PATH = RUN_DIR / MODEL_FILENAME

# GCS fallbacks (recommended for Cloud Run)
# Example:
#   CONFIG_GCS_URI=gs://dtu-mlops-dvc-storage-482907/ct_scan_mlops/serving/config.yaml
#   MODEL_GCS_URI=gs://dtu-mlops-dvc-storage-482907/ct_scan_mlops/serving/model.pt
CONFIG_GCS_URI = os.environ.get("CONFIG_GCS_URI", "")
MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")

# Where we download artifacts inside the container
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp/ct_scan_mlops"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCS_TIMEOUT_S = float(os.environ.get("GCS_TIMEOUT_S", "120"))

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

# ----------------------------
# System metrics (Prometheus)
# ----------------------------
PROC = psutil.Process()

system_cpu_percent = Gauge("system_cpu_percent", "System CPU utilization percent")
system_memory_percent = Gauge("system_memory_percent", "System memory utilization percent")
process_rss_bytes = Gauge("process_rss_bytes", "Process resident set size in bytes")


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri.removeprefix("gs://")
    bucket, _, blob = no_scheme.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI (need bucket/path): {uri}")
    return bucket, blob


def _download_from_gcs(gs_uri: str, dest: Path) -> None:
    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    client = storage.Client()  # uses ADC on Cloud Run
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dest, timeout=GCS_TIMEOUT_S)


async def _metrics_loop(stop_event: asyncio.Event, interval_s: float = 5.0) -> None:
    # Prime CPU calculation
    psutil.cpu_percent(interval=None)

    while not stop_event.is_set():
        system_cpu_percent.set(psutil.cpu_percent(interval=None))
        system_memory_percent.set(psutil.virtual_memory().percent)
        process_rss_bytes.set(PROC.memory_info().rss)

        # sleep until next tick OR stop_event
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model

    # If local files missing, try fetching from GCS (Cloud Run path)
    effective_config_path = CONFIG_PATH
    effective_model_path = MODEL_PATH

    if not effective_config_path.exists():
        if not CONFIG_GCS_URI:
            raise RuntimeError(
                f"Missing Hydra config: {effective_config_path}. "
                "Set CONFIG_GCS_URI to a gs://.../config.yaml to download it at startup."
            )
        effective_config_path = DOWNLOAD_DIR / "config.yaml"
        _download_from_gcs(CONFIG_GCS_URI, effective_config_path)

    if not effective_model_path.exists():
        if not MODEL_GCS_URI:
            raise RuntimeError(
                f"Missing model weights: {effective_model_path}. "
                "Set MODEL_GCS_URI to a gs://.../model.pt to download it at startup."
            )
        effective_model_path = DOWNLOAD_DIR / "model.pt"
        _download_from_gcs(MODEL_GCS_URI, effective_model_path)

    cfg = OmegaConf.load(effective_config_path)
    model = build_model(cfg)

    state = torch.load(effective_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # Start system metrics background loop
    stop_event = asyncio.Event()
    task = asyncio.create_task(_metrics_loop(stop_event))

    yield

    stop_event.set()
    task.cancel()
    with suppress(Exception):
        await task

    del model


app = FastAPI(title="CT Scan Inference API", version="0.1.0", lifespan=lifespan)

# HTTP metrics + /metrics endpoint (Prometheus)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health() -> dict:
    """Check API health status and model availability."""
    return {
        "ok": True,
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "run_dir": str(RUN_DIR),
        "config_path": str(CONFIG_PATH),
        "model_path": str(MODEL_PATH),
        "config_gcs_uri": CONFIG_GCS_URI or None,
        "model_gcs_uri": MODEL_GCS_URI or None,
        "download_dir": str(DOWNLOAD_DIR),
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

    return {"pred_index": pred, "pred_class": CLASS_NAMES[pred]}
