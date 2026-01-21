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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from torchvision import transforms

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Deployment-friendly paths (no dependence on outputs/<date>/<time>) ---
# You can override these in Cloud Run via env vars: CONFIG_PATH, MODEL_PATH
DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"  # change if your file name differs
DEFAULT_CKPT_PATH = Path("models") / "best_model.ckpt"
DEFAULT_PT_PATH = Path("models") / "model.pt"

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
MODEL_PATH_ENV = os.environ.get("MODEL_PATH")


def resolve_model_path() -> Path:
    """Resolve model path with .ckpt preferred, .pt as fallback.

    Env var MODEL_PATH takes precedence when set.
    """
    if MODEL_PATH_ENV:
        return Path(MODEL_PATH_ENV)
    if DEFAULT_CKPT_PATH.exists():
        return DEFAULT_CKPT_PATH
    return DEFAULT_PT_PATH


def load_config(cfg_path: Path):
    """Load a Hydra config, composing if needed."""
    cfg = OmegaConf.load(cfg_path)
    if "model" in cfg:
        return cfg

    if "defaults" in cfg:
        with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
            return compose(config_name=cfg_path.stem)

    return cfg


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model

    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config: {CONFIG_PATH}. Set CONFIG_PATH or add the default file.")
    model_path = resolve_model_path()
    if not model_path.exists():
        raise RuntimeError(f"Missing model weights: {model_path}. Set MODEL_PATH or include weights in the image.")

    cfg = load_config(CONFIG_PATH)
    model = build_model(cfg)

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
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
