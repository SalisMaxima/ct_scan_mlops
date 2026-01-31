"""Benchmark ONNX vs PyTorch inference speed and accuracy."""

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ct_scan_mlops.inference_onnx import ONNXPredictor
from ct_scan_mlops.model import build_model

CONFIG_PATH = Path("configs/model/dual_pathway_bn_finetune_kygevxv0_clean.yaml")
MODEL_PT_PATH = Path("models/dual_pathway_bn_finetune_kygevxv0.pt")
MODEL_ONNX_PATH = Path("models/dual_pathway.onnx")


def benchmark():
    # 1. Load PyTorch Model
    print("Loading PyTorch model...")
    cfg = OmegaConf.load(CONFIG_PATH)
    pt_model = build_model(cfg)
    state = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=True)

    if "state_dict" in state:
        state = state["state_dict"]
    if all(k.startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    pt_model.load_state_dict(state)
    pt_model.eval()

    # 2. Load ONNX Model
    print("Loading ONNX Model...")
    if not MODEL_ONNX_PATH.exists():
        print("ONNX model not found. Please run scripts/export_onnx.py first.")
        return

    onnx_predictor = ONNXPredictor(MODEL_ONNX_PATH)

    # 3. Create Dummy Input
    batch_size = 1
    rng = np.random.default_rng()
    dummy_img_np = rng.standard_normal((batch_size, 3, 224, 224), dtype=np.float32)
    dummy_feat_np = rng.standard_normal((batch_size, cfg.model.radiomics_dim), dtype=np.float32)

    dummy_img_pt = torch.from_numpy(dummy_img_np)
    dummy_feat_pt = torch.from_numpy(dummy_feat_np)

    # 4. Verify Correctness
    print("\nVerifying correctness...")
    with torch.no_grad():
        pt_logits = pt_model(dummy_img_pt, dummy_feat_pt).numpy()

    onnx_logits = onnx_predictor.predict(dummy_img_np, dummy_feat_np)

    diff = np.abs(pt_logits - onnx_logits).max()
    print(f"Max absolute difference: {diff:.6f}")
    if diff < 1e-4:
        print("✓ Accuracy verified (outputs match)")
    else:
        print("✗ Accuracy mismatch!")

    # 5. Benchmark Speed
    print("\nBenchmarking speed (100 iterations)...")
    n_iter = 100

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            pt_model(dummy_img_pt, dummy_feat_pt)
        onnx_predictor.predict(dummy_img_np, dummy_feat_np)

    # PyTorch
    start = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            pt_model(dummy_img_pt, dummy_feat_pt)
    pt_time = (time.time() - start) / n_iter * 1000
    print(f"PyTorch (CPU): {pt_time:.2f} ms/sample")

    # ONNX
    start = time.time()
    for _ in range(n_iter):
        onnx_predictor.predict(dummy_img_np, dummy_feat_np)
    onnx_time = (time.time() - start) / n_iter * 1000
    print(f"ONNX (CPU):    {onnx_time:.2f} ms/sample")

    speedup = pt_time / onnx_time
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
