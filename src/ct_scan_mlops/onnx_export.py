from pathlib import Path

import torch
from omegaconf import OmegaConf

from ct_scan_mlops.model import build_model

DEVICE = torch.device("cpu")


def main():
    run_dir = Path("outputs/2026-01-15/11-57-23")

    config_path = run_dir / ".hydra" / "config.yaml"
    model_path = run_dir / "model.pt"
    onnx_path = run_dir / "model.onnx"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cfg = OmegaConf.load(config_path)

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(DEVICE)

    image_size = cfg.data.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size, device=DEVICE)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
        )

    print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
