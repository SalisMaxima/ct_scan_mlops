"""Export DualPathway model to ONNX format."""

import argparse
from pathlib import Path

import torch
import torch.onnx
from omegaconf import OmegaConf

from ct_scan_mlops.model import build_model

# Default paths
DEFAULT_CONFIG = Path("configs/model/dual_pathway_bn_finetune_kygevxv0_clean.yaml")
DEFAULT_MODEL = Path("models/dual_pathway_bn_finetune_kygevxv0.pt")
OUTPUT_ONNX = Path("models/dual_pathway.onnx")


def export_model(config_path: Path, model_path: Path, output_path: Path) -> None:
    """Export PyTorch model to ONNX."""
    print(f"Loading config from {config_path}...")
    # Load config directly since it's a full config file
    cfg = OmegaConf.load(config_path)

    print(f"Building model: {cfg.model.name}...")
    model = build_model(cfg)

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Handle Lightning checkpoint if needed (though .pt usually is state_dict)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove "model." prefix if present
    if all(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy inputs
    # Image: (Batch, Channels, Height, Width)
    dummy_image = torch.randn(1, 3, 224, 224)

    # Features: (Batch, FeatureDim)
    # Get dim from config or use 16 (default for this model)
    feature_dim = cfg.model.get("radiomics_dim", 16)
    dummy_features = torch.randn(1, feature_dim)

    print(f"Exporting to {output_path}...")
    print(f"Input shapes: Image {dummy_image.shape}, Features {dummy_features.shape}")

    torch.onnx.export(
        model,
        (dummy_image, dummy_features),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image", "features"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print("Export complete!")

    # Verify
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DualPathway model to ONNX")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .pt model file")
    parser.add_argument("--output", type=Path, default=OUTPUT_ONNX, help="Output ONNX file path")

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    if not args.model.exists():
        print(f"Error: Model file not found at {args.model}")
        exit(1)

    export_model(args.config, args.model, args.output)
