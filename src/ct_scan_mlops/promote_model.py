"""Promote model to production in W&B Model Registry.

Following DTU MLOps course CML patterns:
https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/cml/

This module also provides utilities to convert PyTorch Lightning checkpoints
(.ckpt) to production-ready state dictionaries (.pt) for secure, fast loading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf


def convert_ckpt_to_pt(ckpt_path: Path, pt_path: Path) -> None:
    """Convert a PyTorch Lightning checkpoint to a clean state_dict file.

    This strips optimizer states, scheduler states, and other training metadata
    from the checkpoint, keeping only the model weights. The resulting .pt file
    is smaller, loads faster, and can be loaded securely with weights_only=True.

    Args:
        ckpt_path: Path to the input .ckpt file (PyTorch Lightning checkpoint).
        pt_path: Path where the output .pt file will be saved.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        RuntimeError: If the checkpoint cannot be loaded or converted.
    """
    if not ckpt_path.exists():
        msg = f"Checkpoint file not found: {ckpt_path}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading checkpoint from {ckpt_path}")

    try:
        # We trust our own training artifacts, so we load with weights_only=False.
        # We add '# nosec' to ignore Bandit security warnings (B301/B614) about unsafe deserialization.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # nosec
        logger.info("Loaded checkpoint (trusted mode)")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Cannot load checkpoint from {ckpt_path}") from e

    # Extract state_dict from the checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        # Fallback: assume the dict itself is the state_dict
        state_dict = checkpoint
    else:
        msg = f"Unexpected checkpoint format: {type(checkpoint)}"
        raise RuntimeError(msg)

    # Remove "model." prefix added by LightningModule wrapper
    # (Lightning wraps the model as self.model, so keys become "model.layer.weight")
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "", 1) if key.startswith("model.") else key
        clean_state_dict[new_key] = value

    # Save clean state_dict (can be loaded with weights_only=True)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_state_dict, pt_path)

    bytes_per_mb = 1024 * 1024
    ckpt_size_mb = ckpt_path.stat().st_size / bytes_per_mb
    pt_size_mb = pt_path.stat().st_size / bytes_per_mb
    logger.success(f"Converted {ckpt_path.name} ({ckpt_size_mb:.1f}MB) -> {pt_path.name} ({pt_size_mb:.1f}MB)")


def promote_to_production(model_path: str, project: str) -> None:
    """Promote a model artifact to production alias in W&B.

    Args:
        model_path: W&B artifact path (e.g., entity/project/model:staging)
        project: W&B project name
    """
    logger.info(f"Promoting model to production: {model_path}")

    # Initialize wandb
    run = wandb.init(project=project, job_type="model-promotion", mode="online")

    try:
        # Get the artifact
        artifact = run.use_artifact(model_path, type="model")
        logger.info(f"Found artifact: {artifact.name} (version: {artifact.version})")

        # Get the artifact's collection (registered model)
        # The artifact path format is: entity/project/artifact_name:alias_or_version
        artifact_name = artifact.name.split(":")[0]  # Remove version/alias

        # Link to model registry with production alias
        # UPDATE: Changed target_path to support new W&B Registry (Core)
        # "wandb-registry-model" is the default registry name after migration.
        # If your registry is named differently, update "model" below.
        target_path = f"wandb-registry-model/{artifact_name}"

        logger.info(f"Linking to registry path: {target_path}")
        run.link_artifact(artifact, target_path=target_path, aliases=["production"])

        logger.success("Model promoted to production!")
        logger.info(f"  - Artifact: {artifact.name}")
        logger.info(f"  - Version: {artifact.version}")
        logger.info("  - New alias: production")

        # Log promotion event
        wandb.log(
            {
                "promotion_event": 1,
                "promoted_artifact": artifact.name,
                "promoted_version": artifact.version,
            }
        )

    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="Promote model to production in W&B")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="W&B artifact path (e.g., entity/project/model:staging)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (default: from config.yaml)",
    )
    args = parser.parse_args()

    # Load config to get wandb project if not provided
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    wandb_project = args.wandb_project or cfg.wandb.project

    try:
        promote_to_production(args.model_path, wandb_project)
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
