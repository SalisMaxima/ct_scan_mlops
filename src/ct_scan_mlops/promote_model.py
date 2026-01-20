"""Promote model to production in W&B Model Registry.

Following DTU MLOps course CML patterns:
https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/cml/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb
from loguru import logger
from omegaconf import OmegaConf


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
        # This adds the 'production' alias to the artifact
        run.link_artifact(artifact, target_path=f"model-registry/{artifact_name}", aliases=["production"])

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
