"""W&B sweep-compatible training entrypoint.

Why this exists:
- W&B Sweeps pass hyperparameters as CLI flags like `--lr=...`.
- Hydra expects overrides like `train.optimizer.lr=...` (without `--`).

This module provides a simple CLI that maps sweep parameters into Hydra config
overrides, then calls the existing Lightning training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import typer
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger

from ct_scan_mlops.train import configure_logging, train_model

# Find project root for Hydra config path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = str(_PROJECT_ROOT / "configs")


def _resolve_output_base(cfg: DictConfig) -> Path:
    """Resolve the output base directory from config.

    Handles both absolute and relative paths correctly.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    out = resolved.get("output_dir", "outputs") if isinstance(resolved, dict) else "outputs"
    out_path = Path(out)
    if out_path.is_absolute():
        return out_path
    return (_PROJECT_ROOT / out_path).resolve()


def sweep_train(
    lr: float = typer.Option(None, help="Learning rate (maps to train.optimizer.lr)"),
    weight_decay: float = typer.Option(
        None,
        "--weight_decay",
        "--weight-decay",
        help="Weight decay (maps to train.optimizer.weight_decay)",
    ),
    batch_size: int = typer.Option(
        None,
        "--batch_size",
        "--batch-size",
        help="Batch size (maps to data.batch_size)",
    ),
    model: str = typer.Option(
        None, help="Model config group: cnn | resnet18 | dual_pathway | dual_pathway_top_features"
    ),
    features: str = typer.Option(None, help="Features config: default | top_features"),
    max_epochs: int = typer.Option(
        None,
        "--max_epochs",
        "--max-epochs",
        help="Max epochs (maps to train.max_epochs)",
    ),
    seed: int = typer.Option(None, help="Random seed"),
    wandb_project: str = typer.Option(
        None,
        "--wandb_project",
        "--wandb-project",
        help="Override W&B project",
    ),
    wandb_entity: str = typer.Option(
        None,
        "--wandb_entity",
        "--wandb-entity",
        help="Override W&B entity (team/user)",
    ),
    wandb_mode: str = typer.Option(
        None,
        "--wandb_mode",
        "--wandb-mode",
        help="online | offline | disabled",
    ),
    disable_profiling: str = typer.Option(
        "True",
        "--disable_profiling",
        "--disable-profiling",
        help="Disable the one-time PyTorch profiler run",
    ),
    dropout: float = typer.Option(None, help="Dropout rate (maps to model.dropout)"),
    fusion_hidden: int = typer.Option(
        None,
        "--fusion_hidden",
        "--fusion-hidden",
        help="Fusion hidden dim (maps to model.fusion_hidden)",
    ),
    radiomics_hidden: int = typer.Option(
        None,
        "--radiomics_hidden",
        "--radiomics-hidden",
        help="Radiomics hidden dim (maps to model.radiomics_hidden)",
    ),
    freeze_backbone: str = typer.Option(
        None,
        "--freeze_backbone",
        "--freeze-backbone",
        help="Freeze backbone (maps to model.freeze_backbone)",
    ),
    eta_min: float = typer.Option(
        None,
        "--eta_min",
        "--eta-min",
        help="Min LR (maps to train.scheduler.eta_min)",
    ),
) -> None:
    """Train one run (designed to be launched by `wandb agent`)."""

    overrides: list[str] = []

    if model:
        overrides.append(f"model={model}")

    if features:
        overrides.append(f"features={features}")

    if lr is not None:
        overrides.append(f"train.optimizer.lr={lr}")

    if weight_decay is not None:
        overrides.append(f"train.optimizer.weight_decay={weight_decay}")

    if batch_size is not None:
        overrides.append(f"data.batch_size={batch_size}")

    if max_epochs is not None:
        overrides.append(f"train.max_epochs={max_epochs}")

    if seed is not None:
        overrides.append(f"seed={seed}")

    if wandb_project:
        overrides.append(f"wandb.project={wandb_project}")

    if wandb_entity:
        overrides.append(f"wandb.entity={wandb_entity}")

    if wandb_mode:
        overrides.append(f"wandb.mode={wandb_mode}")

    if disable_profiling and str(disable_profiling).lower() == "true":
        overrides.append("train.profiling.enabled=false")

    if dropout is not None:
        overrides.append(f"model.dropout={dropout}")

    if fusion_hidden is not None:
        overrides.append(f"model.fusion_hidden={fusion_hidden}")

    if radiomics_hidden is not None:
        overrides.append(f"model.radiomics_hidden={radiomics_hidden}")

    if freeze_backbone is not None:
        # Normalize string boolean to lowercase "true"/"false"
        val = str(freeze_backbone).lower()
        if val not in ("true", "false"):
            raise ValueError(f"Invalid boolean value for freeze_backbone: {freeze_backbone}")
        overrides.append(f"model.freeze_backbone={val}")

    if eta_min is not None:
        overrides.append(f"train.scheduler.eta_min={eta_min}")

    # Hydra's initialize(config_path=...) requires a *relative* path.
    # For sweeps (which can be launched from various working directories),
    # we use initialize_config_dir which accepts an absolute path.
    with hydra.initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.3"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    # Create a W&B run early so we can derive an output dir from the run id.
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        tags=list(cfg.wandb.get("tags", [])),
        mode=cfg.wandb.get("mode", "online"),
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        output_base = _resolve_output_base(cfg)
        output_dir = output_base / "sweeps" / run.id
        output_dir.mkdir(parents=True, exist_ok=True)

        configure_logging(str(output_dir))
        logger.info(f"Sweep run output dir: {output_dir}")

        # Attach to existing wandb run instead of creating a new one
        # This prevents duplicate runs in the W&B dashboard
        wandb_logger = WandbLogger(
            experiment=run,
            save_dir=str(output_dir),
        )

        model_path = train_model(cfg, str(output_dir), wandb_logger)
        logger.info(f"Sweep training complete. Model saved to {model_path}")

        run.summary["output_dir"] = str(output_dir)
        run.summary["model_path"] = model_path

    except Exception as e:
        logger.exception(f"Sweep run failed: {e}")
        run.summary["error"] = str(e)
        raise
    finally:
        wandb.finish()


def main() -> None:
    typer.run(sweep_train)


if __name__ == "__main__":
    main()
