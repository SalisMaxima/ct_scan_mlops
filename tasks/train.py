"""Training and hyperparameter tuning tasks."""

import os
from pathlib import Path

from invoke import Context, task
from loguru import logger

WINDOWS = os.name == "nt"
PROJECT_NAME = "ct_scan_mlops"


@task
def train(ctx: Context, entity: str = "", args: str = "") -> None:
    """Train model with wandb logging.

    Args:
        entity: Wandb entity (optional, defaults to team entity from config)
        args: Hydra config overrides (e.g., "train.max_epochs=100 model=resnet18")

    Examples:
        invoke train.train
        invoke train.train --args "model=resnet18"
        invoke train.train --args "train.max_epochs=100"
        invoke train.train --entity your-personal-username  # Override default team entity
        invoke train.train --args "wandb.mode=disabled"  # Skip wandb
    """
    # Build command with entity override if provided
    # If no entity provided, config.yaml default (team entity) will be used
    entity_override = f"wandb.entity={entity}" if entity else ""
    full_args = f"{entity_override} {args}".strip()

    ctx.run(f"uv run python -m {PROJECT_NAME}.train {full_args}", echo=True, pty=not WINDOWS)


@task
def train_dual(ctx: Context, top_features: bool = False, entity: str = "", args: str = "") -> None:
    """Train dual pathway model (CNN + radiomics features).

    Args:
        top_features: Use only top 16 features (default: False, uses all 50)
        entity: Wandb entity (optional)
        args: Additional Hydra config overrides

    Examples:
        invoke train.train-dual                    # Train with all 50 features
        invoke train.train-dual --top-features     # Train with top 16 features
        invoke train.train-dual --args "train.max_epochs=100"
    """
    model = "dual_pathway_top_features" if top_features else "dual_pathway"

    # Verify config file exists
    config_path = Path(f"configs/model/{model}.yaml")
    if not config_path.exists():
        logger.error(f"ERROR: Model config not found: {config_path}")
        logger.info("Please ensure the file exists or specify a different model configuration.")
        return

    features = "features=top_features" if top_features else ""
    entity_override = f"wandb.entity={entity}" if entity else ""

    full_args = f"model={model} {features} {entity_override} {args}".strip()
    ctx.run(f"uv run python -m {PROJECT_NAME}.train {full_args}", echo=True, pty=not WINDOWS)


@task
def sweep(
    ctx: Context, sweep_config: str = "configs/sweeps/train_sweep.yaml", project: str = "", entity: str = ""
) -> None:
    """Create a W&B sweep from a sweep YAML.

    Args:
        sweep_config: Path to sweep yaml (default: configs/sweeps/train_sweep.yaml)
        project: Optional override for W&B project
        entity: Optional override for W&B entity

    Examples:
        invoke train.sweep
        invoke train.sweep --sweep-config configs/sweeps/dual_pathway.yaml
        invoke train.sweep --entity your-username
    """
    sweep_config_path = Path(sweep_config)
    if not sweep_config_path.exists():
        logger.error(f"ERROR: Sweep config not found: {sweep_config}")
        logger.info(f"Available sweep configs: {list(Path('configs/sweeps').glob('*.yaml'))}")
        return

    cmd = f"uv run wandb sweep {sweep_config}"
    if project:
        cmd += f" --project {project}"
    if entity:
        cmd += f" --entity {entity}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def sweep_agent(ctx: Context, sweep_id: str) -> None:
    """Run a W&B sweep agent.

    Args:
        sweep_id: The full sweep id, e.g. ENTITY/PROJECT/SWEEP_ID

    Example:
        invoke train.sweep-agent --sweep-id mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps/abc123
    """
    ctx.run(f"uv run wandb agent {sweep_id}", echo=True, pty=not WINDOWS)


@task
def sweep_best(ctx: Context, sweep_id: str, metric: str = "val_acc", goal: str = "maximize") -> None:
    """Print the best run (and its config) for a sweep.

    Args:
        sweep_id: The full sweep id, e.g. ENTITY/PROJECT/SWEEP_ID
        metric: Metric name to optimize (default: val_acc)
        goal: maximize | minimize

    Example:
        invoke train.sweep-best --sweep-id ENTITY/PROJECT/SWEEP_ID
    """
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.sweep_best {sweep_id} --metric {metric} --goal {goal}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def sweep_report(
    ctx: Context,
    sweep_id: str,
    output_dir: str = "outputs/reports/sweep_analysis",
    metric: str = "test_acc",
    goal: str = "maximize",
) -> None:
    """Generate a comprehensive analysis report for a W&B sweep.

    Args:
        sweep_id: The full sweep id, e.g. ENTITY/PROJECT/SWEEP_ID
        output_dir: Output directory for the report
        metric: Metric to analyze (default: test_acc)
        goal: maximize | minimize

    Example:
        invoke train.sweep-report --sweep-id ENTITY/PROJECT/SWEEP_ID
    """
    ctx.run(
        f"uv run python -m {PROJECT_NAME}.analysis.sweep_report {sweep_id} --output-dir {output_dir} --metric {metric} --goal {goal}",
        echo=True,
        pty=not WINDOWS,
    )
