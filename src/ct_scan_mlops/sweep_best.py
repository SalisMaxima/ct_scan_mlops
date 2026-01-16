"""Utilities for working with W&B sweeps.

Provides a small CLI to identify the best run in a sweep and print the corresponding
hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import wandb
from omegaconf import OmegaConf


def _get_summary_metric(run: Any, metric: str) -> float | None:
    # W&B summary is a dict-like object.
    summary = getattr(run, "summary", None)
    if not summary:
        return None

    value = summary.get(metric)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_default_entity_project() -> tuple[str | None, str | None]:
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = project_root / "configs" / "config.yaml"
    if not cfg_path.exists():
        return None, None

    cfg = OmegaConf.load(cfg_path)
    wandb_cfg = cfg.get("wandb", {})
    entity = wandb_cfg.get("entity")
    project = wandb_cfg.get("project")
    return entity, project


def _normalize_sweep_id(sweep_id: str, entity: str | None, project: str | None) -> str:
    # Accept either:
    # - ENTITY/PROJECT/SWEEP_ID
    # - SWEEP_ID (then require entity+project, defaulting from configs/config.yaml)
    if sweep_id.count("/") >= 2:
        return sweep_id

    default_entity, default_project = _get_default_entity_project()
    entity = entity or default_entity
    project = project or default_project

    if not entity or not project:
        raise typer.BadParameter(
            "Provide a full sweep id ENTITY/PROJECT/SWEEP_ID, or pass --entity and --project (or set them in configs/config.yaml)."
        )

    # Treat whatever was passed as the sweep id component.
    return f"{entity}/{project}/{sweep_id}"


def best(
    sweep_id: str = typer.Argument(..., help="Sweep id: ENTITY/PROJECT/SWEEP_ID"),
    entity: str = typer.Option(None, help="W&B entity (used if sweep_id is not a full path)"),
    project: str = typer.Option(None, help="W&B project (used if sweep_id is not a full path)"),
    metric: str = typer.Option("val_acc", help="Metric to optimize (default: val_acc)"),
    goal: str = typer.Option("maximize", help="maximize | minimize"),
    include_config: bool = typer.Option(True, help="Print run config (hyperparameters)"),
) -> None:
    """Print the best run for a given sweep."""

    normalized = _normalize_sweep_id(sweep_id, entity=entity, project=project)

    api = wandb.Api()
    try:
        sweep = api.sweep(normalized)
    except Exception as exc:  # wandb throws CommError/ValueError depending on failure mode
        raise typer.BadParameter(
            f"Could not find sweep '{normalized}'. Double-check entity/project/sweep id (case-sensitive) in the W&B UI."
        ) from exc

    best_run = None
    best_value: float | None = None

    for run in sweep.runs:
        # Only consider completed runs.
        state = getattr(run, "state", "")
        if state not in {"finished", "crashed", "failed"}:
            continue
        if state != "finished":
            continue

        value = _get_summary_metric(run, metric)
        if value is None:
            continue

        if best_value is None:
            best_run, best_value = run, value
            continue

        if goal == "minimize":
            if value < best_value:
                best_run, best_value = run, value
        else:
            if value > best_value:
                best_run, best_value = run, value

    if best_run is None or best_value is None:
        raise typer.BadParameter(
            "No finished runs with the requested metric were found in this sweep yet. "
            "Wait for at least one run to finish successfully, then rerun sweep-best."
        )

    result: dict[str, Any] = {
        "sweep_id": normalized,
        "metric": metric,
        "goal": goal,
        "best": {
            "run_id": best_run.id,
            "run_name": best_run.name,
            "url": best_run.url,
            "value": best_value,
        },
    }

    if include_config:
        # W&B configs contain internal keys; keep only user-facing sweep params where possible.
        cfg = dict(getattr(best_run, "config", {}) or {})
        cleaned = {k: v for k, v in cfg.items() if not k.startswith("_")}
        result["best"]["config"] = cleaned

    print(json.dumps(result, indent=2))


def main() -> None:
    typer.run(best)


if __name__ == "__main__":
    main()
