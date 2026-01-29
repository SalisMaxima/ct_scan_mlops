#!/usr/bin/env python
"""
Training Time Profiler
======================

Measures actual training time per epoch to provide accurate estimates
for the adenocarcinoma improvement experiment suite.

Usage:
    uv run python scripts/profile_training_time.py
    uv run python scripts/profile_training_time.py --epochs 3 --batch-size 16
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ct_scan_mlops.data import create_dataloaders
from ct_scan_mlops.model import build_model
from ct_scan_mlops.utils import get_device


def profile_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_features: bool = True,
) -> float:
    """Profile training time for one epoch."""
    model.train()
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch
        if len(batch) == 3 and use_features:
            images, features, targets = batch
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(images, features)
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)

        # Forward + backward + optimize
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Progress indicator
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}", end="\r")

    epoch_time = time.time() - start_time
    print()  # Clear progress line
    return epoch_time


def profile_configuration(
    cfg: DictConfig,
    num_epochs: int = 2,
    config_name: str = "default",
) -> dict:
    """Profile a specific model configuration."""
    logger.info(f"Profiling configuration: {config_name}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # Build model
    logger.info("Building model...")
    model = build_model(cfg).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Determine if using features
    use_features = cfg.model.name.lower() in ["dual_pathway", "dualpathway", "hybrid"]

    # Profile epochs
    logger.info(f"Profiling {num_epochs} epochs...")
    epoch_times = []

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_time = profile_one_epoch(model, train_loader, criterion, optimizer, device, use_features)
        epoch_times.append(epoch_time)
        logger.info(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds ({epoch_time / 60:.2f} minutes)")

    # Calculate statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_time = sum(epoch_times)

    return {
        "config_name": config_name,
        "device": str(device),
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else cfg.data.batch_size,
        "total_time_min": total_time / 60,
        "avg_epoch_time_sec": avg_epoch_time,
        "avg_epoch_time_min": avg_epoch_time / 60,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0,
        "batches_per_epoch": len(train_loader),
        "num_workers": train_loader.num_workers if hasattr(train_loader, "num_workers") else 0,
        "epoch_times": epoch_times,
        "model_config": {
            "name": cfg.model.name,
            "radiomics_hidden": cfg.model.get("radiomics_hidden", None),
            "fusion_hidden": cfg.model.get("fusion_hidden", None),
            "dropout": cfg.model.dropout,
        },
    }


def print_results(results: dict):
    """Print profiling results in a formatted manner."""
    print("\n" + "=" * 80)
    print(f"PROFILING RESULTS: {results['config_name']}")
    print("=" * 80)
    print()

    print("System Info:")
    print(f"  Device: {results['device']}")
    print(f"  Model: {results['model_config']['name']}")
    print(f"  Total Parameters: {results['total_params']:,}")
    print(f"  Trainable Parameters: {results['trainable_params']:,}")
    print()

    print("Dataset Info:")
    print(f"  Training Samples: {results['train_samples']}")
    print(f"  Batch Size: {results['batch_size']}")
    print(f"  Batches per Epoch: {results['batches_per_epoch']}")
    print(f"  Num Workers: {results['num_workers']}")
    print()

    print("Timing Results:")
    for i, epoch_time in enumerate(results["epoch_times"], 1):
        print(f"  Epoch {i}: {epoch_time:.2f} sec ({epoch_time / 60:.2f} min)")
    print(f"  Average: {results['avg_epoch_time_sec']:.2f} sec ({results['avg_epoch_time_min']:.2f} min)")
    print()


def print_experiment_estimates(results: dict):
    """Print time estimates for the full experiment suite."""
    avg_min = results["avg_epoch_time_min"]
    cooldown = 3  # minutes
    overhead = 2  # minutes per experiment

    print("=" * 80)
    print("EXPERIMENT SUITE TIME ESTIMATES")
    print("=" * 80)
    print()
    print(f"Average time per epoch: {avg_min:.2f} minutes")
    print(f"Cooldown between experiments: {cooldown} minutes")
    print(f"Setup/teardown overhead: {overhead} minutes per experiment")
    print()

    # Calculate per-tier estimates
    estimates = [
        ("Tier 1: Loss Functions", 7, 25, 1.0),
        ("Tier 2: Weighted Sampling", 4, 25, 1.0),
        ("Tier 3: Data Augmentation", 6, 25, 1.05),  # 5% slower with augmentations
        ("Tier 4: Architecture Mods", 5, 25, 1.20),  # 20% slower with larger model
        ("Tier 5: Extended Training (40 epochs)", 2, 40, 1.0),
        ("Tier 5: Extended Training (50 epochs)", 2, 50, 1.0),
    ]

    total_time = 0
    print("Per-Tier Estimates:")
    print("-" * 80)

    for name, num_exp, epochs, slowdown in estimates:
        time_per_exp = (epochs * avg_min * slowdown) + cooldown + overhead
        tier_time = num_exp * time_per_exp
        total_time += tier_time

        print(f"{name:45s} {num_exp:2d} exps × {epochs:2d} epochs = {tier_time / 60:5.1f} hours")

    print("-" * 80)
    print(f"{'TOTAL':45s} {'26':>2s} exps            = {total_time / 60:5.1f} hours")
    print()

    # Additional scenarios
    print("Quick Scenarios:")
    print(f"  Single 25-epoch experiment: {(25 * avg_min + cooldown + overhead) / 60:.1f} hours")
    print(f"  Single 50-epoch experiment: {(50 * avg_min + cooldown + overhead) / 60:.1f} hours")
    print(f"  Tier 1 only (7 experiments): {(7 * (25 * avg_min + cooldown + overhead)) / 60:.1f} hours")
    print(f"  Tiers 1-3 (17 experiments): {((7 + 4 + 6) * (25 * avg_min * 1.02 + cooldown + overhead)) / 60:.1f} hours")
    print()

    return total_time


def check_wandb_history():
    """Check W&B for historical timing data."""
    try:
        import wandb

        print("=" * 80)
        print("HISTORICAL W&B TIMING DATA")
        print("=" * 80)
        print()

        api = wandb.Api()
        runs = api.runs("mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps")

        # Find recent dual_pathway runs
        dual_pathway_runs = []
        for run in runs[:50]:
            if run.state == "finished" and "dual_pathway" in run.name.lower():
                runtime = run.summary.get("_runtime", 0)
                epochs = run.config.get("train", {}).get("max_epochs", 0)
                if runtime > 0 and epochs > 0:
                    dual_pathway_runs.append(
                        {
                            "name": run.name,
                            "runtime_min": runtime / 60,
                            "epochs": epochs,
                            "min_per_epoch": (runtime / 60) / epochs,
                            "created_at": run.created_at,
                        }
                    )

        if dual_pathway_runs:
            # Sort by creation date
            dual_pathway_runs.sort(key=lambda x: x["created_at"], reverse=True)

            print(f"Found {len(dual_pathway_runs)} dual_pathway runs in W&B")
            print()
            print(f"{'Run Name':50s} {'Epochs':>7s} {'Total':>8s} {'Min/Ep':>8s}")
            print("-" * 80)

            for run in dual_pathway_runs[:10]:
                print(
                    f"{run['name']:50s} {run['epochs']:>7d} {run['runtime_min']:>7.1f}m {run['min_per_epoch']:>7.2f}m"
                )

            print()
            avg_time_per_epoch = sum(r["min_per_epoch"] for r in dual_pathway_runs[:10]) / min(
                10, len(dual_pathway_runs)
            )
            print(f"Average time per epoch (last 10 runs): {avg_time_per_epoch:.2f} minutes")
            print()
        else:
            print("No dual_pathway runs found in W&B history.")
            print()

    except ImportError:
        print("wandb not available. Skipping historical data check.")
        print()
    except Exception as e:
        print(f"Error fetching W&B data: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Profile training time per epoch")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to profile (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: use config value)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dual_pathway_top_features",
        help="Model config to profile (default: dual_pathway_top_features)",
    )
    parser.add_argument(
        "--check-wandb",
        action="store_true",
        help="Check W&B for historical timing data",
    )
    parser.add_argument(
        "--profile-variants",
        action="store_true",
        help="Profile multiple model variants (baseline, large, full features)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("TRAINING TIME PROFILER")
    print("=" * 80)
    print()

    # Check W&B history first if requested
    if args.check_wandb:
        check_wandb_history()

    # Load base configuration
    project_dir = Path(__file__).parent.parent
    train_cfg_path = project_dir / "configs" / "train" / "default.yaml"
    data_cfg_path = project_dir / "configs" / "data" / "chest_ct.yaml"

    base_cfg = OmegaConf.create(
        {
            "train": OmegaConf.load(train_cfg_path),
            "data": OmegaConf.load(data_cfg_path),
        }
    )

    # Override batch size if specified
    if args.batch_size is not None:
        base_cfg.data.batch_size = args.batch_size

    if args.profile_variants:
        # Profile multiple configurations
        configs_to_profile = [
            ("Baseline (radiomics_hidden=512)", "dual_pathway_top_features", {}),
            (
                "Large (radiomics_hidden=768)",
                "dual_pathway_top_features",
                {"radiomics_hidden": 768, "fusion_hidden": 384},
            ),
            (
                "Extra Large (radiomics_hidden=1024)",
                "dual_pathway_top_features",
                {"radiomics_hidden": 1024, "fusion_hidden": 512},
            ),
        ]

        all_results = []
        for name, config_name, overrides in configs_to_profile:
            # Load model config
            model_cfg_path = project_dir / "configs" / "model" / f"{config_name}.yaml"
            cfg = OmegaConf.merge(base_cfg, {"model": OmegaConf.load(model_cfg_path)})

            # Apply overrides
            for key, value in overrides.items():
                OmegaConf.update(cfg.model, key, value)

            # Profile
            results = profile_configuration(cfg, num_epochs=args.epochs, config_name=name)
            all_results.append(results)
            print_results(results)

        # Print comparison
        print("\n" + "=" * 80)
        print("CONFIGURATION COMPARISON")
        print("=" * 80)
        print()
        print(f"{'Configuration':40s} {'Params':>12s} {'Time/Epoch':>12s} {'25 Epochs':>12s}")
        print("-" * 80)

        for r in all_results:
            time_25_epochs = 25 * r["avg_epoch_time_min"]
            print(
                f"{r['config_name']:40s} "
                f"{r['total_params']:>12,d} "
                f"{r['avg_epoch_time_min']:>11.2f}m "
                f"{time_25_epochs:>11.1f}m"
            )
        print()

        # Use baseline for estimates
        total_time = print_experiment_estimates(all_results[0])

    else:
        # Profile single configuration
        model_cfg_path = project_dir / "configs" / "model" / f"{args.config}.yaml"
        cfg = OmegaConf.merge(base_cfg, {"model": OmegaConf.load(model_cfg_path)})

        results = profile_configuration(cfg, num_epochs=args.epochs, config_name=args.config)
        print_results(results)
        total_time = print_experiment_estimates(results)

    # Save results to file
    output_file = project_dir / "logs" / "profiling_results.txt"
    output_file.parent.mkdir(exist_ok=True)

    with output_file.open("w") as f:
        f.write("Training Time Profiling Results\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Average time per epoch: {results['avg_epoch_time_min']:.2f} minutes\n")
        f.write(f"Estimated total experiment suite time: {total_time / 60:.1f} hours\n")

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
