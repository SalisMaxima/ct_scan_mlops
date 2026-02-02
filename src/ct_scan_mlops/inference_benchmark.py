"""Benchmark model inference speed and throughput."""

import time
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from ct_scan_mlops.analysis.core import load_model_from_checkpoint
from ct_scan_mlops.data import ChestCTDataModule


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Benchmark model inference."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to benchmark")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info("=" * 60)
    logger.info("Model Inference Benchmark")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = load_model_from_checkpoint(checkpoint_path, cfg, device)
    model = loaded_model.model
    model.eval()

    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    datamodule = ChestCTDataModule(cfg.data)
    datamodule.setup("test")

    if args.dataset == "train":
        dataloader = datamodule.train_dataloader()
    elif args.dataset == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    # Warm up
    logger.info("Warming up...")
    with torch.no_grad():
        batch = next(iter(dataloader))
        x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        _ = model(x)

    # Benchmark
    logger.info(f"\nBenchmarking {args.num_batches} batches...")
    times = []
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_batches:
                break

            x = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
            batch_size = x.shape[0]
            total_samples += batch_size

            # Time inference
            start_time = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            times.append(batch_time)

            if (i + 1) % 5 == 0:
                logger.info(f"  Batch {i + 1}/{args.num_batches}: {batch_time * 1000:.2f}ms")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    throughput = total_samples / sum(times)

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Average batch time: {avg_time * 1000:.2f} ms")
    logger.info(f"Average sample time: {avg_time / args.batch_size * 1000:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} samples/sec")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
