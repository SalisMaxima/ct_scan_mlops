from __future__ import annotations

from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import torch
import typer
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from ct_scan_mlops.data import CLASSES, ChestCTDataset, _find_data_root


def _label_distribution(labels: list[int]) -> torch.Tensor:
    """Compute label distribution as counts tensor."""
    if not labels:
        return torch.zeros(len(CLASSES), dtype=torch.long)
    return torch.bincount(torch.tensor(labels), minlength=len(CLASSES))


def _collect_class_samples(dataset: ChestCTDataset, per_class: int = 1) -> list[tuple[Path, int]]:
    """Collect up to per_class samples per class from dataset samples."""
    collected: dict[int, list[Path]] = {i: [] for i in range(len(CLASSES))}
    for img_path, label in dataset.samples:
        if len(collected[label]) < per_class:
            collected[label].append(img_path)
        if all(len(paths) >= per_class for paths in collected.values()):
            break

    samples: list[tuple[Path, int]] = []
    for label, paths in collected.items():
        if not paths:
            print(f"Warning: No samples found for class '{CLASSES[label]}'")
            continue
        for img_path in paths:
            samples.append((img_path, label))
    return samples


def _build_pil_grid(samples: list[tuple[Path, int]], image_size: int = 224, ncols: int = 4) -> Image.Image:
    """Build a PIL image grid from samples."""
    if not samples:
        raise ValueError("No samples available to build image grid")

    nrows = (len(samples) + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * image_size, nrows * image_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except OSError as exc:
        # Best-effort font loading: continue without custom font if it fails
        print(f"Warning: Failed to load default font: {exc!r}")
        font = None

    for idx, (img_path, label) in enumerate(samples):
        row = idx // ncols
        col = idx % ncols
        with Image.open(img_path) as img_src:
            img = img_src.convert("RGB").resize((image_size, image_size))
        grid.paste(img, (col * image_size, row * image_size))
        label_text = CLASSES[label]
        draw.text((col * image_size + 5, row * image_size + 5), label_text, fill=(255, 255, 255), font=font)

    return grid


def dataset_statistics(datadir: str = "data/raw", show_images: bool = False, save_images: bool = False) -> None:
    """Compute dataset statistics for the Chest CT dataset."""
    data_root = _find_data_root(Path(datadir))

    transform = A.Compose([A.Resize(224, 224), ToTensorV2()])

    splits: dict[str, ChestCTDataset | None] = {"train": None, "valid": None, "test": None}
    for split in list(splits.keys()):
        split_dir = data_root / split
        if split_dir.exists():
            splits[split] = ChestCTDataset(data_root, split=split, transform=transform)
        else:
            print(f"Split '{split}' not found at {split_dir}, skipping")

    for split, dataset in splits.items():
        if dataset is None:
            continue
        print(f"{split.title()} dataset")
        print(f"Number of images: {len(dataset)}")
        sample_image, _ = dataset[0]
        print(f"Image shape: {sample_image.shape}\n")

    output_dir = Path("reports/figures")
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = splits.get("train")
    if train_dataset is not None:
        train_labels = [label for _, label in train_dataset.samples]
        train_counts = _label_distribution(train_labels)

        class_samples = _collect_class_samples(train_dataset, per_class=1)
        grid = _build_pil_grid(class_samples, image_size=224, ncols=4)
        if save_images:
            grid.save(output_dir / "ct_scan_images.png")
        if show_images:
            grid.show()

        plt.figure(figsize=(8, 4))
        plt.bar(torch.arange(len(CLASSES)), train_counts)
        plt.xticks(torch.arange(len(CLASSES)), CLASSES, rotation=20, ha="right")
        plt.title("Train label distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        if save_images:
            plt.savefig(output_dir / "train_label_distribution.png")
        if show_images:
            plt.show()
        else:
            plt.close()

    for split in ["valid", "test"]:
        dataset = splits.get(split)
        if dataset is None:
            continue
        labels = [label for _, label in dataset.samples]
        counts = _label_distribution(labels)
        plt.figure(figsize=(8, 4))
        plt.bar(torch.arange(len(CLASSES)), counts)
        plt.xticks(torch.arange(len(CLASSES)), CLASSES, rotation=20, ha="right")
        plt.title(f"{split.title()} label distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        if save_images:
            plt.savefig(output_dir / f"{split}_label_distribution.png")
        if show_images:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
