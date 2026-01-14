from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Canonical classes (index order)
CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

# Folder names in train/valid are longer, but start with these prefixes.
PREFIX_TO_CLASS = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "normal": "normal",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
}

IMG_EXTS = {".png", ".jpg"}


def _infer_label_from_folder(folder_name: str, class_to_idx: dict[str, int]) -> int:
    name = folder_name.lower()
    for prefix, cls in PREFIX_TO_CLASS.items():
        if name.startswith(prefix):
            return class_to_idx[cls]
    raise ValueError(f"Could not infer class from folder name: {folder_name}")


def _find_data_root(raw_dir: Path) -> Path:
    """
    Your structure in the screenshot is:
      data/raw/chest-ctscan-images/Data/{train,valid,test}/...
    This function finds ".../Data".
    """
    raw_dir = raw_dir.expanduser().resolve()
    # common expected location
    candidate = raw_dir / "chest-ctscan-images" / "Data"
    if candidate.exists():
        return candidate

    # fallback: search for a folder literally named "Data"
    for p in raw_dir.rglob("Data"):
        if p.is_dir() and (p / "train").exists() and (p / "test").exists():
            return p

    raise FileNotFoundError(
        f"Could not find dataset 'Data' folder under {raw_dir}. "
        f"Expected something like data/raw/chest-ctscan-images/Data/..."
    )


class ChestCTDataset(Dataset):
    def __init__(self, raw_dir: str | Path = "data/raw", split: str = "train", image_size: int = 224):
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: 'train', 'valid', 'test'")

        self.data_root = _find_data_root(Path(raw_dir))
        self.split_dir = self.data_root / split

        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # minimal transforms (no augmentation here; keep it simple)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # float32, [0,1], shape (C,H,W)
            ]
        )

        self.samples: list[tuple[Path, int]] = []
        for class_folder in sorted([p for p in self.split_dir.iterdir() if p.is_dir()]):
            label = _infer_label_from_folder(class_folder.name, self.class_to_idx)
            for img_path in class_folder.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def chest_ct(raw_dir: str | Path = "data/raw", image_size: int = 224):
    """Convenience helper: returns (train_ds, val_ds, test_ds)."""
    train_ds = ChestCTDataset(raw_dir=raw_dir, split="train", image_size=image_size)
    val_ds = ChestCTDataset(raw_dir=raw_dir, split="valid", image_size=image_size)
    test_ds = ChestCTDataset(raw_dir=raw_dir, split="test", image_size=image_size)
    return train_ds, val_ds, test_ds
