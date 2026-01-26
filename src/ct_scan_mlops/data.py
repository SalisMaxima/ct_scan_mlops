from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
import typer
from albumentations.pytorch import ToTensorV2
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# Canonical class labels
CLASSES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma",
    "normal",
]

# Folder name prefixes that map to class labels
PREFIX_TO_CLASS = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large_cell_carcinoma",
    "large_cell_carcinoma": "large_cell_carcinoma",
    "squamous.cell.carcinoma": "squamous_cell_carcinoma",
    "squamous_cell_carcinoma": "squamous_cell_carcinoma",
    "normal": "normal",
}

IMG_EXTS = {".png", ".jpg", ".jpeg"}

# ImageNet normalization stats (default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to mean=0, std=1 per channel."""
    # images shape: [N, C, H, W]
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), keepdim=True)
    return (images - mean) / (std + 1e-8)


def _infer_label_from_folder(folder_name: str, class_to_idx: dict[str, int]) -> int:
    """Infer class label from folder name using prefix matching."""
    name = folder_name.lower()
    for prefix, cls in PREFIX_TO_CLASS.items():
        if name.startswith(prefix):
            return class_to_idx[cls]
    raise ValueError(f"Could not infer class from folder name: {folder_name}")


def _find_data_root(raw_dir: Path) -> Path:
    """Find the Data directory containing train/valid/test splits."""
    raw_dir = raw_dir.expanduser().resolve()

    # Common expected location
    candidate = raw_dir / "chest-ctscan-images" / "Data"
    if candidate.exists():
        return candidate

    # Try direct path
    if (raw_dir / "train").exists() and (raw_dir / "test").exists():
        return raw_dir

    # Fallback: search for a folder named "Data"
    for p in raw_dir.rglob("Data"):
        if p.is_dir() and (p / "train").exists():
            return p

    raise FileNotFoundError(
        f"Could not find dataset 'Data' folder under {raw_dir}. "
        f"Expected structure: data/raw/chest-ctscan-images/Data/..."
    )


def get_transforms(
    split: str,
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    augmentation_cfg: DictConfig | None = None,
) -> A.Compose:
    """Get Albumentations transforms for a given split.

    Args:
        split: One of 'train', 'valid', 'test'
        image_size: Target image size
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
        augmentation_cfg: Hydra config for augmentation settings

    Returns:
        Albumentations Compose pipeline
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    # Base transforms (always applied)
    base_transforms = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    if split == "train" and augmentation_cfg is not None:
        train_cfg = augmentation_cfg.get("train", {})
        aug_transforms = []

        if train_cfg.get("horizontal_flip", False):
            aug_transforms.append(A.HorizontalFlip(p=0.5))

        if train_cfg.get("vertical_flip", False):
            aug_transforms.append(A.VerticalFlip(p=0.5))

        rotation_limit = train_cfg.get("rotation_limit", 0)
        if rotation_limit > 0:
            aug_transforms.append(A.Rotate(limit=rotation_limit, p=0.5))

        brightness = train_cfg.get("brightness_limit", 0)
        contrast = train_cfg.get("contrast_limit", 0)
        if brightness > 0 or contrast > 0:
            aug_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.5,
                )
            )

        # Enhanced augmentations for shape confusion mitigation
        if train_cfg.get("elastic_transform", False):
            aug_transforms.append(
                A.ElasticTransform(
                    alpha=train_cfg.get("elastic_alpha", 120),
                    sigma=train_cfg.get("elastic_sigma", 6),
                    p=train_cfg.get("elastic_p", 0.3),
                )
            )

        if train_cfg.get("grid_distortion", False):
            aug_transforms.append(
                A.GridDistortion(
                    num_steps=train_cfg.get("grid_steps", 5),
                    distort_limit=train_cfg.get("grid_distort_limit", 0.3),
                    p=train_cfg.get("grid_p", 0.3),
                )
            )

        if train_cfg.get("coarse_dropout", False):
            aug_transforms.append(
                A.CoarseDropout(
                    max_holes=train_cfg.get("dropout_max_holes", 8),
                    max_height=train_cfg.get("dropout_max_height", 32),
                    max_width=train_cfg.get("dropout_max_width", 32),
                    min_holes=train_cfg.get("dropout_min_holes", 1),
                    min_height=train_cfg.get("dropout_min_height", 8),
                    min_width=train_cfg.get("dropout_min_width", 8),
                    fill_value=0,
                    p=train_cfg.get("dropout_p", 0.3),
                )
            )

        if train_cfg.get("gaussian_noise", False):
            aug_transforms.append(
                A.GaussNoise(
                    var_limit=train_cfg.get("noise_var_limit", (10.0, 50.0)),
                    p=train_cfg.get("noise_p", 0.3),
                )
            )

        if train_cfg.get("blur", False):
            aug_transforms.append(
                A.GaussianBlur(
                    blur_limit=train_cfg.get("blur_limit", (3, 7)),
                    p=train_cfg.get("blur_p", 0.2),
                )
            )

        # Augmentations first, then resize + normalize
        return A.Compose(aug_transforms + base_transforms)

    # Validation/test: only resize and normalize
    return A.Compose(base_transforms)


class ProcessedChestCTDataset(Dataset):
    """Dataset for preprocessed Chest CT scan images (loads from .pt files).

    This is the fast dataset class that loads pre-processed tensors from disk.
    Use preprocess() to generate the .pt files first.
    """

    def __init__(self, data_path: Path, split: str = "train") -> None:
        """Initialize dataset from processed tensors.

        Args:
            data_path: Path to processed data folder (containing .pt files)
            split: One of 'train', 'valid', 'test'
        """
        self.data_path = Path(data_path)
        self.split = split

        # Load pre-processed tensors
        images_path = self.data_path / f"{split}_images.pt"
        labels_path = self.data_path / f"{split}_labels.pt"

        if not images_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {images_path}. "
                f"Run 'invoke preprocess-data' or 'python -m ct_scan_mlops.data' first."
            )

        self.images = torch.load(images_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)

        # Class mapping
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(CLASSES)

        logger.info(f"Loaded {len(self.labels)} processed samples for split '{split}'")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class RadiomicsChestCTDataset(Dataset):
    """Dataset for preprocessed CT images with radiomics features.

    Returns (image, features, label) tuples for training the dual-pathway model.
    Use extract_features() to generate the feature files first.
    """

    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        normalize_features: bool = True,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ) -> None:
        """Initialize dataset from processed tensors and features.

        Args:
            data_path: Path to processed data folder (containing .pt files)
            split: One of 'train', 'valid', 'test'
            normalize_features: Whether to normalize features
            feature_mean: Mean for feature normalization (computed from train if None)
            feature_std: Std for feature normalization (computed from train if None)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.normalize_features = normalize_features

        # Load pre-processed tensors
        images_path = self.data_path / f"{split}_images.pt"
        labels_path = self.data_path / f"{split}_labels.pt"
        features_path = self.data_path / f"{split}_features.pt"

        if not images_path.exists():
            raise FileNotFoundError(f"Processed data not found at {images_path}. Run 'invoke preprocess-data' first.")

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found at {features_path}. Run 'invoke extract-features' first.")

        self.images = torch.load(images_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)
        self.features = torch.load(features_path, weights_only=True)

        # Store normalization stats
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Compute normalization stats from training data if not provided
        if normalize_features and split == "train" and feature_mean is None:
            self.feature_mean = self.features.mean(dim=0)
            self.feature_std = self.features.std(dim=0)
            # Avoid division by zero
            self.feature_std = torch.where(
                self.feature_std > 1e-8,
                self.feature_std,
                torch.ones_like(self.feature_std),
            )

        # Class mapping
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(CLASSES)

        logger.info(f"Loaded {len(self.labels)} samples with {self.features.shape[1]} features for split '{split}'")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (image, features, label) tuple."""
        image = self.images[idx]
        features = self.features[idx]
        label = self.labels[idx]

        # Normalize features
        if self.normalize_features and self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        return image, features, label


class ChestCTDataset(Dataset):
    """Dataset for Chest CT scan images (loads from raw images).

    Loads images from disk and applies transforms on-the-fly.
    Use ProcessedChestCTDataset for faster loading with preprocessed tensors.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform: A.Compose | None = None,
        image_size: int = 224,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Path to data root (containing train/valid/test folders)
            split: One of 'train', 'valid', 'test'
            transform: Albumentations transform pipeline
            image_size: Target image size (used if transform is None)
        """
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: 'train', 'valid', 'test'")

        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Class mapping
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(CLASSES)

        # Transform pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(split, image_size=image_size)

        # Collect samples
        self.samples: list[tuple[Path, int]] = []
        for class_folder in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            try:
                label = _infer_label_from_folder(class_folder.name, self.class_to_idx)
            except ValueError as e:
                logger.warning(f"Skipping folder: {e}")
                continue

            for img_path in class_folder.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {self.split_dir}")

        logger.info(f"Loaded {len(self.samples)} images for split '{split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # Load image as RGB numpy array
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Apply transforms
        transformed = self.transform(image=img_array)
        x = transformed["image"]

        y = torch.tensor(label, dtype=torch.long)
        return x, y


def download_data(target_dir: str | Path = "data/raw") -> Path:
    """Download CT scan dataset from Kaggle.

    Args:
        target_dir: Directory to save the downloaded data

    Returns:
        Path to the downloaded data directory

    Raises:
        ImportError: If kagglehub is not installed
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required for downloading data. Install it with: pip install kagglehub or uv add kagglehub"
        ) from e

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")

    # Copy to target directory
    final_path = target_path / "chest-ctscan-images"
    if final_path.exists():
        print(f"Removing existing data at {final_path}...")
        shutil.rmtree(final_path)

    print(f"Copying data to {final_path}...")
    shutil.copytree(cache_path, final_path)

    print(f"✓ Dataset downloaded successfully to: {final_path}")
    return final_path


def preprocess(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    image_size: int = 224,
) -> dict[str, Any]:
    """Preprocess raw images and save as tensors.

    Loads all images, resizes them, normalizes to mean=0/std=1,
    and saves as PyTorch tensors for fast loading during training.

    Args:
        raw_dir: Path to raw data directory
        output_dir: Path to save processed data
        image_size: Target image size

    Returns:
        Dictionary with preprocessing statistics
    """
    print(f"Preprocessing data from {raw_dir}...")

    data_root = _find_data_root(Path(raw_dir))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    stats: dict[str, Any] = {"image_size": image_size, "classes": CLASSES}

    # Simple transform for preprocessing (resize only, no augmentation)
    preprocess_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ]
    )

    for split in ["train", "valid", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  Warning: Split '{split}' not found, skipping")
            continue

        images_list: list[torch.Tensor] = []
        labels_list: list[int] = []

        # Collect all image paths
        all_samples: list[tuple[Path, int]] = []
        for class_folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            try:
                label = _infer_label_from_folder(class_folder.name, class_to_idx)
            except ValueError:
                continue

            for img_path in class_folder.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                    all_samples.append((img_path, label))

        print(f"  Processing {len(all_samples)} images for split '{split}'...")

        for img_path, label in tqdm(all_samples, desc=f"  {split}"):
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            transformed = preprocess_transform(image=img_array)
            images_list.append(transformed["image"])
            labels_list.append(label)

        # Stack into tensors
        images = torch.stack(images_list).float() / 255.0  # Scale to [0, 1]
        labels = torch.tensor(labels_list, dtype=torch.long)

        # Normalize to mean=0, std=1 (computed per-split for train, applied same way)
        if split == "train":
            # Compute and save normalization stats from training data
            mean = images.mean(dim=(0, 2, 3)).tolist()
            std = images.std(dim=(0, 2, 3)).tolist()
            stats["mean"] = mean
            stats["std"] = std
            print(
                f"  Computed normalization stats - mean: {[f'{m:.4f}' for m in mean]}, std: {[f'{s:.4f}' for s in std]}"
            )

        # Apply normalization
        images = normalize(images)

        # Save tensors
        torch.save(images, output_path / f"{split}_images.pt")
        torch.save(labels, output_path / f"{split}_labels.pt")

        stats[f"{split}_count"] = len(labels)
        stats[f"{split}_shape"] = list(images.shape)
        print(f"  Saved {split}: {images.shape}")

    # Save stats
    torch.save(stats, output_path / "stats.pt")

    print("\nPreprocessing complete!")
    print(f"  Output directory: {output_path}")
    print(f"  Train samples: {stats.get('train_count', 0)}")
    print(f"  Valid samples: {stats.get('valid_count', 0)}")
    print(f"  Test samples: {stats.get('test_count', 0)}")

    return stats


def create_dataloaders(
    cfg: DictConfig,
    use_processed: bool = True,
    use_features: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Create train, validation, and test dataloaders from config.

    Args:
        cfg: Hydra config containing data and paths settings
        use_processed: If True, use processed tensors (faster). If False, load raw images.
        use_features: If True, use RadiomicsChestCTDataset (returns image, features, label).

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_cfg = cfg.data
    processed_path = Path(data_cfg.get("processed_path", "data/processed"))

    # Check if using radiomics features
    if use_features and (processed_path / "train_features.pt").exists():
        logger.info(f"Using radiomics features from {processed_path}")
        # Create training dataset first to compute normalization stats
        train_ds = RadiomicsChestCTDataset(processed_path, split="train", normalize_features=True)
        # Use training stats for validation and test
        val_ds = RadiomicsChestCTDataset(
            processed_path,
            split="valid",
            normalize_features=True,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
        )
        test_ds = RadiomicsChestCTDataset(
            processed_path,
            split="test",
            normalize_features=True,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
        )
    # Check if processed data exists
    elif use_processed and (processed_path / "train_images.pt").exists():
        logger.info(f"Using preprocessed data from {processed_path}")
        train_ds = ProcessedChestCTDataset(processed_path, split="train")
        val_ds = ProcessedChestCTDataset(processed_path, split="valid")
        test_ds = ProcessedChestCTDataset(processed_path, split="test")
    else:
        if use_processed:
            logger.warning("Processed data not found, falling back to raw images")
        logger.info("Loading raw images (slower)")

        data_dir = _find_data_root(Path(cfg.paths.data_dir))

        # Get normalization stats from config
        mean = list(data_cfg.normalize.mean)
        std = list(data_cfg.normalize.std)
        image_size = data_cfg.image_size

        # Create transforms for each split
        train_transform = get_transforms(
            "train",
            image_size=image_size,
            mean=mean,
            std=std,
            augmentation_cfg=data_cfg.augmentation,
        )
        eval_transform = get_transforms(
            "valid",
            image_size=image_size,
            mean=mean,
            std=std,
        )

        train_ds = ChestCTDataset(data_dir, split="train", transform=train_transform)
        val_ds = ChestCTDataset(data_dir, split="valid", transform=eval_transform)
        test_ds = ChestCTDataset(data_dir, split="test", transform=eval_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", False) and data_cfg.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", False) and data_cfg.num_workers > 0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", False) and data_cfg.num_workers > 0,
    )

    return train_loader, val_loader, test_loader


class ChestCTDataModule(pl.LightningDataModule):
    """Lightning DataModule for Chest CT scan dataset.

    Provides a reusable, self-contained data handling structure following
    PyTorch Lightning best practices.

    Args:
        cfg: Hydra configuration containing data and paths settings
        use_processed: If True, use processed tensors (faster). If False, load raw images.
        use_features: If True, use RadiomicsChestCTDataset (returns image, features, label).
    """

    def __init__(
        self,
        cfg: DictConfig,
        use_processed: bool = True,
        use_features: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_processed = use_processed
        self.use_features = use_features
        self.data_cfg = cfg.data

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

        # Store feature normalization stats (computed from training data)
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load datasets for the specified stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'
        """
        processed_path = Path(self.data_cfg.get("processed_path", "data/processed"))

        # Check if using radiomics features
        if self.use_features and (processed_path / "train_features.pt").exists():
            logger.info(f"Using radiomics features from {processed_path}")
            if stage == "fit" or stage is None:
                # Create training dataset first to compute normalization stats
                self.train_ds = RadiomicsChestCTDataset(processed_path, split="train", normalize_features=True)
                # Get normalization stats from training data
                self.feature_mean = self.train_ds.feature_mean
                self.feature_std = self.train_ds.feature_std

                # Create validation dataset with training stats
                self.val_ds = RadiomicsChestCTDataset(
                    processed_path,
                    split="valid",
                    normalize_features=True,
                    feature_mean=self.feature_mean,
                    feature_std=self.feature_std,
                )
            if stage == "test" or stage is None:
                # Use training stats if available, otherwise load train to compute
                if self.feature_mean is None:
                    train_ds = RadiomicsChestCTDataset(processed_path, split="train", normalize_features=True)
                    self.feature_mean = train_ds.feature_mean
                    self.feature_std = train_ds.feature_std

                self.test_ds = RadiomicsChestCTDataset(
                    processed_path,
                    split="test",
                    normalize_features=True,
                    feature_mean=self.feature_mean,
                    feature_std=self.feature_std,
                )
            return

        # Check if processed data exists
        if self.use_processed and (processed_path / "train_images.pt").exists():
            logger.info(f"Using preprocessed data from {processed_path}")
            if stage == "fit" or stage is None:
                self.train_ds = ProcessedChestCTDataset(processed_path, split="train")
                self.val_ds = ProcessedChestCTDataset(processed_path, split="valid")
            if stage == "test" or stage is None:
                self.test_ds = ProcessedChestCTDataset(processed_path, split="test")
        else:
            if self.use_processed:
                logger.warning("Processed data not found, falling back to raw images")
            logger.info("Loading raw images (slower)")

            raw_cfg_path = Path(self.cfg.paths.data_dir)
            if not raw_cfg_path.exists():
                raise FileNotFoundError(
                    "Raw dataset path does not exist: "
                    f"{raw_cfg_path}.\n"
                    "Fix by pulling/downloading data and (optionally) preprocessing:\n"
                    "  - invoke dvc-pull\n"
                    "  - invoke preprocess-data\n"
                    "Or update configs/config.yaml -> paths.data_dir to point at your dataset root."
                )

            data_dir = _find_data_root(Path(self.cfg.paths.data_dir))

            # Get normalization stats from config
            mean = list(self.data_cfg.normalize.mean)
            std = list(self.data_cfg.normalize.std)
            image_size = self.data_cfg.image_size

            # Create transforms for each split
            train_transform = get_transforms(
                "train",
                image_size=image_size,
                mean=mean,
                std=std,
                augmentation_cfg=self.data_cfg.augmentation,
            )
            eval_transform = get_transforms(
                "valid",
                image_size=image_size,
                mean=mean,
                std=std,
            )

            if stage == "fit" or stage is None:
                self.train_ds = ChestCTDataset(data_dir, split="train", transform=train_transform)
                self.val_ds = ChestCTDataset(data_dir, split="valid", transform=eval_transform)
            if stage == "test" or stage is None:
                self.test_ds = ChestCTDataset(data_dir, split="test", transform=eval_transform)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with optional weighted sampling."""
        # Check if weighted sampling is enabled
        sampling_cfg = self.data_cfg.get("sampling", {})
        use_weighted = sampling_cfg.get("weighted", False)

        if use_weighted:
            # Get class weights from config
            class_weights = sampling_cfg.get("class_weights", [1.0, 1.0, 1.0, 1.0])
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

            # Get labels from dataset
            if hasattr(self.train_ds, "labels"):
                labels = self.train_ds.labels
            elif hasattr(self.train_ds, "samples"):
                labels = torch.tensor([s[1] for s in self.train_ds.samples])
            else:
                logger.warning("Cannot determine labels for weighted sampling, falling back to shuffle")
                use_weighted = False

        if use_weighted:
            # Compute sample weights based on class
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            logger.info(f"Using weighted sampling with class_weights: {class_weights.tolist()}")

            return DataLoader(
                self.train_ds,
                batch_size=self.data_cfg.batch_size,
                sampler=sampler,  # Cannot use shuffle with sampler
                num_workers=self.data_cfg.num_workers,
                pin_memory=self.data_cfg.get("pin_memory", True),
                persistent_workers=self.data_cfg.get("persistent_workers", False) and self.data_cfg.num_workers > 0,
            )

        # Default: shuffle without weighted sampling
        return DataLoader(
            self.train_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.get("pin_memory", True),
            persistent_workers=self.data_cfg.get("persistent_workers", False) and self.data_cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.get("pin_memory", True),
            persistent_workers=self.data_cfg.get("persistent_workers", False) and self.data_cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.get("pin_memory", True),
            persistent_workers=self.data_cfg.get("persistent_workers", False) and self.data_cfg.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader (uses test set by default)."""
        if self.test_ds is None:
            self.setup(stage="test")
        return self.test_dataloader()


def chest_ct(
    raw_dir: str | Path = "data/raw",
    image_size: int = 224,
) -> tuple[Dataset, Dataset, Dataset]:
    """Convenience helper to load train, val, and test datasets from raw images.

    Args:
        raw_dir: Path to raw data directory
        image_size: Target image size

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_root = _find_data_root(Path(raw_dir))

    train_ds = ChestCTDataset(data_root, split="train", image_size=image_size)
    val_ds = ChestCTDataset(data_root, split="valid", image_size=image_size)
    test_ds = ChestCTDataset(data_root, split="test", image_size=image_size)

    return train_ds, val_ds, test_ds


def processed_chest_ct(
    processed_dir: str | Path = "data/processed",
) -> tuple[Dataset, Dataset, Dataset]:
    """Convenience helper to load train, val, and test datasets from processed tensors.

    Args:
        processed_dir: Path to processed data directory

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    processed_path = Path(processed_dir)

    train_ds = ProcessedChestCTDataset(processed_path, split="train")
    val_ds = ProcessedChestCTDataset(processed_path, split="valid")
    test_ds = ProcessedChestCTDataset(processed_path, split="test")

    return train_ds, val_ds, test_ds


# CLI Application
app = typer.Typer()


@app.command()
def download(
    target_dir: str = typer.Option("data/raw", help="Directory to save downloaded data"),
) -> None:
    """Download CT scan dataset from Kaggle.

    Example:
        python -m ct_scan_mlops.data download
        python -m ct_scan_mlops.data download --target-dir data/raw
    """
    download_data(target_dir=target_dir)


@app.command(name="preprocess")
def preprocess_cmd(
    raw_dir: str = typer.Option("data/raw", help="Path to raw data directory"),
    output_dir: str = typer.Option("data/processed", help="Path to save processed data"),
    image_size: int = typer.Option(224, help="Target image size"),
) -> None:
    """Preprocess raw CT scan images and save as tensors.

    This command loads all images from data/raw, resizes and normalizes them,
    then saves as .pt tensor files in data/processed for fast loading during training.

    Example:
        python -m ct_scan_mlops.data preprocess
        python -m ct_scan_mlops.data preprocess --image-size 256
    """
    preprocess(raw_dir=raw_dir, output_dir=output_dir, image_size=image_size)


if __name__ == "__main__":
    app()
