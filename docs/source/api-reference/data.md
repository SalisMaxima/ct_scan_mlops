# Data Module

Data loading and preprocessing for CT scan images.

## Overview

The data module provides:

- **Dataset classes** for loading raw images or preprocessed tensors
- **Lightning DataModule** for integration with PyTorch Lightning
- **Preprocessing functions** to convert raw images to tensors
- **Data augmentation** with Albumentations

## Constants

```python
# Canonical class labels
CLASSES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma",
    "normal",
]

# ImageNet normalization stats (default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

## Dataset Classes

::: ct_scan_mlops.data.ChestCTDataset
    options:
      heading_level: 3

::: ct_scan_mlops.data.ProcessedChestCTDataset
    options:
      heading_level: 3

## Lightning DataModule

::: ct_scan_mlops.data.ChestCTDataModule
    options:
      heading_level: 3

## Functions

### Data Download

::: ct_scan_mlops.data.download_data

### Preprocessing

::: ct_scan_mlops.data.preprocess

### Transform Creation

::: ct_scan_mlops.data.get_transforms

### DataLoader Creation

::: ct_scan_mlops.data.create_dataloaders

### Convenience Helpers

::: ct_scan_mlops.data.chest_ct

::: ct_scan_mlops.data.processed_chest_ct

## Usage Examples

### Load Preprocessed Data

```python
from ct_scan_mlops.data import ProcessedChestCTDataset

train_ds = ProcessedChestCTDataset("data/processed", split="train")
val_ds = ProcessedChestCTDataset("data/processed", split="valid")
test_ds = ProcessedChestCTDataset("data/processed", split="test")
```

### Use with Lightning

```python
from ct_scan_mlops.data import ChestCTDataModule

datamodule = ChestCTDataModule(cfg)
datamodule.setup(stage="fit")

trainer.fit(model, datamodule=datamodule)
```

### Preprocess Raw Data

```bash
python -m ct_scan_mlops.data preprocess
```
