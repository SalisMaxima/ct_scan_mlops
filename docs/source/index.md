# CT Scan MLOps

Chest CT scan multi-classification model for lung tumor detection using PyTorch Lightning.

## Project Goal

This project builds an image multi-classification model that detects whether a chest CT scan shows signs of three different types of lung tumor or is normal:

1. **Adenocarcinoma** (left lower lobe)
2. **Large cell carcinoma** (left hilum)
3. **Squamous cell carcinoma** (left hilum)
4. **Normal**

## Key Features

- **PyTorch Lightning** for structured, scalable training
- **Hydra** for flexible configuration management
- **Weights & Biases** for experiment tracking and team collaboration
- **DVC** for data versioning with GCS backend
- **FastAPI** for model inference API
- **Docker** support for containerized training and deployment
- **Comprehensive testing** with pytest

## Quick Start

```bash
# Clone and setup
git clone https://github.com/SalisMaxima/ct_scan_mlops.git
cd ct_scan_mlops
uv venv && source .venv/bin/activate
uv sync --all-groups

# Pull data and train
dvc pull
invoke train
```

## Models

The project includes two model architectures:

| Model | Description |
|-------|-------------|
| **CustomCNN** | Configurable CNN baseline for comparison |
| **ResNet18** | Transfer learning with pretrained ImageNet weights |

## Dataset

**Source:** [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle

The dataset contains ~1000 images with balanced volume across 4 classifications. Data augmentation with rotations and translations is applied to improve model robustness.

## Documentation Sections

<div class="grid cards" markdown>

-   :material-download: **[Getting Started](getting-started/index.md)**

    ---

    Installation, setup, and first steps

-   :material-book-open: **[User Guide](user-guide/index.md)**

    ---

    Training, evaluation, and configuration

-   :material-api: **[API Reference](api-reference/index.md)**

    ---

    Module and class documentation

-   :material-wrench: **[Development](development/index.md)**

    ---

    Contributing and project structure

</div>

## Links

- [GitHub Repository](https://github.com/SalisMaxima/ct_scan_mlops)
- [W&B Dashboard](https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps)
- [Kaggle Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- [DTU MLOps Course](https://skaftenicki.github.io/dtu_mlops/)
