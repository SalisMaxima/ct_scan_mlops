# API Reference

Auto-generated documentation from source code docstrings.

## Modules

| Module | Description |
|--------|-------------|
| [Data](data.md) | Data loading and preprocessing |
| [Model](model.md) | Neural network architectures |
| [Training](train.md) | Training pipeline and Lightning module |
| [Evaluation](evaluate.md) | Model evaluation (legacy compatibility wrapper) |
| [FastAPI](api.md) | Inference API endpoints |
| [Utilities](utils.md) | Shared utility functions |

### Analysis Package (New)

The modular analysis system:
- `ct_scan_mlops.analysis.core` - Model loading and inference engine
- `ct_scan_mlops.analysis.diagnostics` - Performance evaluation and error analysis
- `ct_scan_mlops.analysis.explainability` - Feature importance methods
- `ct_scan_mlops.analysis.comparison` - Model comparison utilities
- `ct_scan_mlops.analysis.cli` - Unified command-line interface

## Quick Links

### Data Module

- [`ChestCTDataset`](data.md#ct_scan_mlops.data.ChestCTDataset) - Raw image dataset
- [`ProcessedChestCTDataset`](data.md#ct_scan_mlops.data.ProcessedChestCTDataset) - Preprocessed tensor dataset
- [`ChestCTDataModule`](data.md#ct_scan_mlops.data.ChestCTDataModule) - Lightning DataModule
- [`preprocess()`](data.md#ct_scan_mlops.data.preprocess) - Preprocess raw images to tensors

### Model Module

- [`CustomCNN`](model.md#ct_scan_mlops.model.CustomCNN) - Configurable CNN
- [`ResNet18`](model.md#ct_scan_mlops.model.ResNet18) - Transfer learning model
- [`build_model()`](model.md#ct_scan_mlops.model.build_model) - Build model from config

### Training Module

- [`LitModel`](train.md#ct_scan_mlops.train.LitModel) - Lightning module wrapper
- [`train_model()`](train.md#ct_scan_mlops.train.train_model) - Main training function

### Evaluation Module (Legacy)

> **Note**: For new code, use `ct_scan_mlops.analysis` instead.

- [`evaluate_model()`](evaluate.md#ct_scan_mlops.evaluate.evaluate_model) - Evaluate a model (backward compatibility)
- [`load_model_from_checkpoint()`](evaluate.md#ct_scan_mlops.evaluate.load_model_from_checkpoint) - Load model from checkpoint (backward compatibility)
