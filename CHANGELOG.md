# Changelog

All notable changes to the CT Scan MLOps project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-01-31

### 🎉 Major Release: Dual Pathway Architecture

This release introduces a dual pathway neural network that combines CNN image features with engineered radiomics features, achieving **95.87% test accuracy** (vs ~92% baseline). Includes comprehensive analysis tools, experiment automation, and production deployment enhancements.

### Added

#### Core Features
- **Dual Pathway Model** (`src/ct_scan_mlops/model.py`)
  - CNN pathway with ResNet18 backbone
  - Radiomics pathway with 50+ engineered features
  - Batch normalization for training stability
  - Configurable fusion strategies
  - Support for backbone freezing and custom projections

- **Radiomics Feature Extraction** (`src/ct_scan_mlops/features/`)
  - `intensity.py` - Statistical intensity features (mean, std, skewness, kurtosis, entropy)
  - `texture.py` - GLCM texture features (13 features: contrast, homogeneity, energy, etc.)
  - `shape.py` - Geometric shape features (9 features: area, perimeter, eccentricity, sphericity, etc.)
  - `region.py` - Region-based features (6 features: solid ratio, GGO ratio, density profiles)
  - `wavelet.py` - Multi-scale wavelet features (14 features from Daubechies wavelets)
  - `extractor.py` - Main extraction pipeline with caching and batch processing
  - Total: 50+ features, with top 16 feature subset for optimal performance

- **Analysis & Explainability Tools** (`src/ct_scan_mlops/analysis/`)
  - `cli.py` - Unified command-line interface for all analysis tools
  - `diagnostics.py` - Error analysis, confusion matrices, confidence distributions
  - `comparison.py` - Baseline vs improved model comparison with statistical tests
  - `explainability.py` - Permutation importance and gradient-based attribution
  - `sweep_report.py` - Programmatic W&B report generation
  - `core.py` - Shared analysis utilities
  - `utils.py` - Helper functions for metrics and visualization

- **Advanced Loss Functions** (`src/ct_scan_mlops/losses.py`)
  - Focal Loss for class imbalance
  - Label Smoothing Cross Entropy
  - Class-weighted Cross Entropy
  - Configurable via Hydra

- **Experiment Automation Scripts** (`scripts/`)
  - `phase2_finetune_experiments.sh` - Phase 2 fine-tuning workflow (170 lines)
  - `phase2_run_and_shutdown.sh` - Automated experiment queue with system shutdown (627 lines)
  - `adenocarcinoma_improvement_experiments.sh` - Targeted experiments for class confusion (415 lines)
  - `tier1_loss_experiments.sh` - Loss function ablation studies (277 lines)
  - `analyze_tier1_results.py` - Post-experiment analysis automation (401 lines)
  - `check_tier1_progress.sh` - Progress monitoring script (79 lines)
  - `export_onnx.py` - Model export for production (95 lines)
  - `benchmark_onnx.py` - ONNX runtime benchmarking (94 lines)
  - `profile_training_time.py` - Training performance profiling (420 lines)
  - `convert_config.py` - Config migration utilities (67 lines)
  - `get_misclassified_files.py` - Extract misclassified samples (97 lines)

- **Claude Code Agents** (`.claude/agents/`)
  - `code-reviewer.md` - Automated code review with linting integration
  - `dependabot-pr-merger.md` - Smart dependency update batching and analysis
  - `ml-experiment-planner.md` - Experiment design assistant (Opus-powered, 131 lines)
  - `gemini-analyzer.md` - Large codebase analysis delegation (44 lines)
  - `repo-structure-enforcer.md` - File organization compliance checker (106 lines)

- **Claude Code Skill** (`.claude/skills/`)
  - `repo-consolidation.md` - Directory structure optimization skill (314 lines)

- **Configuration Files**
  - `configs/features/default.yaml` - All 50 features configuration
  - `configs/features/top_features.yaml` - Top 16 features (optimized)
  - `configs/model/dual_pathway.yaml` - Base dual pathway config
  - `configs/model/dual_pathway_top_features.yaml` - Optimized config
  - `configs/model/dual_pathway_bn_finetune_kygevxv0.yaml` - Best checkpoint config (236 lines)
  - `configs/model/dual_pathway_bn_finetune_kygevxv0_clean.yaml` - Cleaned config (251 lines)
  - `configs/sweeps/dual_pathway_sweep.yaml` - Bayesian optimization sweep (79 lines)
  - `configs/sweeps/dual_pathway_finetune.yaml` - Fine-tuning sweep (81 lines)
  - `configs/sweeps/model_comparison_sweep.yaml` - Architecture comparison sweep (46 lines)

- **Documentation**
  - `docs/MIGRATION_DUAL_PATHWAY.md` - Comprehensive migration guide (600+ lines)
  - `docs/MIGRATION_QUICK_REFERENCE.md` - Quick reference card
  - `docs/ExperimentPlan_Phase2.md` - Phase 2 experiment methodology (322 lines)
  - `docs/ExperimentAnalysis_20260129.md` - Detailed results analysis (192 lines)
  - `docs/AdenocarcinomaImprovementPlan.md` - Targeted improvement strategy (549 lines)
  - `ToDo.md` - Architecture improvement roadmap (73 lines)
  - `GEMINI.md` - Gemini CLI integration guide (47 lines)
  - `CHANGELOG.md` - This file

- **Testing**
  - `tests/test_features.py` - Feature extraction validation (418 lines)
  - `tests/test_analysis_utils.py` - Analysis utility tests (238 lines)
  - `tests/test_checkpoint.py` - Checkpoint loading tests (120 lines)
  - Enhanced `tests/test_model.py` with dual pathway tests

- **Deployment**
  - `cloudbuild-api.yaml` - Cloud Build configuration for API (25 lines)
  - `test_api.sh` - API testing script (42 lines)
  - `dockerfiles/api.cloudrun.dockerfile` - Cloud Run optimized Dockerfile

- **Invoke Commands**
  - `invoke extract-features` - Extract radiomics features
  - `invoke prepare-sweep-features` - Prepare all feature configs for sweeps
  - `invoke analyze-errors` - Error analysis with visualizations
  - `invoke analyze-features` - Feature importance analysis
  - `invoke compare-baselines` - Compare two models
  - `invoke sync-ai-config` - Sync CLAUDE.md to copilot-instructions.md

### Changed

#### Training & Evaluation
- **Sweep metric changed from `val_acc` to `test_acc`**
  - Models now automatically evaluate on test set after training
  - Better indicator of generalization performance
  - W&B sweeps optimize for test accuracy
  - Impact: Old sweeps need config update

- **Training loop enhancements** (`src/ct_scan_mlops/train.py`)
  - Support for dual pathway models
  - Automatic test evaluation when `train.test_after_training=true`
  - Improved checkpoint handling
  - Better logging and metrics

- **Sweep training** (`src/ct_scan_mlops/sweep_train.py`)
  - Support for dual pathway hyperparameter search
  - Feature config selection in sweep parameters
  - Improved error handling for missing features
  - Better W&B integration

#### Data Pipeline
- **Enhanced data module** (`src/ct_scan_mlops/data.py`)
  - Feature loading for dual pathway models
  - Weighted sampling support for class imbalance
  - Enhanced augmentation pipeline
  - Better caching and preprocessing
  - Feature validation and error messages

- **Data configuration** (`configs/data/chest_ct.yaml`)
  - Increased rotation limit (15° → 30°) for better rotation invariance
  - Added optional advanced augmentations (elastic transform, grid distortion, coarse dropout, etc.)
  - Weighted sampling configuration for class balancing
  - All new augmentations disabled by default for backward compatibility

#### API & Deployment
- **API enhancements** (`src/ct_scan_mlops/api.py`)
  - Support for dual pathway model serving
  - Automatic model type detection
  - Feature extraction in inference pipeline
  - Improved error handling and validation
  - Health check endpoints

- **Dockerfile improvements**
  - `dockerfiles/api.cloudrun.dockerfile` - Cloud Run optimized image
  - Model checkpoint baking into container
  - Reduced image size
  - Better caching

- **Deployment workflow** (`.github/workflows/deploy_cloudrun.yml`)
  - Updated for dual pathway models
  - Improved build process
  - Better error handling

#### Evaluation System
- **Refactored evaluate.py** (`src/ct_scan_mlops/evaluate.py`)
  - Now uses modular analysis tools from `analysis/` module
  - Cleaner code with separation of concerns
  - Better error messages
  - Backward compatible CLI

- **Enhanced evaluation documentation** (`docs/source/api-reference/evaluate.md`)
  - Updated with new analysis commands
  - Examples for dual pathway models
  - Troubleshooting section

#### Project Structure
- **Moved monitoring to submodule** (`src/ct_scan_mlops/monitoring/`)
  - Previously: `src/ct_scan_mlops/drift_*.py`
  - Now: `src/ct_scan_mlops/monitoring/drift_*.py`
  - Backward compatible imports via `__init__.py`

- **Moved scripts from root to `scripts/` directory**
  - Better organization
  - Clearer separation of utilities

- **Consolidated outputs to `outputs/` directory**
  - All training artifacts, logs, plots now in `outputs/`
  - Prevents root directory pollution
  - Better .gitignore patterns

#### CI/CD Infrastructure
- **GitHub Actions upgrades**
  - `actions/labeler` v5 → v6
  - `google-github-actions/setup-gcloud` v2 → v3
  - `actions/create-github-app-token` v1 → v2
  - `iterative/setup-cml` v2 → v3
  - `astral-sh/setup-uv` v4 → v7

- **Docker publish workflow**
  - Changed from automatic (on push) to manual trigger only
  - Safer deployment process
  - Prevents accidental publishes

- **CML workflow** (`.github/workflows/cml_data.yaml`)
  - Updated for dual pathway support
  - Better reporting

#### Documentation
- **Updated README.md**
  - Added dual pathway model description
  - New command examples
  - Migration guide link
  - Performance comparisons
  - Sweep usage with dual pathway

- **Updated CLAUDE.md**
  - New sweep commands
  - Analysis commands
  - Feature extraction commands

- **Updated Structure.md**
  - New modules documented
  - Updated directory structure
  - Cross-references to analysis tools

- **Updated API reference docs**
  - Dual pathway model documentation
  - Feature extraction guide
  - Analysis tools reference

### Fixed

- **Training stability issues**
  - Added BatchNorm to dual pathway model projections
  - Reduced loss oscillations
  - Faster convergence

- **macOS multiprocessing errors**
  - Set `num_workers=0` in data config for macOS
  - Resolved DataLoader deadlocks

- **NumPy serialization issues in W&B**
  - Fixed JSON serialization for sweep reports
  - Proper type conversion

- **Feature dimension mismatches**
  - Clear error messages when features don't match model
  - Validation in data module

### Deprecated

Nothing deprecated. All v1.x features remain fully functional.

### Removed

- Cleaned up redundant utility scripts from root
- Removed unused imports
- Cleaned up temporary experiment files

### Security

No security changes in this release.

### Dependencies

#### Added
- `scikit-image` - For radiomics feature extraction
- `pywavelets` - For wavelet transform features
- `wandb` updates for report API support

#### Updated
- `torch` 2.9.1 → 2.10.0
- `torchvision` 0.24.1 → 0.25.0
- `streamlit` 1.53.0 → 1.53.1
- `ruff` 0.14.13 → 0.14.14
- `kagglehub` 0.4.0 → 0.4.1

### Performance Metrics

| Model | Test Accuracy | Training Time (V100) | Inference Time |
|-------|---------------|----------------------|----------------|
| CNN (v1.x baseline) | 91.2% | 1.5h | 8ms |
| ResNet18 (v1.x) | 92.5% | 2h | 8ms |
| Dual Pathway (top features) | **95.87%** | 2.5h | 12ms |
| Dual Pathway (all features) | 95.5% | 3h | 12ms |

### Migration Notes

- **Backward compatible**: All v1.x models and configs continue to work
- **Opt-in**: Dual pathway features require explicit model selection
- **Prerequisites**: Dual pathway models require feature extraction (`invoke extract-features`)
- **Sweep metric**: Update sweep configs to use `test_acc` instead of `val_acc`

See [MIGRATION_DUAL_PATHWAY.md](docs/MIGRATION_DUAL_PATHWAY.md) for detailed migration guide.

### Contributors

This release represents significant contributions from the CT Scan MLOps team with automated code review, experiment planning, and analysis powered by Claude Code agents.

---

## [1.0.0] - 2025-12-XX (Approximate)

### Added
- Initial CNN baseline model
- ResNet18 transfer learning model
- PyTorch Lightning training pipeline
- W&B experiment tracking
- DVC data versioning
- Hydra configuration management
- Basic evaluation metrics
- Kaggle data download
- Data preprocessing pipeline
- Docker containers for training and API
- GitHub Actions CI/CD
- Documentation and tutorials

### Initial Release Features
- 4-class lung tumor classification
- Training on Chest CT-Scan Images Dataset
- Data augmentation (rotation, flip, brightness, contrast)
- W&B team collaboration setup
- Basic sweep support
- API deployment
- Cloud infrastructure setup

---

## Format Guidelines

### Types of Changes
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Vulnerability fixes

### Version Format
- **Major** (X.0.0) - Breaking changes
- **Minor** (x.X.0) - New features, backward compatible
- **Patch** (x.x.X) - Bug fixes, backward compatible

---

**Full diff**: [`master...dual_pathway_workflow`](https://github.com/DTU-MLOps-Group-2/ct_scan_mlops/compare/master...dual_pathway_workflow)
