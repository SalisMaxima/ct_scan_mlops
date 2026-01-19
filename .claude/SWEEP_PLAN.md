# Plan: Model Architecture Experiments with W&B Sweeps

> **To execute this plan:** Ask Claude to "implement the sweep plan from .claude/SWEEP_PLAN.md"

## Overview

Design and implement a comprehensive hyperparameter sweep strategy for CT scan classification, testing multiple model architectures using Weights & Biases sweep functionality.

---

## Part 1: Problem Analysis

### Dataset Characteristics
| Metric | Value |
|--------|-------|
| **Total Images** | 1,000 |
| **Train/Val/Test Split** | 613 / 72 / 315 |
| **Image Size** | 224x224 RGB |
| **Classes** | 4 (adenocarcinoma, large_cell, squamous_cell, normal) |
| **Class Balance** | Imbalanced (19-32% per class) |

### Key Constraints
1. **Small Dataset** → High overfitting risk, transfer learning essential
2. **Medical Imaging** → Requires careful augmentation, interpretability matters
3. **Class Imbalance** → Need weighted loss or balanced sampling
4. **Limited Compute** → Efficient sweep strategy needed

---

## Part 2: Model Architectures to Test

**Selected Scope:** Tier 1 models only (3 architectures)
**Compute Budget:** ~75 runs (Medium)

### Models to Sweep

#### 1. ResNet18 (Transfer Learning) - BASELINE
**Why:** Best for small datasets, proven on medical imaging, your existing implementation
```yaml
parameters:
  pretrained: true
  freeze_backbone: [true, false]  # Feature extraction vs fine-tuning
```

#### 2. ResNet34 (Deeper Transfer Learning)
**Why:** More capacity than ResNet18, still efficient, same architecture family
```yaml
parameters:
  pretrained: true
  freeze_backbone: [true, false]
```

#### 3. EfficientNet-B0 (Efficient Architecture)
**Why:** State-of-the-art efficiency, compound scaling, excellent for small datasets
```yaml
parameters:
  pretrained: true
  drop_rate: [0.2, 0.3, 0.4]
```

---

## Part 3: Hyperparameters to Sweep

### Learning Rate & Optimization
| Parameter | Range | Scale |
|-----------|-------|-------|
| `lr` | 1e-5 to 1e-2 | log |
| `weight_decay` | 1e-6 to 1e-3 | log |
| `optimizer` | adam, adamw, sgd | categorical |

### Model Architecture
| Parameter | Values |
|-----------|--------|
| `model_name` | resnet18, resnet34, efficientnet_b0 |
| `freeze_backbone` | true, false |
| `dropout` | 0.2, 0.3, 0.4, 0.5 |

### Training
| Parameter | Range |
|-----------|-------|
| `batch_size` | 16, 32, 64 |
| `max_epochs` | 30, 50, 75 |
| `scheduler` | cosine, step, plateau |

### Regularization
| Parameter | Values |
|-----------|--------|
| `label_smoothing` | 0.0, 0.1, 0.2 |
| `mixup_alpha` | 0.0, 0.2, 0.4 |
| `weight_decay` | 1e-5, 1e-4, 1e-3 |

---

## Part 4: Sweep Strategy (~75 runs total)

### Phase 1: Architecture + LR Search (Bayesian, 45 runs)
**Goal:** Find best model architecture and learning rate together
**Sweep Parameters:**
- `model`: resnet18, resnet34, efficientnet_b0
- `freeze_backbone`: true, false
- `lr`: log_uniform [1e-5, 1e-2]
- `weight_decay`: log_uniform [1e-6, 1e-3]
- `batch_size`: 16, 32, 64

### Phase 2: Fine-Tuning Top Models (Bayesian, 20 runs)
**Goal:** Optimize regularization for top 2 architectures from Phase 1
**Fixed:** Best architecture + LR from Phase 1
**Sweep Parameters:**
- `dropout`: 0.2, 0.3, 0.4, 0.5
- `label_smoothing`: 0.0, 0.1, 0.2

### Phase 3: Final Validation (Grid, 10 runs)
**Goal:** Train final models with best configs, multiple seeds
**Action:** Top 2 configurations × 5 random seeds each

---

## Part 5: Implementation Plan

### Step 1: Add New Model Architectures to model.py

```python
# Add to src/ct_scan_mlops/model.py

import timm

class TimmModel(nn.Module):
    """Universal wrapper for timm models (ResNet34, EfficientNet, etc.)."""

    def __init__(
        self,
        model_name: str = "resnet34",
        num_classes: int = 4,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

**Update build_model() to include:**
```python
if name in {"resnet34", "efficientnet_b0", "efficientnet", "densenet121", "mobilenetv3"}:
    return TimmModel(
        model_name=model_cfg.get("timm_model", name),
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.get("pretrained", True),
        drop_rate=model_cfg.get("drop_rate", 0.0),
    )
```

### Step 2: Create Sweep Configuration File

**File:** `configs/sweep.yaml`

```yaml
program: src/ct_scan_mlops/sweep.py
method: bayes
metric:
  name: val_acc
  goal: maximize

parameters:
  model_name:
    values: [resnet18, resnet34, efficientnet_b0]

  freeze_backbone:
    values: [true, false]

  lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.01

  weight_decay:
    distribution: log_uniform_values
    min: 0.000001
    max: 0.001

  batch_size:
    values: [16, 32, 64]

  dropout:
    values: [0.2, 0.3, 0.4]

early_terminate:
  type: hyperband
  min_iter: 5
  eta: 3

run_cap: 45
```

### Step 3: Create Sweep Entry Point

**File:** `src/ct_scan_mlops/sweep.py`

```python
"""W&B Sweep agent for hyperparameter optimization."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ct_scan_mlops.train import train_model, configure_logging, get_device
from ct_scan_mlops.data import create_dataloaders
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


def sweep_train():
    """Training function called by W&B sweep agent."""
    # Initialize wandb run (sweep agent handles this)
    run = wandb.init()

    # Get sweep hyperparameters
    sweep_config = dict(wandb.config)

    # Load base config
    with hydra.initialize(config_path="../../configs", version_base="1.3"):
        cfg = hydra.compose(config_name="config")

    # Override with sweep parameters
    if "model_name" in sweep_config:
        cfg.model.name = sweep_config["model_name"]
    if "lr" in sweep_config:
        cfg.train.optimizer.lr = sweep_config["lr"]
    if "weight_decay" in sweep_config:
        cfg.train.optimizer.weight_decay = sweep_config["weight_decay"]
    if "batch_size" in sweep_config:
        cfg.data.batch_size = sweep_config["batch_size"]
    if "dropout" in sweep_config:
        cfg.model.dropout = sweep_config["dropout"]
    if "freeze_backbone" in sweep_config:
        cfg.model.freeze_backbone = sweep_config["freeze_backbone"]

    # Set shorter epochs for sweep
    cfg.train.max_epochs = 30

    # Create output directory
    output_dir = Path("outputs/sweeps") / run.id
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(str(output_dir))

    # Create WandbLogger
    wandb_logger = WandbLogger(
        experiment=run,
        save_dir=str(output_dir),
    )

    # Train
    try:
        train_model(cfg, str(output_dir), wandb_logger)
    except Exception as e:
        print(f"Training failed: {e}")
        wandb.finish(exit_code=1)
        raise

    wandb.finish()


if __name__ == "__main__":
    sweep_train()
```

### Step 4: Add Model Config Files

**File:** `configs/model/resnet34.yaml`
```yaml
name: resnet34
num_classes: 4
pretrained: true
freeze_backbone: false
timm_model: resnet34
drop_rate: 0.0
```

**File:** `configs/model/efficientnet.yaml`
```yaml
name: efficientnet_b0
num_classes: 4
pretrained: true
timm_model: efficientnet_b0
drop_rate: 0.3
```

### Step 5: Add Sweep Commands to tasks.py

```python
@task
def sweep_create(ctx: Context, config: str = "configs/sweep.yaml", project: str = "CT_Scan_MLOps") -> None:
    """Create a new W&B sweep.

    Args:
        config: Path to sweep configuration YAML
        project: W&B project name

    Example:
        invoke sweep-create
        invoke sweep-create --config configs/sweep_phase2.yaml
    """
    ctx.run(f"wandb sweep {config} --project {project}", echo=True, pty=not WINDOWS)


@task
def sweep_agent(ctx: Context, sweep_id: str, count: int = 10, entity: str = "") -> None:
    """Run W&B sweep agent.

    Args:
        sweep_id: The sweep ID from wandb sweep command
        count: Number of runs to execute
        entity: W&B entity (optional)

    Example:
        invoke sweep-agent --sweep-id abc123 --count 45
    """
    entity_flag = f"--entity {entity}" if entity else ""
    ctx.run(f"wandb agent {entity_flag} {sweep_id} --count {count}", echo=True, pty=not WINDOWS)
```

---

## Part 6: Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/ct_scan_mlops/model.py` | Modify | Add TimmModel class for ResNet34 + EfficientNet |
| `src/ct_scan_mlops/sweep.py` | Create | Sweep agent entry point with Hydra integration |
| `configs/sweep.yaml` | Create | Phase 1 sweep config (architecture + LR) |
| `configs/sweep_phase2.yaml` | Create | Phase 2 sweep config (regularization) |
| `configs/model/resnet34.yaml` | Create | ResNet34 config |
| `configs/model/efficientnet.yaml` | Create | EfficientNet-B0 config |
| `tasks.py` | Modify | Add sweep-create, sweep-agent commands |

---

## Part 7: Verification Plan

### Test 1: Model Architecture Verification
```bash
# Test each new model can be instantiated
uv run python -c "
from ct_scan_mlops.model import TimmModel
import torch

# Test ResNet34
model = TimmModel('resnet34', num_classes=4, pretrained=False)
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(f'ResNet34 output shape: {out.shape}')

# Test EfficientNet
model = TimmModel('efficientnet_b0', num_classes=4, pretrained=False)
out = model(x)
print(f'EfficientNet output shape: {out.shape}')
"
```

### Test 2: Single Training Run with New Model
```bash
# Test training works with new model
invoke train --args "model=resnet34 train.max_epochs=2 wandb.mode=disabled"
invoke train --args "model=efficientnet train.max_epochs=2 wandb.mode=disabled"
```

### Test 3: Sweep Configuration Validation
```bash
# Validate sweep config syntax (if wandb CLI supports dry-run)
cat configs/sweep.yaml
```

### Test 4: Full Sweep Test (3 runs)
```bash
# Create sweep and run 3 test iterations
invoke sweep-create
# Note the sweep ID printed
invoke sweep-agent --sweep-id <SWEEP_ID> --count 3
```

### Test 5: Results Analysis
- Check W&B dashboard for logged metrics
- Verify hyperparameters are correctly recorded
- Confirm model comparison charts work

---

## Part 8: Expected Outcomes

### Metrics to Track
- `val_acc` (primary metric - sweep optimization target)
- `val_loss`
- `train_acc`
- `train_loss`
- `lr` (learning rate schedule)
- Training time per epoch
- GPU memory usage
- Model parameter count

### Deliverables
1. Best model architecture identified
2. Optimal hyperparameters for top 3 models
3. W&B sweep dashboard with all experiments
4. Final trained model with best configuration
5. Comparison report of all architectures

---

## Part 9: Execution Commands

### Complete Workflow

```bash
# Step 1: Create Phase 1 sweep
invoke sweep-create --config configs/sweep.yaml

# Step 2: Run Phase 1 (45 runs) - copy sweep ID from step 1
invoke sweep-agent --sweep-id <SWEEP_ID> --count 45

# Step 3: Analyze results in W&B dashboard
# - Go to https://wandb.ai/<entity>/CT_Scan_MLOps/sweeps/<sweep_id>
# - Identify top 2 performing model architectures
# - Note their best hyperparameters

# Step 4: Create Phase 2 sweep for top models (update sweep_phase2.yaml first)
invoke sweep-create --config configs/sweep_phase2.yaml

# Step 5: Run Phase 2 (20 runs)
invoke sweep-agent --sweep-id <SWEEP_ID> --count 20

# Step 6: Final training with best configuration
invoke train --args "model=<best_model> train.optimizer.lr=<best_lr> train.optimizer.weight_decay=<best_wd> model.dropout=<best_dropout>"
```

### Quick Test (Verify Infrastructure)

```bash
# Run 3 quick sweep iterations to verify everything works
invoke sweep-create
invoke sweep-agent --sweep-id <ID> --count 3
```

---

## Summary

**Total Runs:** ~75
**Models:** ResNet18, ResNet34, EfficientNet-B0
**Primary Metric:** val_acc (maximize)
**Search Method:** Bayesian optimization with Hyperband early termination

**Phase Breakdown:**
1. Phase 1 (45 runs): Architecture + LR search
2. Phase 2 (20 runs): Regularization tuning
3. Phase 3 (10 runs): Final validation with seeds
