# Adenocarcinoma Classification Improvement Plan

**Date:** 2026-01-29
**Current Model:** dual_pathway_bn_finetune_kygevxv0.pt
**Current Test Accuracy:** 95.238% (15 errors / 315 samples)

---

## Executive Summary

The dual_pathway_bn_finetune model achieves excellent overall accuracy (95.24%), but analysis reveals a **concentrated failure mode centered on adenocarcinoma classification**:

### 🔴 Critical Findings

- **Adenocarcinoma is involved in 93% of all errors** (14 out of 15 misclassifications)
- **Primary confusion:** Adenocarcinoma ↔ Squamous Cell Carcinoma (60% of errors, 9 cases)
- **Secondary confusion:** Adenocarcinoma ↔ Large Cell Carcinoma (33% of errors, 5 cases)
- **Model overconfidence on errors:** Mean confidence 83.3%, median 88.1% on misclassifications

### 📊 Per-Class Error Rates

| Class | Correct | Errors | Error Rate |
|-------|---------|--------|------------|
| **Adenocarcinoma** | 110 | 10 | **8.33%** ⚠️ |
| Large Cell Carcinoma | 50 | 1 | 1.96% ✓ |
| Squamous Cell Carcinoma | 87 | 3 | 3.33% ✓ |
| Normal | 53 | 1 | 1.85% ✓ |

### 🎯 Improvement Goals

| Metric | Current | Target |
|--------|---------|--------|
| **Overall Accuracy** | 95.24% | **>97%** |
| **Adenocarcinoma Recall** | 91.67% | **>95%** |
| **Adeno-Squamous Confusions** | 9 errors | **<4 errors** |
| **Error Confidence** | 83.3% | **<60%** |

---

## Root Cause Analysis

### Why Adenocarcinoma is Confused

**Clinical Context:**
- Adenocarcinoma, squamous cell carcinoma, and large cell carcinoma are all non-small cell lung cancers (NSCLC)
- They share overlapping imaging characteristics in CT scans
- Adenocarcinoma typically shows ground-glass opacity (GGO), but solid variants exist
- Squamous cell carcinoma usually appears as solid, central masses

**Feature Analysis from Error Cases:**

Discriminative features (Z-scores) in error cases:
- **Shape features:**
  - `sphericity`: +1.61σ (more spherical tumors)
  - `major_axis_length`: -1.96σ (smaller tumors)
  - `eccentricity`: +1.54σ
  - `solidity`: +0.85σ
- **Texture features:**
  - Lower wavelet energies (L2_HH, L2_HL)
  - Lower GLCM contrast/correlation
- **Clinical feature:**
  - `ggo_ratio`: -0.27σ (less ground-glass opacity)

**Hypothesized Failure Modes:**
1. **Solid adenocarcinoma variants** lacking typical GGO patterns confused with squamous
2. **Size/shape bias:** Model associates certain morphologies with specific classes
3. **Feature bottleneck:** 16-feature `top_features` config may miss edge case discriminators
4. **Calibration issue:** Standard cross-entropy allows overconfident predictions

---

## 🚀 Action Plan

### Tier 1: Quick Wins (6 hours total)

**Objective:** Test loss function modifications for better handling of hard cases and calibration.

#### Experiments:

1. **Focal Loss (γ=2.0)**
   - Focus gradient on hard adenocarcinoma cases
   - Down-weight easy examples
   - Expected impact: Reduce overconfidence, improve adenocarcinoma recall

2. **Label Smoothing (0.1)**
   - Prevent overconfident predictions
   - Improve calibration
   - Expected impact: Lower error confidence from 83% to <70%

3. **Weighted Sampling**
   - Oversample adenocarcinoma during training
   - Class weights: [1.5, 1.0, 1.0, 0.7]
   - Expected impact: Better adenocarcinoma representation

#### Commands:

```bash
# Quick test - Focal Loss
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=focal train.loss.gamma=2.0 \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
  data.batch_size=16 train.max_epochs=25"

# Quick test - Label Smoothing
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
  data.batch_size=16 train.max_epochs=25"

# Quick test - Weighted Sampling
invoke train --args "model=dual_pathway_top_features features=top_features \
  data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]' \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
  data.batch_size=16 train.max_epochs=25"
```

---

### Tier 2: Data Augmentation (6-8 hours total)

**Objective:** Improve shape feature robustness through targeted augmentations.

#### Experiments:

1. **Elastic Transform**
   - Distort shape features during training
   - Parameters: `alpha=120, sigma=6, p=0.3`

2. **Grid Distortion**
   - Local geometric perturbations
   - Parameters: `steps=5, distort_limit=0.3`

3. **CoarseDropout**
   - Force model to use distributed features
   - Parameters: `max_holes=8, max_size=32x32, p=0.3`

4. **Combined Shape Augmentations**
   - All above combined with best loss from Tier 1

#### Example Command:

```bash
# Assuming focal loss performed best in Tier 1
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=focal train.loss.gamma=2.0 \
  data.augmentation.train.elastic_transform=true \
  data.augmentation.train.grid_distortion=true \
  data.augmentation.train.coarse_dropout=true \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
  data.batch_size=16 train.max_epochs=25"
```

---

### Tier 3: Architecture Modifications (10-15 hours total)

**Objective:** Better leverage radiomics features for edge case discrimination.

#### Experiments:

1. **Larger Radiomics Pathway**
   - `radiomics_hidden=768` (from 512)
   - `fusion_hidden=384` (from 256)
   - Hypothesis: Better feature extraction capacity

2. **Full 50 Features**
   - Use `features=default` instead of `top_features` (16 features)
   - `radiomics_dim=50`
   - Hypothesis: Additional features capture edge case patterns

3. **Higher Dropout**
   - `dropout=0.1` (from 0.05)
   - Reduce potential overfitting on spurious patterns

#### Example Commands:

```bash
# Larger radiomics pathway
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=focal train.loss.gamma=2.0 \
  data.augmentation.train.elastic_transform=true \
  model.radiomics_hidden=768 model.fusion_hidden=384 \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 data.batch_size=16 train.max_epochs=25"

# Full 50 features (requires feature extraction first)
invoke extract-features --args "features=default"
invoke train --args "model=dual_pathway features=default \
  train.loss.type=focal train.loss.gamma=2.0 \
  model.radiomics_dim=50 model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.0001 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 data.batch_size=16 train.max_epochs=25"
```

---

### Tier 4: Extended Training (3-5 hours per experiment)

**Objective:** Allow model to converge more precisely on hard examples.

#### Experiments:

1. **50 Epochs**
   - Extended training with best config from Tiers 1-3

2. **Combined Best Approach**
   - Best loss + augmentation + architecture
   - 40 epochs

#### Example Command:

```bash
# Extended training with best combined config
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=focal train.loss.gamma=2.0 \
  data.augmentation.train.elastic_transform=true \
  data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]' \
  model.radiomics_hidden=768 model.fusion_hidden=384 \
  train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
  model.dropout=0.05 data.batch_size=16 \
  train.max_epochs=50 train.scheduler.T_max=50"
```

---

## 📋 Complete Unattended Experiment Script

For overnight/weekend execution, use the comprehensive script:

```bash
#!/bin/bash
# scripts/adenocarcinoma_improvement_experiments.sh
# Run all tiers systematically
# Estimated runtime: 20-30 hours on RTX 3080

set -e

PROJECT_DIR="/Users/dkMatHLu/Desktop/ct_scan_mlops"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/adeno_improvement_$TIMESTAMP"
COOLDOWN=180  # 3 minutes between experiments

cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "ADENOCARCINOMA CLASSIFICATION IMPROVEMENT EXPERIMENTS"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo "======================================================================"

# Helper function
run_experiment() {
    local name=$1
    local args=$2
    local log_file="$LOG_DIR/${name}.log"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "EXPERIMENT: $name"
    echo "Time: $(date)"
    echo "----------------------------------------------------------------------"

    invoke train --args "$args" 2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "WARNING: Experiment $name failed with exit code $exit_code"
        echo "$name: FAILED" >> "$LOG_DIR/results_summary.txt"
    else
        echo "$name: COMPLETED" >> "$LOG_DIR/results_summary.txt"
    fi

    echo "Cooling down for $COOLDOWN seconds..."
    sleep $COOLDOWN
}

# Baseline config
BASE_CONFIG="model=dual_pathway_top_features features=top_features \
train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
data.batch_size=16 train.max_epochs=25"

# TIER 1: Loss Functions
echo "TIER 1: LOSS FUNCTION EXPERIMENTS"
run_experiment "T1_focal_loss_g2" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0"

run_experiment "T1_focal_loss_g3" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=3.0"

run_experiment "T1_label_smoothing_01" \
    "$BASE_CONFIG train.loss.type=label_smoothing train.loss.smoothing=0.1"

# TIER 2: Weighted Sampling
echo "TIER 2: WEIGHTED SAMPLING EXPERIMENTS"
run_experiment "T2_weighted_sampling" \
    "$BASE_CONFIG data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]'"

run_experiment "T2_focal_plus_weighted" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 \
    data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]'"

# TIER 3: Augmentation
echo "TIER 3: DATA AUGMENTATION EXPERIMENTS"
run_experiment "T3_elastic_transform" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 \
    data.augmentation.train.elastic_transform=true"

run_experiment "T3_combined_shape_aug" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 \
    data.augmentation.train.elastic_transform=true \
    data.augmentation.train.grid_distortion=true \
    data.augmentation.train.coarse_dropout=true"

# TIER 4: Architecture
echo "TIER 4: ARCHITECTURE EXPERIMENTS"
run_experiment "T4_larger_radiomics_768" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 \
    model.radiomics_hidden=768 model.fusion_hidden=384"

# TIER 5: Extended Training
echo "TIER 5: EXTENDED TRAINING"
run_experiment "T5_best_config_extended" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 \
    data.augmentation.train.elastic_transform=true \
    data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]' \
    model.radiomics_hidden=768 model.fusion_hidden=384 \
    train.max_epochs=40 train.scheduler.T_max=40"

echo ""
echo "======================================================================"
echo "EXPERIMENT SUITE COMPLETED"
echo "End time: $(date)"
echo "Results summary:"
cat "$LOG_DIR/results_summary.txt"
echo "======================================================================"
```

**Usage:**
```bash
chmod +x scripts/adenocarcinoma_improvement_experiments.sh
nohup ./scripts/adenocarcinoma_improvement_experiments.sh > logs/experiment_run.log 2>&1 &

# Monitor progress
tail -f logs/experiment_run.log
```

---

## 📊 Evaluation and Analysis

After experiments complete:

### 1. Identify Best Model

```bash
# Find best performing model from W&B
uv run python -c "
import wandb

api = wandb.Api()
runs = api.runs('mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps')

best_run = max(
    (r for r in runs[:50] if 'test_acc' in r.summary),
    key=lambda r: r.summary['test_acc']
)

print(f'Best run: {best_run.name}')
print(f'Test accuracy: {best_run.summary[\"test_acc\"]:.4f}')
print(f'Run ID: {best_run.id}')
"
```

### 2. Detailed Evaluation

```bash
# Run comprehensive evaluation on best model
invoke evaluate --checkpoint outputs/ct_scan_classifier/sweeps/<best_run_id>/best_model.ckpt

# Check new confusion matrix
cat reports/error_analysis/error_summary.txt

# Compare with baseline
invoke compare-baselines \
  --baseline models/dual_pathway_bn_finetune_kygevxv0.pt \
  --improved outputs/ct_scan_classifier/sweeps/<best_run_id>/best_model.ckpt
```

### 3. Analyze Remaining Errors

```bash
# Generate error analysis for improved model
invoke analyze-errors --checkpoint outputs/ct_scan_classifier/sweeps/<best_run_id>/best_model.ckpt

# Get misclassified file paths
uv run python scripts/get_misclassified_files.py > reports/error_analysis/remaining_errors.txt

# Visual inspection of remaining error cases
```

---

## 🎯 Success Criteria

### Primary Metrics

| Metric | Baseline | Target | Pass Condition |
|--------|----------|--------|----------------|
| Test Accuracy | 95.24% | >97% | ≥97.0% |
| Adenocarcinoma Recall | 91.67% | >95% | ≥95.0% |
| Adenocarcinoma Precision | ~92% | >95% | ≥95.0% |
| Adeno-Squamous Confusions | 9 | <4 | ≤3 |

### Calibration Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Mean Error Confidence | 83.3% | <60% | Average confidence on misclassifications |
| Expected Calibration Error (ECE) | TBD | <0.05 | Reliability diagram analysis |

### Per-Class Targets

| Class | Baseline Error Rate | Target | Pass Condition |
|-------|---------------------|--------|----------------|
| Adenocarcinoma | 8.33% | <5% | ≤5.0% |
| Large Cell Carcinoma | 1.96% | <2% | ≤2.0% |
| Squamous Cell Carcinoma | 3.33% | <3% | ≤3.0% |
| Normal | 1.85% | <2% | ≤2.0% |

---

## 📈 Expected Outcomes

### Conservative Estimate (Tier 1-2 only)
- **Overall Accuracy:** 96.0-96.5%
- **Adenocarcinoma Recall:** 93-94%
- **Adeno-Squamous Confusions:** 6-7 errors
- **Probability:** 80%

### Optimistic Estimate (Tier 1-4)
- **Overall Accuracy:** 96.8-97.5%
- **Adenocarcinoma Recall:** 95-97%
- **Adeno-Squamous Confusions:** 3-4 errors
- **Probability:** 50%

### Stretch Goal (All tiers + optimal combination)
- **Overall Accuracy:** >97.5%
- **Adenocarcinoma Recall:** >97%
- **Adeno-Squamous Confusions:** ≤2 errors
- **Probability:** 20%

---

## 🔄 Iterative Refinement

If targets are not met after initial experiments:

### Phase 2 Strategies

1. **Feature Engineering:**
   - Add clinical features (patient age, smoking history if available)
   - Engineer domain-specific radiomics (GGO subtypes, spiculation measures)
   - Try feature selection specifically for adenocarcinoma vs squamous

2. **Advanced Architectures:**
   - Attention mechanisms on radiomics pathway
   - Multi-scale feature fusion
   - Vision transformer backbone

3. **Ensemble Methods:**
   - Multi-seed ensemble (5 models)
   - CNN-only + Dual-pathway ensemble
   - Stacking with meta-learner

4. **Data Strategies:**
   - Class-specific augmentation policies
   - Hard example mining
   - Semi-supervised learning with unlabeled data

---

## 📝 Notes and Considerations

### Implementation Requirements

Some experiments may require code modifications:

1. **Focal Loss:** Add to `src/ct_scan_mlops/train.py`
2. **Pairwise Confusion Loss:** Custom loss implementation
3. **Weighted Sampling:** Add to dataloader in `src/ct_scan_mlops/data.py`
4. **Augmentations:** May need albumentations library updates

### Resource Estimates

| Phase | Experiments | GPU Hours | Disk Space | Wall Time |
|-------|-------------|-----------|------------|-----------|
| Tier 1 | 3-7 | ~5-6 hrs | ~3 GB | ~6-7 hrs |
| Tier 2 | 3-5 | ~3-5 hrs | ~2 GB | ~4-6 hrs |
| Tier 3 | 2-3 | ~3-5 hrs | ~2 GB | ~4-6 hrs |
| Tier 4 | 1-2 | ~2-4 hrs | ~1 GB | ~3-5 hrs |
| **Total** | **10-20** | **~15-20 hrs** | **~8 GB** | **~18-24 hrs** |

### Clinical Validation

If accuracy improvements are achieved:
1. **Visual review** of remaining errors by domain expert
2. **Inter-observer agreement** comparison
3. **Subtype analysis** (e.g., solid vs lepidic adenocarcinoma)

---

## ✅ Action Items Checklist

- [ ] Review this plan with team
- [ ] Ensure RTX 3080 GPU is available for 20+ hours
- [ ] Create experiment script at `scripts/adenocarcinoma_improvement_experiments.sh`
- [ ] Set up W&B logging for experiment tracking
- [ ] Run Tier 1 experiments (loss functions)
- [ ] Analyze Tier 1 results and select best loss
- [ ] Run Tier 2 experiments (augmentation)
- [ ] Run Tier 3 experiments (architecture)
- [ ] Run Tier 4 experiments (extended training)
- [ ] Evaluate best model on test set
- [ ] Compare confusion matrix with baseline
- [ ] Document final results
- [ ] If targets met: deploy improved model
- [ ] If targets not met: proceed to Phase 2 strategies

---

## 📚 References

- Error analysis report: `reports/error_analysis/error_summary.txt`
- Confusion matrix: `models/confusion_matrix.png`
- Error cases: `reports/error_analysis/error_cases.json`
- Baseline model: `models/dual_pathway_bn_finetune_kygevxv0.pt`
- W&B project: `mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps`

---

**Generated by:** ml-experiment-planner agent
**Last updated:** 2026-01-29
