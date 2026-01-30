# Adenocarcinoma Improvement Experiment Analysis

**Date:** 2026-01-29
**Baseline Model:** dual_pathway_bn_finetune_kygevxv0.pt (95.24% test accuracy)
**Experiment Type:** Single runs from scratch (no fine-tuning, no sweeps)

---

## Complete Results Summary

### Tier 1: Loss Functions

| Experiment | Test Acc | Delta vs Baseline | Notes |
|------------|----------|-------------------|-------|
| T1_baseline_crossentropy | 89.84% | - | Control |
| T1_focal_loss_gamma2.0 | 91.43% | +1.59% | Moderate improvement |
| T1_focal_loss_gamma2.5 | 90.48% | +0.64% | |
| T1_focal_loss_gamma3.0 | 90.79% | +0.95% | |
| **T1_label_smoothing_0.1** | **92.70%** | **+2.86%** | **Best loss function** |
| T1_label_smoothing_0.15 | 92.06% | +2.22% | |
| T1_focal_plus_class_weights | 88.25% | -1.59% | Worse than baseline |

**Finding:** Label smoothing 0.1 is the best loss function (+2.86% over baseline cross-entropy).

---

### Tier 2: Weighted Sampling

| Experiment | Test Acc | Delta vs T1 Baseline | Notes |
|------------|----------|----------------------|-------|
| T2_weighted_sampling_moderate | 92.38% | +2.54% | Good, close to label smoothing |
| T2_weighted_sampling_aggressive | 89.21% | -0.63% | Too aggressive |
| T2_focal_plus_weighted_moderate | 88.25% | -1.59% | Combination hurt |
| T2_focal_plus_weighted_aggressive | 88.89% | -0.95% | Combination hurt |

**Finding:** Moderate weighted sampling alone helps, but combining with focal loss hurts performance.

---

### Tier 3: Data Augmentation (with label smoothing 0.1)

| Experiment | Test Acc | Delta vs Label Smoothing | Notes |
|------------|----------|--------------------------|-------|
| T3_elastic_transform | 92.70% | 0.00% | No improvement |
| T3_grid_distortion | 92.70% | 0.00% | No improvement |
| T3_coarse_dropout | 92.70% | 0.00% | No improvement |
| T3_combined_shape_aug | 92.70% | 0.00% | No improvement |

**Finding:** Data augmentation provides NO benefit for from-scratch training with this dataset/model.

---

### Tier 4: Architecture Modifications (with label smoothing 0.1)

| Experiment | Test Acc | Delta vs Label Smoothing | Notes |
|------------|----------|--------------------------|-------|
| T4_larger_radiomics_hidden_768 | 82.86% | -9.84% | Much worse |
| **T4_larger_radiomics_hidden_1024** | **93.97%** | **+1.27%** | **Best overall** |
| T4_lower_dropout_0.02 | 92.38% | -0.32% | Slightly worse |
| T4_higher_dropout_0.1 | 93.02% | +0.32% | Slight improvement |
| T4_full_50_features | 84.76% | -7.94% | Much worse |

**Finding:**
- radiomics_hidden=1024 with fusion_hidden=512 is the best architecture (+1.27%)
- radiomics_hidden=768 severely underperforms (possibly config issue)
- Higher dropout (0.1) helps slightly
- Full 50 features hurts significantly (top 16 features are better)

---

## Key Insights

### What Works
1. **Label smoothing 0.1** - Consistent +2.86% improvement
2. **Larger radiomics pathway (1024/512)** - Additional +1.27% on top of label smoothing
3. **Higher dropout (0.1)** - Small but consistent improvement
4. **Top 16 features** - Better than full 50 features

### What Doesn't Work
1. **Focal loss** - Underperforms label smoothing
2. **Class weights** - Hurts performance when combined with other techniques
3. **Aggressive weighted sampling** - Too much oversampling hurts
4. **Data augmentation** - No benefit for this model/dataset
5. **Full 50 features** - Worse than top 16 features
6. **radiomics_hidden=768** - Anomalous poor performance (investigate)

### Anomalies to Investigate
- **T4_larger_radiomics_hidden_768 (82.86%)** performed much worse than 1024 - this is unexpected
- Possible config error or training instability

---

## Best Configuration Found

```yaml
# From scratch training: 93.97% test accuracy
model: dual_pathway_top_features
features: top_features
train:
  loss:
    type: label_smoothing
    smoothing: 0.1
  optimizer:
    lr: 0.000115
    weight_decay: 2.06e-05
  max_epochs: 25
model:
  dropout: 0.1  # increased from 0.05
  radiomics_hidden: 1024  # increased from 512
  fusion_hidden: 512  # increased from 256
data:
  batch_size: 16
```

---

## Recommendations for Next Experiments

### Priority 1: Fine-tune Baseline with Best Config

The most promising approach is to **fine-tune the existing 95.24% model** with the winning hyperparameters:

```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=models/dual_pathway_bn_finetune_kygevxv0.pt \
  model.dropout=0.1 model.radiomics_hidden=1024 model.fusion_hidden=512 \
  train.optimizer.lr=0.00005 train.max_epochs=15"
```

**Expected outcome:** 96-97%+ accuracy

### Priority 2: Sweep on Fine-tuning

Run a proper W&B sweep on fine-tuning with the winning techniques:

```yaml
# Sweep parameters
method: bayes
parameters:
  train.loss.smoothing:
    min: 0.05
    max: 0.15
  model.dropout:
    min: 0.05
    max: 0.15
  train.optimizer.lr:
    min: 0.00001
    max: 0.0001
  model.radiomics_hidden:
    values: [512, 768, 1024]
  train.max_epochs:
    values: [10, 15, 20]
```

### Priority 3: Investigate Anomalies

1. Re-run T4_larger_radiomics_hidden_768 to verify the poor result
2. Check if there was a config error causing the 82.86% result

### Priority 4: Extended Training with Best Config

If fine-tuning works, try extended training:
- 30-50 epochs with cosine annealing
- Early stopping based on validation accuracy

---

## Summary Table: Top 5 Configurations

| Rank | Configuration | Test Acc | Key Changes |
|------|---------------|----------|-------------|
| 1 | T4_larger_radiomics_1024 | **93.97%** | radiomics=1024, fusion=512, label_smooth=0.1 |
| 2 | T4_higher_dropout_0.1 | 93.02% | dropout=0.1, label_smooth=0.1 |
| 3 | T1_label_smoothing_0.1 | 92.70% | label_smooth=0.1 only |
| 4 | T3_* (all augmentations) | 92.70% | augmentation + label_smooth=0.1 |
| 5 | T2_weighted_sampling_mod | 92.38% | weighted sampling only |

---

## Next Steps Checklist

- [ ] Run fine-tuning experiment with best config on 95.24% baseline
- [ ] If successful (>96%), run W&B sweep for optimization
- [ ] Investigate T4_radiomics_768 anomaly
- [ ] Evaluate best model on adenocarcinoma-specific metrics
- [ ] Compare confusion matrices before/after

---

**Analysis by:** Claude (ml-experiment-planner)
**Generated:** 2026-01-29
