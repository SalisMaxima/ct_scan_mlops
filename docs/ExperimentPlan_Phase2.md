# Adenocarcinoma Improvement Plan - Phase 2

**Date:** 2026-01-29
**Based on:** Phase 1 screening experiments (26 single runs)
**Goal:** Improve from 95.24% → >97% test accuracy

---

## Phase 1 Summary

From 26 screening experiments, we identified:

| Winning Technique | Improvement | Confidence |
|-------------------|-------------|------------|
| Label smoothing 0.1 | +2.86% | High |
| radiomics_hidden=1024, fusion=512 | +1.27% | Medium |
| Dropout 0.1 | +0.32% | Medium |

**Best from-scratch:** 93.97% (T4_larger_radiomics_hidden_1024)
**Current baseline:** 95.24% (fine-tuned model)

---

## Phase 2 Experiments

### Experiment 2.1: Fine-tune Baseline with Best Config

**Objective:** Apply winning hyperparameters to the fine-tuned baseline model.

**Hypothesis:** Combining the pre-trained weights (95.24%) with the best loss function and architecture should yield >96% accuracy.

**Command:**
```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=models/dual_pathway_bn_finetune_kygevxv0.pt \
  model.dropout=0.1 \
  model.radiomics_hidden=1024 model.fusion_hidden=512 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15"
```

**Success criteria:** Test accuracy >96%

---

### Experiment 2.2: Fine-tune with Label Smoothing Only

**Objective:** Isolate the effect of label smoothing on fine-tuning (without architecture changes).

**Rationale:** The architecture change (1024/512) might not be compatible with the pre-trained checkpoint weights.

**Command:**
```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=models/dual_pathway_bn_finetune_kygevxv0.pt \
  model.dropout=0.05 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15"
```

**Success criteria:** Test accuracy >95.5%

---

### Experiment 2.3: Fine-tune with Higher Dropout

**Objective:** Test if increased dropout helps reduce overfitting during fine-tuning.

**Command:**
```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=models/dual_pathway_bn_finetune_kygevxv0.pt \
  model.dropout=0.1 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15"
```

**Success criteria:** Test accuracy >95.5%

---

### Experiment 2.4: Investigate radiomics_hidden=768 Anomaly

**Objective:** Re-run the poorly performing 768 config to verify if it was a fluke or real.

**Command:**
```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  model.dropout=0.05 \
  model.radiomics_hidden=768 model.fusion_hidden=384 \
  train.optimizer.lr=0.000115 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=25"
```

**Expected:** Should be between 92-94% if Phase 1 result was anomalous.

---

### Experiment 2.5: Extended Fine-tuning (if 2.1-2.3 succeed)

**Objective:** Longer training with best fine-tuning config.

**Command:**
```bash
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=models/dual_pathway_bn_finetune_kygevxv0.pt \
  model.dropout=0.1 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00003 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=30 \
  train.scheduler.T_max=30"
```

**Success criteria:** Test accuracy >96.5%

---

## Experiment Script

```bash
#!/bin/bash
# scripts/phase2_finetune_experiments.sh

set -e
PROJECT_DIR="/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/ct_scan_mlops"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/phase2_finetune_$TIMESTAMP"
COOLDOWN=10
CHECKPOINT="models/dual_pathway_bn_finetune_kygevxv0.pt"

cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p "$LOG_DIR"

echo "Phase 2: Fine-tuning Experiments"
echo "================================"
echo "Baseline: 95.24%"
echo "Target: >97%"
echo ""

# Experiment 2.1: Full best config
echo "=== Experiment 2.1: Best Config Fine-tune ==="
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=$CHECKPOINT \
  model.dropout=0.1 \
  model.radiomics_hidden=1024 model.fusion_hidden=512 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15" 2>&1 | tee "$LOG_DIR/exp2.1_best_config.log"
sleep $COOLDOWN

# Experiment 2.2: Label smoothing only
echo "=== Experiment 2.2: Label Smoothing Only ==="
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=$CHECKPOINT \
  model.dropout=0.05 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15" 2>&1 | tee "$LOG_DIR/exp2.2_label_smoothing.log"
sleep $COOLDOWN

# Experiment 2.3: Label smoothing + higher dropout
echo "=== Experiment 2.3: Label Smoothing + Dropout 0.1 ==="
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=$CHECKPOINT \
  model.dropout=0.1 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00005 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=15" 2>&1 | tee "$LOG_DIR/exp2.3_dropout.log"
sleep $COOLDOWN

# Experiment 2.4: Investigate 768 anomaly (from scratch)
echo "=== Experiment 2.4: Investigate 768 Anomaly ==="
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  model.dropout=0.05 \
  model.radiomics_hidden=768 model.fusion_hidden=384 \
  train.optimizer.lr=0.000115 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=25" 2>&1 | tee "$LOG_DIR/exp2.4_768_rerun.log"
sleep $COOLDOWN

# Experiment 2.5: Extended fine-tuning
echo "=== Experiment 2.5: Extended Fine-tuning ==="
invoke train --args "model=dual_pathway_top_features features=top_features \
  train.loss.type=label_smoothing train.loss.smoothing=0.1 \
  train.checkpoint=$CHECKPOINT \
  model.dropout=0.1 \
  model.radiomics_hidden=512 model.fusion_hidden=256 \
  train.optimizer.lr=0.00003 \
  train.optimizer.weight_decay=2.06e-05 \
  data.batch_size=16 \
  train.max_epochs=30 \
  train.scheduler.T_max=30" 2>&1 | tee "$LOG_DIR/exp2.5_extended.log"

echo ""
echo "Phase 2 Complete"
echo "================"
echo "Results in: $LOG_DIR"
grep -h "test_acc" "$LOG_DIR"/*.log | grep "0\.[0-9]"
```

---

## Decision Tree

```
Phase 2.1-2.3 Results
        │
        ▼
   Any >96%?
   ┌───┴───┐
  Yes      No
   │        │
   ▼        ▼
Run 2.5   Check if checkpoint
Extended   loading works
   │        │
   ▼        ▼
>97%?    Fix checkpoint issue
┌──┴──┐   and re-run
Yes   No
│      │
▼      ▼
Done!  Run W&B sweep
       on fine-tuning
```

---

## Success Metrics

| Experiment | Minimum | Target | Stretch |
|------------|---------|--------|---------|
| 2.1 Best Config | >95.5% | >96.5% | >97% |
| 2.2 Label Smooth | >95.5% | >96% | >96.5% |
| 2.3 + Dropout | >95.5% | >96% | >96.5% |
| 2.5 Extended | >96% | >97% | >97.5% |

---

## Contingency: W&B Sweep (if needed)

If fine-tuning experiments don't reach >96.5%, run a proper sweep:

```yaml
# configs/sweep/finetune_sweep.yaml
program: src/ct_scan_mlops/train.py
method: bayes
metric:
  name: test_acc
  goal: maximize
parameters:
  train.loss.smoothing:
    distribution: uniform
    min: 0.05
    max: 0.2
  model.dropout:
    distribution: uniform
    min: 0.03
    max: 0.15
  train.optimizer.lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.0002
  train.max_epochs:
    values: [10, 15, 20, 25]
```

---

## Timeline

| Phase | Experiments | Est. Time |
|-------|-------------|-----------|
| 2.1-2.4 | 4 experiments | ~10 min |
| 2.5 | Extended training | ~3 min |
| Analysis | Review results | ~5 min |
| **Total** | | **~20 min** |

---

## Checklist

- [ ] Run experiments 2.1-2.4
- [ ] Analyze results
- [ ] If >96%: run experiment 2.5
- [ ] If >97%: evaluate on adenocarcinoma metrics
- [ ] If <96%: investigate checkpoint loading, consider sweep
- [ ] Document final results
- [ ] Update AdenocarcinomaImprovementPlan.md with findings

---

**Plan created:** 2026-01-29
**Author:** Claude (ml-experiment-planner)
