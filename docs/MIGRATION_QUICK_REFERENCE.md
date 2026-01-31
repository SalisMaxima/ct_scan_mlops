# Dual Pathway Migration - Quick Reference Card

**Version**: 2.0 | **Last Updated**: 2026-01-31

---

## 🚀 Quick Start (New Users)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Extract radiomics features
invoke extract-features --features top_features

# 3. Train dual pathway model
invoke train model=dual_pathway_top_features

# 4. Analyze results
invoke analyze-errors --checkpoint outputs/checkpoints/best_model.ckpt
```

---

## ⚡ Common Commands Cheat Sheet

### Feature Extraction
```bash
invoke extract-features                          # All 50 features (~15 min)
invoke extract-features --features top_features  # Top 16 features (~5 min)
invoke prepare-sweep-features                    # Prepare all configs
```

### Training
```bash
# Dual pathway models
invoke train model=dual_pathway_top_features
invoke train model=dual_pathway

# Legacy models (still work)
invoke train model=cnn
invoke train model=resnet18
```

### Analysis
```bash
invoke analyze-errors --checkpoint <path>        # Error analysis + confusion matrix
invoke analyze-features --checkpoint <path>      # Feature importance
invoke compare-baselines \
  --baseline old.ckpt \
  --improved new.ckpt                            # Compare two models
```

### Sweeps
```bash
invoke sweep --sweep-config configs/sweeps/dual_pathway_sweep.yaml
invoke sweep-agent <SWEEP_ID>
invoke sweep-best --sweep-id <SWEEP_ID>
```

---

## 🔄 Migration Scenarios

### Scenario 1: "Keep using old models"
**No changes needed!** All old commands work as before:
```bash
invoke train model=cnn
invoke train model=resnet18
```

### Scenario 2: "Try dual pathway"
```bash
invoke extract-features --features top_features  # One-time setup
invoke train model=dual_pathway_top_features     # Train
invoke analyze-errors --checkpoint <path>        # Analyze
```

### Scenario 3: "Run hyperparameter sweep"
```bash
invoke extract-features --features top_features  # Prerequisites
invoke sweep --sweep-config configs/sweeps/dual_pathway_sweep.yaml
invoke sweep-agent <SWEEP_ID>
```

### Scenario 4: "Deploy to production"
```bash
invoke train model=dual_pathway_top_features
uv run python scripts/export_onnx.py \
  --checkpoint <path> --output model.onnx
uv run python scripts/benchmark_onnx.py --model model.onnx
```

### Scenario 5: "Run overnight experiments"
```bash
bash scripts/phase2_finetune_experiments.sh      # Sequential experiments
bash scripts/phase2_run_and_shutdown.sh          # With auto-shutdown
```

---

## 🐛 Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: features_*.pkl` | Features not extracted | `invoke extract-features --features top_features` |
| `size mismatch, expected 50, got 16` | Config mismatch | Match model to features: `top_features` → `dual_pathway_top_features` |
| `Metric 'test_acc' not found` | Old sweep config | Update sweep yaml: `metric.name: test_acc` |
| API fails with dual pathway | Features missing in API env | Copy features: `cp data/processed/features_*.pkl models/` |
| Empty analysis outputs | Test not run during training | Set `train.test_after_training=true` |

---

## 📊 Performance Comparison

| Model | Test Accuracy | Training Time (V100) | Inference Time |
|-------|---------------|----------------------|----------------|
| CNN | 91.2% | 1.5h | 8ms |
| ResNet18 | 92.5% | 2h | 8ms |
| Dual Pathway (top) | **95.87%** | 2.5h | 12ms |
| Dual Pathway (all) | 95.5% | 3h | 12ms |

---

## 🎯 Best Practices

### ✅ DO
- ✅ Start with `top_features` (faster, less overfitting)
- ✅ Extract features **before** training dual pathway models
- ✅ Use `invoke analyze-errors` to understand model behavior
- ✅ Run sweeps with `dual_pathway_sweep.yaml` for optimization
- ✅ Export to ONNX for production deployment

### ❌ DON'T
- ❌ Mix feature configs (top_features model needs top_features extraction)
- ❌ Skip feature extraction (dual pathway won't work)
- ❌ Ignore error analysis (59% of errors are adenocarcinoma ↔ squamous confusion)
- ❌ Use all features for initial experiments (slower, more overfitting risk)

---

## 📁 New File Structure

```
ct_scan_mlops/
├── src/ct_scan_mlops/
│   ├── features/          # NEW: Radiomics extraction (intensity, texture, shape, etc.)
│   ├── analysis/          # NEW: Error analysis, comparison, explainability
│   ├── losses.py          # NEW: Focal loss, label smoothing, etc.
│   └── monitoring/        # MOVED: Drift detection (was in root)
├── scripts/               # NEW: Experiment automation scripts
├── configs/
│   ├── features/          # NEW: Feature extraction configs
│   ├── model/
│   │   └── dual_pathway*.yaml  # NEW: Dual pathway configs
│   └── sweeps/
│       └── dual_pathway*.yaml  # NEW: Dual pathway sweeps
├── outputs/               # ENHANCED: All artifacts consolidated here
└── .claude/agents/        # NEW: AI automation agents
```

---

## 📚 Documentation Links

- **Full Migration Guide**: [MIGRATION_DUAL_PATHWAY.md](MIGRATION_DUAL_PATHWAY.md)
- **Experiment Plan**: [ExperimentPlan_Phase2.md](ExperimentPlan_Phase2.md)
- **Results Analysis**: [ExperimentAnalysis_20260129.md](ExperimentAnalysis_20260129.md)
- **Improvement Plan**: [AdenocarcinomaImprovementPlan.md](AdenocarcinomaImprovementPlan.md)
- **Project Structure**: [Structure.md](Structure.md)

---

## 🆘 Getting Help

**Stuck?** Check the full migration guide:
```bash
cat docs/MIGRATION_DUAL_PATHWAY.md | less
```

**Report Issues**: GitHub Issues with `[dual-pathway]` tag

**Share Results**: Post to W&B team project

---

## 🔄 Backward Compatibility

✅ **All v1.x features still work**
- Old models (CNN, ResNet18)
- Old configs
- Old checkpoints
- Old API deployments
- Old sweeps

⚠️ **Minor changes (non-breaking)**
- Sweep metric: `val_acc` → `test_acc`
- Output location: Root → `outputs/`
- Import paths: Root modules → `monitoring/`

---

## 📋 Pre-Flight Checklist

Before training dual pathway models:
- [ ] Environment activated: `source .venv/bin/activate`
- [ ] Features extracted: `invoke extract-features --features top_features`
- [ ] Data preprocessed: `invoke preprocess-data` (if not done)
- [ ] DVC data pulled: `invoke dvc-pull` (if using DVC)
- [ ] W&B logged in: `wandb login` (if using W&B)

Before deploying to production:
- [ ] Model trained and validated
- [ ] Exported to ONNX: `scripts/export_onnx.py`
- [ ] Benchmarked: `scripts/benchmark_onnx.py`
- [ ] API tested: `bash test_api.sh`
- [ ] Features available in deployment environment

---

**Print this page and keep it handy!** 🖨️

For detailed explanations, see: [MIGRATION_DUAL_PATHWAY.md](MIGRATION_DUAL_PATHWAY.md)
