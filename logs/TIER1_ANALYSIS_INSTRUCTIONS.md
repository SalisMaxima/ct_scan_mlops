# Tier 1 Analysis Instructions

## When Experiments Complete

The analysis tools are ready! Here's what to do:

### 1. Check if experiments are done

```bash
./scripts/check_tier1_progress.sh
```

### 2. Run comprehensive analysis

```bash
uv run python scripts/analyze_tier1_results.py
```

This will:
- ✅ Fetch all 7 experiment results from W&B
- ✅ Compare test accuracies
- ✅ Identify the best loss function
- ✅ Calculate improvement over baseline
- ✅ Show top 3 performers
- ✅ Generate recommendations for next steps
- ✅ Save detailed report to `logs/tier1_analysis.txt`

### 3. Review the output

The analysis script will show:

```
🏆 BEST PERFORMING MODEL
📊 IMPROVEMENT OVER BASELINE
🥇 TOP 3 EXPERIMENTS
📋 ALL EXPERIMENTS COMPARISON
🎯 RECOMMENDATIONS
```

### 4. Next steps based on results

#### If improvement is >1.5% (Excellent)
```bash
# Proceed to Tier 2 with best loss function
# Edit scripts/adenocarcinoma_improvement_experiments.sh
# Update BEST_LOSS_T1 variable with winning loss function
# Or run full suite
```

#### If improvement is 0.8-1.5% (Good)
```bash
# Evaluate best model for confusion matrix
invoke evaluate --checkpoint outputs/ct_scan_classifier/*/checkpoints/best_model.ckpt

# If adenocarcinoma improved, proceed to Tier 2
```

#### If improvement is <0.8% (Modest)
```bash
# Investigate before proceeding
# Check training logs
# Analyze confusion patterns
# Consider alternative approaches
```

## Quick Commands Reference

```bash
# Check progress
./scripts/check_tier1_progress.sh

# Analyze results
uv run python scripts/analyze_tier1_results.py

# View detailed report
cat logs/tier1_analysis.txt

# Check W&B dashboard
open https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps

# Evaluate specific model
invoke evaluate --checkpoint outputs/ct_scan_classifier/<run_id>/checkpoints/best_model.ckpt

# View results summary
cat logs/tier1_20260129_152548/results_summary.txt
```

## Files Generated

- `logs/tier1_analysis.txt` - Detailed analysis report
- `logs/tier1_20260129_152548/results_summary.txt` - Quick summary
- `logs/tier1_20260129_152548/T1_*.log` - Individual experiment logs
- `logs/tier1_20260129_152548/durations.txt` - Timing data

## Expected Timeline

- **Started:** 3:25 PM
- **Estimated completion:** 3:40-3:45 PM
- **Total time:** ~15-20 minutes

Current status: 🔄 Running (check with `./scripts/check_tier1_progress.sh`)
