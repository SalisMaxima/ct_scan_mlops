---
description: Show W&B sweep status and results
allowed-tools: Bash, Read
---

# W&B Sweep Status

Live status of hyperparameter sweeps.

## Active Sweeps
```
!uv run wandb sweep --list 2>/dev/null | head -20 || echo "No active sweeps or W&B not configured"
```

## Recent Sweep Runs (Last 24h)
```
!find outputs/ -name "wandb" -type d -mtime -1 2>/dev/null | head -10 || echo "No recent sweep outputs"
```

## Sweep Configuration Files
```
!ls -la configs/sweep*.yaml 2>/dev/null || echo "No sweep configs found"
```

## Current Sweep Config
```
!cat configs/sweep.yaml 2>/dev/null | head -30 || echo "No sweep.yaml found"
```

## Feature Files Available (for dual_pathway sweeps)
```
!ls -lh data/processed/*features*.pt 2>/dev/null || echo "No extracted features found - run: invoke extract-features"
```

---

## Sweep Commands

Common sweep operations:
- `invoke sweep` - Create a new W&B sweep
- `invoke sweep-agent <SWEEP_ID>` - Run a sweep agent
- `invoke prepare-sweep-features` - Extract features for dual_pathway sweeps

## Analyzing Results

To analyze sweep results, you can:
1. Check W&B dashboard for best runs
2. Use `/train-status` to see recent checkpoints
3. Run `invoke compare-baselines` to compare models

What would you like to know about your sweeps?
