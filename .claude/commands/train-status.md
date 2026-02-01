---
description: Show live ML training status (W&B runs, GPU, checkpoints)
allowed-tools: Bash, Read, Glob
---

# Training Status Dashboard

This command gathers live training status to provide context for ML-related questions.

## GPU Status
```
!nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "No GPU available or nvidia-smi not found"
```

## Recent W&B Runs
```
!uv run wandb status 2>/dev/null || echo "W&B not configured or no active runs"
```

## Latest Checkpoints
```
!find outputs/ -name "*.ckpt" -type f -mtime -7 2>/dev/null | head -10 || echo "No recent checkpoints found"
```

## Model Outputs Directory
```
!ls -lhtr outputs/ 2>/dev/null | tail -10 || echo "No outputs directory"
```

## Current Training Processes
```
!ps aux | grep -E "(train|python.*ct_scan)" | grep -v grep | head -5 || echo "No training processes running"
```

---

## Context for Your Question

Based on the status above, I can help you with:
- Analyzing recent training runs
- Debugging GPU memory issues
- Comparing checkpoint performance
- Planning next experiments

What would you like to know about your training status?
