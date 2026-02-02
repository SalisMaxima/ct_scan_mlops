---
description: Show project status (git, DVC, environment, data)
allowed-tools: Bash, Read
---

# Project Status Overview

Live status of the CT Scan MLOps project.

## Git Status
```
!git status --short --branch
```

## Recent Commits
```
!git log --oneline -5
```

## DVC Data Status
```
!dvc status 2>/dev/null || echo "DVC not configured or no tracked files"
```

## DVC Remote Status
```
!dvc status --remote 2>/dev/null | head -20 || echo "Could not check remote status"
```

## Data Directory Status
```
!ls -lh data/ 2>/dev/null || echo "No data directory"
```

## Environment Check
```
!python --version && uv --version 2>/dev/null
```

## Key Dependencies
```
!uv pip show torch pytorch-lightning hydra-core wandb 2>/dev/null | grep -E "^(Name|Version):" || echo "Could not check dependencies"
```

## Disk Usage (Project)
```
!du -sh . data/ outputs/ 2>/dev/null | head -5
```

---

## Quick Actions

Based on the status above, common next steps might be:
- `invoke dvc-pull` - Pull data from remote
- `invoke ruff` - Lint and format code
- `invoke test` - Run test suite
- `invoke train` - Start training

What would you like to do?
