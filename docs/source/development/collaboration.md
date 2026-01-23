# Team Collaboration

Guide to using Weights & Biases for team experiment tracking.

## W&B Team Setup

This project uses **Weights & Biases** for experiment tracking with team collaboration.

### Team Information

- **Team Name**: `mathiashl-danmarks-tekniske-universitet-dtu`
- **Project**: `CT_Scan_MLOps`
- **Dashboard**: [W&B Dashboard](https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps)

## Getting Started

### 1. Accept Team Invitation

You should receive an email invitation to join the W&B team. Click the link to accept.

Alternatively:

1. Go to the [team page](https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu)
2. Request access or contact the team admin

### 2. Verify Team Membership

After joining, verify you're in the team:

```bash
wandb team
```

**Expected output:**
```
mathiashl-danmarks-tekniske-universitet-dtu
```

### 3. Login to W&B (First Time Only)

```bash
wandb login
```

Enter your API key from: [https://wandb.ai/authorize](https://wandb.ai/authorize)

!!! note "One-time setup"
    Your credentials are saved locally. You'll never need to login again on this machine.

## Running Training

Once you're a team member, just run training normally:

```bash
# Basic training
invoke train

# With custom settings
invoke train --args "train.max_epochs=10 model=resnet18"

# Quick test (1 epoch)
invoke train --args "train.max_epochs=1 data.batch_size=16"
```

**No need to specify `--entity`** - it's already configured.

## What Gets Logged

Every training run automatically logs:

| What | Example |
|------|---------|
| **Run Author** | Your W&B username |
| **Git Commit** | Current commit hash |
| **Hyperparameters** | All config values |
| **Metrics** | Loss, accuracy (per batch and epoch) |
| **Learning Rate** | Per epoch |
| **Sample Images** | First batch of first epoch |
| **Training Curves** | Loss/accuracy plots |
| **Model Artifacts** | best_model.pt with metadata |
| **System Info** | GPU, CPU, OS |

## Viewing Runs

### Team Dashboard

[https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps](https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps)

### Your Runs

```
https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps?workspace=user-YOUR_USERNAME
```

### Compare Runs

1. Go to project dashboard
2. Select multiple runs (checkboxes)
3. Click "Compare"
4. View side-by-side metrics and hyperparameters

## Best Practices

### 1. Tag Your Runs

Add descriptive tags for easier filtering:

```bash
invoke train --args "wandb.tags=[experiment1,baseline,high-lr]"
```

### 2. Use Descriptive Experiment Names

```bash
invoke train --args "experiment_name=resnet_pretrained_experiment"
```

### 3. Log Important Notes

After a run, add notes in W&B:

1. Open the run in dashboard
2. Click "Overview" tab
3. Add notes about what you tried, why, results

### 4. Share Interesting Runs

Copy run URL and share with team:
```
https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps/runs/abc123
```

## Disable W&B Logging

For local testing without logging:

```bash
invoke train --args "wandb.mode=disabled"
```

## Troubleshooting

### "Permission denied" error

You're not in the team yet.

**Solution:**

1. Check team membership: `wandb team`
2. Accept invitation email
3. Contact admin if no invitation received

### Wrong team showing

`wandb team` shows your personal account.

**Solution:**

```bash
# Relogin and select team
wandb login --relogin

# Or explicitly set team
export WANDB_ENTITY=mathiashl-danmarks-tekniske-universitet-dtu
```

## Useful Commands

```bash
# Check team
wandb team

# Check login status
wandb status

# View local runs (not synced)
wandb sync --sync-all

# Pull artifact (model)
wandb artifact get mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps/ct_scan_classifier_model:latest

# Login with different account
wandb login --relogin
```

## Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [Team Dashboard](https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu)
- [GitHub Repository](https://github.com/SalisMaxima/ct_scan_mlops)
