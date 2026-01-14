# Team Collaboration Guide

## W&B Team Setup

This project uses **Weights & Biases** for experiment tracking with team collaboration.

### Team Information
- **Team Name**: `mathiashl-danmarks-tekniske-universitet-dtu`
- **Project**: `CT_Scan_MLOps`
- **Dashboard**: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps

---

## For Team Members: Getting Started

### 1. Accept Team Invitation

You should receive an email invitation to join the W&B team. Click the link to accept.

Alternatively:
1. Go to https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu
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

If you see your personal account instead, you need to switch teams or accept the invitation.

### 3. Login to W&B (First Time Only)

```bash
wandb login
```

Enter your API key from: https://wandb.ai/authorize

**Important:** This is a **one-time setup**. Your credentials are saved locally and you'll never need to login again on this machine (unless you logout or switch accounts).

After this, all training runs are **automatically attributed to you** - no verification needed for each run!

---

## Running Training

Once you're a team member, just run training normally:

```bash
# Basic training
invoke train

# With custom settings
python src/ct_scan_mlops/train.py train.max_epochs=10 model=resnet18

# Quick test (1 epoch)
python src/ct_scan_mlops/train.py train.max_epochs=1 data.batch_size=16
```

**No need to specify `--entity`** - it's already configured in `configs/config.yaml`.

---

## What Gets Logged to W&B

Every training run automatically logs:

| What | Example |
|------|---------|
| **Run Author** | Your W&B username |
| **Git Commit** | Current commit hash |
| **Hyperparameters** | All config values (model, training, data) |
| **Metrics** | Loss, accuracy (per batch and epoch) |
| **Learning Rate** | Per epoch |
| **Sample Images** | First batch of first epoch |
| **Training Curves** | Loss/accuracy plots |
| **Model Artifacts** | best_model.pt with metadata |
| **System Info** | GPU, CPU, OS |

---

## Viewing Runs

### Team Dashboard
https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps

### Your Runs
https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps?workspace=user-YOUR_USERNAME

### Compare Runs
1. Go to project dashboard
2. Select multiple runs (checkboxes)
3. Click "Compare"
4. View side-by-side metrics and hyperparameters

---

## Run Naming Convention

Runs are automatically named: `{experiment_name}_{model_name}`

Examples:
- `ct_scan_classifier_custom_cnn`
- `ct_scan_classifier_resnet18`

You can override the experiment name:
```bash
python src/ct_scan_mlops/train.py experiment_name=my_experiment
```

---

## Best Practices

### 1. Tag Your Runs

Add descriptive tags for easier filtering:

```bash
python src/ct_scan_mlops/train.py \
  wandb.tags=[experiment1,baseline,high-lr]
```

### 2. Use Descriptive Experiment Names

For major experiments, use meaningful names:

```bash
python src/ct_scan_mlops/train.py \
  experiment_name=resnet_pretrained_experiment
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

---

## FAQ

### Do I need to verify team membership for each run?

**No!** Once you've logged in with `wandb login` and joined the team, everything is automatic:
- Your credentials are stored locally in `~/.netrc`
- Every run automatically uses your credentials
- W&B automatically attributes runs to your username
- You never need to login again (unless you change computers or logout)

### How does W&B know who I am?

When you run training:
1. W&B reads your API key from `~/.netrc` (saved during `wandb login`)
2. Checks if you're a member of the entity in the config (`mathiashl-danmarks-tekniske-universitet-dtu`)
3. If yes → logs the run under your username
4. If no → shows permission error

### Do I need to specify my username?

**No!** W&B automatically knows your username from your login credentials. Just run:
```bash
invoke train
```

Your username will appear in the dashboard automatically.

### What if I use multiple computers?

You need to `wandb login` on each computer (one time per machine). After that, it's automatic on that machine.

---

## Troubleshooting

### "Permission denied" error

**Problem:** You're not in the team yet.

**Solution:**
1. Check team membership: `wandb team`
2. Accept invitation email
3. Contact admin if no invitation received

### Wrong team showing

**Problem:** `wandb team` shows your personal account.

**Solution:**
```bash
# Relogin and select team
wandb login --relogin

# Or explicitly set team
export WANDB_ENTITY=mathiashl-danmarks-tekniske-universitet-dtu
```

### Want to run without W&B

**For testing without logging:**
```bash
python src/ct_scan_mlops/train.py wandb.mode=disabled
```

---

## Team Roles

| Role | Permissions |
|------|-------------|
| **Admin** | Invite members, manage settings, delete runs |
| **Member** | Log runs, view all runs, create artifacts |
| **Viewer** | View runs only (read-only) |

Contact the admin for role changes.

---

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

---

## Questions?

- W&B Docs: https://docs.wandb.ai/
- Team Dashboard: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu
- GitHub Issues: [Link to your repo issues]
