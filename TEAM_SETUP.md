# Team Setup - Action Items

## For Project Admin (You)

### ✅ Step 1: Invite Team Members

**Go to:** https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/settings

**Actions:**
1. Click "Members" tab
2. Click "Invite Members" button
3. Enter team members' emails:
   - teammate1@email.com
   - teammate2@email.com
   - teammate3@email.com
   - etc.
4. Click "Send Invitations"

**Alternative:** Share this invite link (if available in team settings):
```
https://wandb.ai/join/mathiashl-danmarks-tekniske-universitet-dtu?code=INVITE_CODE
```

---

### ✅ Step 2: Share Repository Access

**Share with team:**
1. Clone instructions
2. This file (TEAM_SETUP.md)
3. Point them to [COLLABORATION.md](COLLABORATION.md)

**Message Template:**
```
Hi team,

I've invited you to our W&B team for the MLOps project.

1. Accept the W&B invitation email
2. Clone the repo: [git clone URL]
3. Follow setup in GetStarted.md
4. Read COLLABORATION.md for W&B usage

Once you're in the team, just run:
  invoke train

All runs will automatically appear in our shared dashboard:
https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps

Let me know if you have issues!
```

---

### ✅ Step 3: Test Your Setup

Before teammates join, verify it works for you:

```bash
# Check you're in the team
wandb team
# Expected: mathiashl-danmarks-tekniske-universitet-dtu

# Quick test run
python src/ct_scan_mlops/train.py train.max_epochs=1 data.batch_size=16

# Check W&B dashboard
# Go to: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps
# You should see your test run listed with your username
```

---

## For Team Members

**You will receive:**
1. Email invitation to W&B team
2. Repository access

**Follow these steps:**
1. Accept W&B invitation (check email)
2. Read [COLLABORATION.md](COLLABORATION.md)
3. Run `wandb team` to verify
4. Start training!

---

## Verification Checklist

Before everyone starts working, verify:

- [ ] Admin: All team members invited
- [ ] Admin: Test run appears in dashboard
- [ ] Team: All members accepted invitations
- [ ] Team: Everyone can see runs in dashboard
- [ ] Team: Each member's runs show their username

---

## Common Issues

### Team member can't see the project

**Cause:** Not accepted invitation yet

**Solution:**
1. Check spam folder for invitation email
2. Or go to: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu
3. Click "Request Access" if invitation lost

### "Permission denied" error

**Cause:** Not in team yet

**Solution:**
```bash
# Check current team
wandb team

# If wrong team, relogin
wandb login --relogin
```

### Runs going to personal account

**Cause:** Not using team entity

**Solution:**
- Make sure you pulled latest code (entity is in config)
- Check: `grep entity configs/config.yaml`
- Should show: `mathiashl-danmarks-tekniske-universitet-dtu`

---

## Team Workflow

Once everyone is set up:

1. **Pull latest code**
   ```bash
   git pull
   dvc pull
   ```

2. **Preprocess data** (if not done)
   ```bash
   invoke preprocess-data
   ```

3. **Run experiments**
   ```bash
   invoke train --args "model=resnet18 train.max_epochs=20"
   ```

4. **View results**
   - Dashboard: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps
   - Your runs: Filter by your username
   - Compare: Select runs and click "Compare"

5. **Share findings**
   - Copy run URL
   - Share in team chat/Discord/Slack
   - Add notes in W&B run page

---

## Support

- W&B Issues: Check [COLLABORATION.md](COLLABORATION.md) troubleshooting
- Code Issues: Create GitHub issue
- Questions: Ask in team chat

---

## Next Steps

After team is set up:

1. Everyone run a quick test to verify setup
2. Divide experiments among team members
3. Track all runs in shared dashboard
4. Compare results and iterate

Good luck with your MLOps project! 🚀
