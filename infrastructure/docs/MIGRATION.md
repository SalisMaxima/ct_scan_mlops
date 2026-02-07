# Terraform Migration Strategy

Strategy for migrating existing GCP infrastructure to Terraform management.

## Overview

**Current State:** GCP resources exist but are manually managed
**Goal:** Codify all infrastructure in Terraform without disrupting services
**Timeline:** 4-8 weeks (4 weeks Terraform, 4 weeks Monitoring)

## 📍 Current Phase Checkpoint

**Last Updated:** 2026-02-02
**Current Phase:** Phase 2 - Import Existing Resources
**Status:** Just started, awaiting user input for credentials

---

## Migration Phases

### Phase 1: Foundation (Week 1) ✅ COMPLETED

**Objective:** Set up Terraform infrastructure

**Tasks:**
- [x] Create infrastructure directory structure
- [x] Write all Terraform modules (9 modules complete)
- [x] Configure prod environment
- [x] Create dev and staging environments
- [x] Write comprehensive documentation
- [x] Fix all 7 critical issues from Gemini review
- [x] Apply 3 bonus improvements

**Deliverables:**
- ✅ All modules written and validated
- ✅ Backend configured (awaiting state bucket creation)
- ✅ Dev, staging, prod environments ready
- ✅ .gitignore created
- ✅ IAM roles explicitly configured
- ✅ Documentation complete (2,484 lines)

**Validation:**
```bash
✅ invoke terraform.validate --environment=prod  # Passed
✅ invoke quality.ruff                           # All checks passed
✅ Gemini architectural review                    # Grade: A
```

**Date Completed:** 2026-02-02

### Phase 2: Import (Weeks 2-3) - CRITICAL 🔄 IN PROGRESS

**Objective:** Import existing resources without changes

**⚠️ This is the most critical phase - allow 2 weeks**

**Status:** Just started - waiting for user to provide:
- Billing account ID
- GitHub repository name
- Alert email address
- Container image URL

**Import Order (dependencies matter):**
1. Storage buckets (no dependencies)
2. Artifact Registry (no dependencies)
3. IAM service accounts (no dependencies)
4. Workload Identity Pool (depends on service accounts)
5. Secret Manager secrets (no dependencies)
6. Firestore database (no dependencies)
7. Cloud Run service (depends on service accounts, buckets, secrets)

**For each resource:**
1. Run import command
2. Run `terraform plan` - expect changes
3. Iteratively fix "noisy diffs" by adjusting Terraform config
4. Repeat until `terraform plan` shows 0 changes

**Common "Noisy Diffs":**

| Resource | Issue | Fix |
|----------|-------|-----|
| GCS Bucket | Missing labels | Add `managed_by = "terraform"` label |
| GCS Bucket | Missing lifecycle rules | Keep lifecycle rules in Terraform |
| IAM Bindings | Order differences | Terraform reorders (safe to apply) |
| Cloud Run | Environment variable order | Terraform reorders (safe to apply) |
| Cloud Run | Volume mount differences | Adjust `gcs_volume_mount` config |
| Artifact Registry | Missing cleanup policies | Add cleanup policies |

**Critical Success Criterion:**
```bash
terraform plan
# Output: No changes. Your infrastructure matches the configuration.
```

**If plan shows changes after import:**
- Review each change carefully
- For additions: Determine if safe to add
- For modifications: Adjust Terraform config to match reality
- For deletions: **STOP** - investigate before proceeding

### Phase 3: Validation (Week 3)

**Objective:** Verify Terraform manages resources correctly

**Tasks:**
1. Make a safe test change (add a label)
2. Apply change
3. Verify in GCP console
4. Revert change
5. Verify `terraform plan` shows 0 changes again

**Test Change Example:**
```hcl
# In main.tf
locals {
  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
    test        = "true"  # Add this
  }
}
```

```bash
terraform plan  # Should show label additions
terraform apply
# Verify labels in GCP console
# Remove test label
terraform plan  # Should show label removals
terraform apply
```

**Success Criteria:**
- [ ] Test change applied successfully
- [ ] Test change reverted successfully
- [ ] No service disruption
- [ ] Terraform plan shows 0 changes after revert

### Phase 4: CI/CD Integration (Week 4)

**Objective:** Automate Terraform workflows

**Tasks:**
1. Create `.github/workflows/terraform.yml`
2. Configure OIDC for GitHub Actions (already in Terraform)
3. Test workflow in dev environment
4. Enable for prod with manual approval

**Workflow Features:**
- Auto-plan on PR
- Post plan as PR comment
- Auto-apply to dev on merge to main
- Manual approval for prod deployment

**Testing:**
1. Create test PR with Terraform change
2. Verify plan comment posted
3. Merge PR
4. Verify dev auto-apply works
5. Manually approve prod deployment
6. Verify prod deployment works

### Phase 5: Monitoring Foundation (Week 4 - Parallel with CI/CD)

**⚠️ Cannot start until Terraform Phase 3 complete**

**Objective:** Migrate feedback storage and deploy monitoring

**Tasks:**
1. Migrate SQLite feedback database to Firestore
2. Deploy notification channels
3. Enable P0 critical alerts only
4. Test alert delivery
5. Deploy monitoring API endpoints

**Firestore Migration:**
```python
# Script to migrate SQLite -> Firestore
# 1. Export SQLite data
# 2. Transform to Firestore format
# 3. Import to Firestore
# 4. Validate data integrity
# 5. Switch application to use Firestore
# 6. Monitor for errors
```

### Phase 6: Monitoring Shadow Mode (Weeks 5-6)

**Objective:** Tune P1/P2 alerts without notifications

**Tasks:**
1. Deploy P1/P2 alerts in shadow mode (`enable_p1_p2_alerts = false`)
2. Monitor alert firing in logs for 1 week
3. Analyze false positive rate
4. Adjust thresholds to achieve <2% false positive rate
5. Build dashboards

**Shadow Mode Verification:**
```bash
# Check alert logs
gcloud logging read "resource.type=monitoring.googleapis.com/alert_policy" \
  --limit=100 \
  --format=json

# Analyze firing frequency
# Target: <2% false positives
```

### Phase 7: Monitoring Production (Week 7)

**Objective:** Enable all alerts with notifications

**Tasks:**
1. Enable P1/P2 alerts (`enable_p1_p2_alerts = true`)
2. Deploy P2/P3 alerts
3. Test all notification channels
4. Conduct chaos testing
5. Validate runbooks with real incidents

**Chaos Testing:**
```bash
# Simulate failures
bash scripts/chaos_testing.sh

# Expected:
# - API downtime alert fires
# - High error rate alert fires
# - Memory exhaustion alert fires
# - All notifications delivered
```

### Phase 8: Finalization (Week 8)

**Objective:** Production hardening and documentation

**Tasks:**
1. Review all alert thresholds with prod data
2. Complete all runbooks with real incident examples
3. Train team on Terraform + monitoring workflows
4. Set up scheduled state backups
5. Document disaster recovery procedures

## Rollback Strategy

### If Import Fails

**Symptoms:**
- Terraform plan shows unexpected deletions
- Import command fails with errors
- Resources in inconsistent state

**Actions:**
1. **DO NOT run terraform apply**
2. Review specific resource causing issues
3. Manually fix resource in GCP if misconfigured
4. Re-attempt import with corrected config
5. If unrecoverable: Remove resource from Terraform, manage manually temporarily

### If Apply Breaks Service

**Symptoms:**
- Cloud Run service down
- API returning errors
- Resource access denied

**Actions:**
1. **Immediate:** Revert in GCP console (faster than Terraform)
   ```bash
   # Rollback Cloud Run to previous revision
   gcloud run services update-traffic ct-scan-api \
     --to-revisions=PREVIOUS_REVISION=100 \
     --region=europe-west1
   ```

2. **Root Cause:** Analyze Terraform plan that was applied
   ```bash
   # View applied plan
   terraform show
   ```

3. **Fix:** Correct Terraform configuration
4. **Re-apply:** Deploy fix via Terraform
5. **Post-mortem:** Document incident and prevention steps

### If State Corrupted

**Symptoms:**
- Terraform commands fail with state errors
- State file missing or corrupted
- Lock errors persist

**Actions:**
1. **Restore from version:**
   ```bash
   gsutil ls -la gs://dtu-mlops-terraform-state-482907/environments/prod/
   # Identify previous version
   gsutil cp gs://bucket/path#version ./terraform.tfstate
   ```

2. **Verify restoration:**
   ```bash
   terraform plan  # Should work now
   ```

3. **If restoration fails:** Rebuild state from scratch
   ```bash
   # Re-import all resources
   bash infrastructure/scripts/import-existing.sh
   ```

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Accidental resource deletion | Medium | Critical | Import first, never create from scratch |
| Service disruption during apply | Low | High | Test in dev, use canary deployments |
| State file corruption | Low | High | GCS versioning, daily backups |
| Import drift (noisy diffs) | High | Medium | Iterative fix, thorough review |
| Team unfamiliarity | Medium | Medium | Training, pair programming |
| False positive alerts | High | Low | Shadow mode tuning |

## Communication Plan

### Stakeholders
- **Development Team:** Daily standups, Slack updates
- **Operations Team:** Weekly sync, incident reviews
- **Management:** Weekly status reports

### Migration Windows
- **Dev Environment:** Anytime (non-production)
- **Staging:** Weekdays during business hours
- **Prod:** Maintenance windows (announce 48h ahead)

### Incident Response
1. **Detection:** Monitoring alerts, user reports
2. **Communication:** Post in #incidents Slack channel
3. **Resolution:** Follow runbooks, escalate if needed
4. **Post-mortem:** Document lessons learned

## Success Metrics

### Terraform Migration
- [ ] All resources in Terraform state
- [ ] `terraform plan` shows 0 changes
- [ ] Zero service disruptions during migration
- [ ] CI/CD pipeline functional
- [ ] Team trained and can operate independently

### Monitoring Implementation
- [ ] All P0 alerts configured and tested
- [ ] <2% false positive rate achieved
- [ ] All notification channels working
- [ ] Runbooks complete and validated
- [ ] Zero P0 incidents missed by alerting system (30-day measure)

## Timeline Summary

```
Week 1: Terraform Foundation
Week 2-3: Import Existing Resources (CRITICAL - allow 2 weeks)
Week 3: Validation & Testing
Week 4: CI/CD + Firestore Migration
Week 5-6: Monitoring Shadow Mode (P1/P2 tuning)
Week 7: Monitoring Production (all alerts enabled)
Week 8: Finalization & Training
```

## Post-Migration

### Ongoing Maintenance
- Review Terraform changes in PR reviews
- Run `terraform plan` weekly to catch drift
- Update modules when GCP provider updates
- Quarterly review of alert thresholds

### Continuous Improvement
- Add new resources to Terraform as they're created
- Refactor modules for better reusability
- Implement automated testing for Terraform code
- Expand monitoring coverage based on incidents

## Appendix: Pre-Migration Checklist

- [ ] Terraform installed (>= 1.5.0)
- [ ] GCP CLI authenticated
- [ ] State bucket created with versioning
- [ ] terraform.tfvars configured
- [ ] Team trained on basic Terraform commands
- [ ] Backup of current infrastructure documented
- [ ] Communication plan approved
- [ ] Maintenance window scheduled
- [ ] Rollback procedures tested in dev
- [ ] All required GCP APIs enabled
