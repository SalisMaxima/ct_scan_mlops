# Plan: Fix Terraform Phase 2 Import Issues

## Summary

Fix 7 blocking issues preventing `terraform plan` from reaching "0 changes" state, ordered by risk severity.

---

## Issue 1 (CRITICAL): Models Bucket Being Destroyed

**Root cause**: The existing bucket `dtu-mlops-data-482907_cloudbuild` is a GCP default Cloud Build bucket, likely in `US` multi-region. Terraform config uses `location = var.region` (europe-west1). GCS bucket location is immutable — Terraform forces destroy + recreate on location mismatch.

**Fix**: Check actual bucket location, then update the storage module to accept per-bucket location overrides.

**Files**:
- `infrastructure/terraform/modules/storage/main.tf` (line 90-116): Add optional `location` override for models bucket
- `infrastructure/terraform/modules/storage/variables.tf`: Add `models_bucket_location` variable
- `infrastructure/terraform/environments/prod/main.tf` (line 37-62): Pass actual bucket location

**Steps**:
1. Run: `gsutil ls -L -b gs://dtu-mlops-data-482907_cloudbuild | grep Location` to get actual location
2. Add `models_bucket_location` variable with default `null` to storage module
3. Use `coalesce(var.models_bucket_location, var.region)` for the models bucket location
4. Pass the actual location in prod/main.tf
5. Re-import: `terraform import module.storage.google_storage_bucket.models dtu-mlops-data-482907_cloudbuild`

---

## Issue 2 (CRITICAL): Cloud Run Configuration Drift

**Root cause**: The existing Cloud Run service has completely different configuration from what Terraform defines. Applying would overwrite all environment variables and break the service.

**Actual service config** (from user's gcloud output):
- Env vars: CONFIG_PATH, DEPLOYMENT_ID, FEATURES_UPDATED, FEATURE_METADATA_PATH, MODEL_PATH, RELOAD_TIME
- Service account: `777769481436-compute@developer.gserviceaccount.com` (default compute SA)
- Volume mount name: `gcs-bucket` (not `models`)
- Min instances: 0, Max instances: 20
- Concurrency: 80

**Terraform config** (mismatched):
- Env vars: PROJECT_ID, REGION (completely different!)
- Service account: `cloud-run@...` (new SA)
- Volume mount name: `models`
- Min instances: 1, Max instances: 10

**Fix**: Update `prod/main.tf` Cloud Run module configuration to match the existing service EXACTLY, then iteratively align in future phases.

**Files**:
- `infrastructure/terraform/environments/prod/main.tf` (lines 190-228): Update Cloud Run module params to match reality
- `infrastructure/terraform/modules/cloud-run/main.tf`: May need to add `concurrency` variable, startup probe support
- `infrastructure/terraform/modules/cloud-run/variables.tf`: Add missing variables

**Steps**:
1. Update `environment_variables` to match actual env vars
2. Change `service_account_email` to the actual compute SA (or remove and let Cloud Run use default)
3. Update min_instances=0, max_instances=20
4. Update volume mount name to `gcs-bucket`
5. Re-import Cloud Run service after config changes

---

## Issue 3 (HIGH): Artifact Registry Cleanup Policy Format

**Root cause**: `older_than` field on line 38 of `artifact-registry/main.tf` uses `"${var.image_retention_days}d"` (e.g. "90d") but the GCP API requires seconds with 's' suffix (e.g. "7776000s").

**Fix**:
- **File**: `infrastructure/terraform/modules/artifact-registry/main.tf` line 38
- Change: `older_than = "${var.image_retention_days}d"` → `older_than = "${var.image_retention_days * 86400}s"`

---

## Issue 4 (HIGH): Terraform State Bucket Conflict

**Root cause**: We created the state bucket manually before `terraform init`. The storage module then tries to create it again (409 conflict).

**Fix**: Import the existing bucket into Terraform state.

**Step**: `terraform import module.storage.google_storage_bucket.terraform_state dtu-mlops-terraform-state-482907`

---

## Issue 5 (HIGH): Missing Monitoring Runbook Files

**Root cause**: The monitoring module uses `file()` to load 6 runbook markdown files that don't exist. `terraform plan` will fail.

**Fix**: Create the 6 runbook files in `infrastructure/terraform/modules/monitoring/runbooks/`:
- `api-downtime.md`
- `high-error-rate.md`
- `memory-exhaustion.md`
- `crash-loop.md`
- `high-latency.md`
- `elevated-error-rate.md`

Each file should contain basic incident response steps (1-2 paragraphs).

---

## Issue 6 (MEDIUM): Firestore API Not Enabled

**Root cause**: Cloud Firestore API has never been used in the project.

**Fix**:
```bash
gcloud services enable firestore.googleapis.com --project=dtu-mlops-data-482907
```

---

## Issue 7 (MEDIUM): Budget API Quota Project

**Root cause**: Billing Budgets API requires a quota project set when using Application Default Credentials.

**Fix**:
```bash
gcloud auth application-default set-quota-project dtu-mlops-data-482907
```

---

## Execution Order

1. **Enable APIs** (Issues 6, 7) — prerequisite, no code changes
2. **Fix artifact registry format** (Issue 3) — simple one-line fix
3. **Create runbook files** (Issue 5) — create 6 placeholder files
4. **Fix models bucket location** (Issue 1) — check location, update module + config
5. **Import state bucket** (Issue 4) — single terraform import command
6. **Fix Cloud Run config drift** (Issue 2) — most complex, update module + config
7. **Run `terraform plan`** — verify approaching 0 changes
8. **Apply safe changes** — labels, lifecycle rules, IAM bindings
9. **Iterate** until `terraform plan` shows "No changes"

---

## Critical Files to Modify

| File | Changes |
|------|---------|
| `infrastructure/terraform/modules/artifact-registry/main.tf` | Fix `older_than` format (line 38) |
| `infrastructure/terraform/modules/storage/main.tf` | Add per-bucket location override for models bucket |
| `infrastructure/terraform/modules/storage/variables.tf` | Add `models_bucket_location` variable |
| `infrastructure/terraform/modules/cloud-run/main.tf` | Add concurrency, startup probe support |
| `infrastructure/terraform/modules/cloud-run/variables.tf` | Add missing variables |
| `infrastructure/terraform/environments/prod/main.tf` | Update Cloud Run config to match reality, pass models bucket location |
| `infrastructure/terraform/modules/monitoring/runbooks/*.md` | Create 6 runbook files |

---

## Verification

After all fixes:
1. `terraform plan` — confirm 0 changes (or only safe additions)
2. `terraform state list` — verify all imported resources are present
3. `gcloud run services describe ct-scan-api --region=europe-west1` — verify service unchanged
4. `gsutil ls gs://dtu-mlops-data-482907_cloudbuild` — verify bucket still intact
5. Check Cloud Run service URL is still responding: `curl https://ct-scan-api-777769481436.europe-west1.run.app/health`
