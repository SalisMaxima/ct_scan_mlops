# Phase 4 Deployment Guide

**Status:** Code complete, deploy pending
**Date:** 2026-02-06

---

## Prerequisites

- `gcloud` CLI authenticated with project owner/editor access
- Terraform installed (>= 1.5.0)
- Docker (for local image build) or GitHub Actions (for CI build)
- `terraform.tfvars` in place at `infrastructure/terraform/environments/prod/`

---

## Step 1: Apply IAM Expansion

This grants the github-actions service account the 7 new roles needed for Terraform CI/CD to manage all 52 resources.

```bash
cd infrastructure/terraform/environments/prod
terraform init
terraform plan
```

Review the plan output. You should see **7 new IAM bindings** being added (one per new role). No resources should be destroyed.

Expected changes:
- `google_project_iam_member` for `roles/secretmanager.admin`
- `google_project_iam_member` for `roles/monitoring.admin`
- `google_project_iam_member` for `roles/datastore.owner`
- `google_project_iam_member` for `roles/iam.serviceAccountAdmin`
- `google_project_iam_member` for `roles/resourcemanager.projectIamAdmin`
- `google_project_iam_member` for `roles/pubsub.admin`
- `google_project_iam_member` for `roles/billing.user`
- Cloud Run env var updates (5 new vars)

If the plan looks correct:

```bash
terraform apply
```

**Verify:**

```bash
terraform plan
# Expected: "No changes"
```

---

## Step 2: Commit and Push

```bash
git add \
  .github/workflows/terraform.yml \
  .github/workflows/drift_detection.yml \
  src/ct_scan_mlops/feedback_store.py \
  src/ct_scan_mlops/api.py \
  infrastructure/terraform/environments/prod/main.tf \
  pyproject.toml \
  uv.lock \
  PHASE_CHECKPOINT.md \
  SESSION_RESUME.md

git commit -m "feat: Phase 4 — Terraform CI/CD, Firestore migration, drift detection

- Add Terraform CI/CD workflow (plan on PR, apply on merge)
- Add drift detection workflow (every 6 hours)
- Migrate SQLite feedback to Firestore with SQLite fallback
- Mount drift API at /monitoring in main API
- Expand github-actions SA with 7 new IAM roles
- Add 5 new Cloud Run env vars for Firestore and drift paths"

git push origin feature/enhanced-frontend-api
```

---

## Step 3: Deploy Updated API to Cloud Run

### Option A: Trigger the existing deploy workflow

Go to **GitHub Actions > Deploy API to Cloud Run** and click **Run workflow**.

This uses `.github/workflows/deploy_cloudrun.yml` which builds the Docker image, pushes to Artifact Registry, and deploys to Cloud Run.

> **Note:** The existing deploy workflow sets env vars via `--set-env-vars` in the gcloud command. You may need to add the 5 new env vars to that command, or rely on Terraform to reconcile them after merge. See Option B for a manual approach that uses Terraform-managed config.

### Option B: Manual deploy via gcloud

```bash
# Build and push the image
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
IMAGE_URI="europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops/ct-scan-api:${IMAGE_TAG}"

docker build -f dockerfiles/api.cloudrun.dockerfile -t "$IMAGE_URI" .

gcloud auth configure-docker europe-west1-docker.pkg.dev --quiet
docker push "$IMAGE_URI"

# Update Cloud Run (Terraform manages env vars, so just update the image)
gcloud run deploy ct-scan-api \
  --image "$IMAGE_URI" \
  --region europe-west1 \
  --platform managed
```

### Option C: Update image reference in Terraform and apply

Edit `infrastructure/terraform/environments/prod/terraform.tfvars` and update `container_image` to the new tag, then:

```bash
cd infrastructure/terraform/environments/prod
terraform apply
```

This is the most consistent approach since Terraform already manages the Cloud Run service config including env vars.

---

## Step 4: Verify Endpoints

Get the service URL:

```bash
SERVICE_URL=$(gcloud run services describe ct-scan-api \
  --region=europe-west1 \
  --format='value(status.url)')
echo "$SERVICE_URL"
```

### 4a. Health check

```bash
curl -s "$SERVICE_URL/health" | python3 -m json.tool
```

Expected: `"status": "healthy"`, `"model_loaded": true`

### 4b. Feedback submission

```bash
# Use any CT scan image for testing
curl -s -X POST "$SERVICE_URL/feedback" \
  -F "file=@path/to/test_image.png" \
  -F "predicted_class=normal" \
  -F "predicted_confidence=0.95" \
  -F "is_correct=true" | python3 -m json.tool
```

Expected: `"logged_to_db": true`, `"feedback_id"` should be a Firestore document ID (not null).

### 4c. Feedback stats

```bash
curl -s "$SERVICE_URL/feedback/stats" | python3 -m json.tool
```

Expected: Returns `total_feedback`, `accuracy`, `class_distribution`, etc.

### 4d. Drift API health

```bash
curl -s "$SERVICE_URL/monitoring/health" | python3 -m json.tool
```

Expected: Returns `ok`, `reference_path`, `current_path`, `report_path` with `/gcs/drift/...` paths.

### 4e. Drift detection (only works if reference + current CSVs exist)

```bash
curl -s "$SERVICE_URL/monitoring/drift" | python3 -m json.tool
```

Expected: Returns drift analysis or a 400 error if CSV files don't exist yet (which is fine — they'll populate as predictions are made).

---

## Step 5: Verify Firestore

Open the [Firebase Console](https://console.firebase.google.com/project/dtu-mlops-data-482907/firestore) or GCP Console Firestore section.

Check that the `feedback` collection exists and contains the document you submitted in Step 4b.

---

## Step 6: Verify Terraform CI/CD

### 6a. Create a test PR

Create a branch with a trivial Terraform change:

```bash
git checkout -b test/terraform-cicd
```

Make a small change (e.g., add a comment to `main.tf`):

```bash
echo '# Terraform CI/CD test' >> infrastructure/terraform/environments/prod/main.tf
git add infrastructure/terraform/environments/prod/main.tf
git commit -m "test: Verify Terraform CI/CD workflow"
git push origin test/terraform-cicd
```

Open a PR from `test/terraform-cicd` to `master`.

### 6b. Check the workflow

1. Go to **GitHub Actions** and verify the `Terraform CI/CD` workflow runs
2. The `terraform-check` job should pass (format + validate)
3. The `terraform-plan` job should post a PR comment with the plan output
4. Review the plan comment — it should show minimal or no changes

### 6c. Clean up

Close the PR without merging and delete the test branch:

```bash
git checkout feature/enhanced-frontend-api
git branch -d test/terraform-cicd
git push origin --delete test/terraform-cicd
```

---

## Step 7: Verify Drift Detection

### 7a. Manual trigger

Go to **GitHub Actions > Drift Detection** and click **Run workflow**.

The workflow will:
1. Auth to GCP via Workload Identity
2. Check API health
3. Call `/monitoring/drift?write_html=true`
4. If drift data exists, analyze results
5. Upload drift report as an artifact

> **Note:** The first run may fail at the drift step if no reference/current CSVs exist yet. This is expected. Once predictions flow through the API, the current.csv populates automatically. You'll need to manually create a reference.csv baseline.

### 7b. Create reference data (if needed)

Once enough predictions have been made through the API, copy the current data as the reference baseline:

```bash
gsutil cp \
  gs://dtu-mlops-data-482907_cloudbuild/drift/current.csv \
  gs://dtu-mlops-data-482907_cloudbuild/drift/reference.csv
```

---

## Troubleshooting

### "Feedback store not available" on /feedback/stats

The Firestore client failed to initialize. Check Cloud Run logs:

```bash
gcloud run services logs read ct-scan-api --region=europe-west1 --limit=50
```

Common causes:
- `GCP_PROJECT_ID` env var not set — verify with `terraform plan`
- Service account lacks `roles/datastore.user` — should already be configured
- Firestore API not enabled — run `gcloud services enable firestore.googleapis.com`

### Terraform CI/CD workflow fails on plan

- Check that the `GCP_WORKLOAD_IDENTITY_PROVIDER` and `GCP_SERVICE_ACCOUNT` GitHub vars are set
- Verify the github-actions SA has the new roles (Step 1)
- Check workflow logs for specific error messages

### Drift detection can't reach /monitoring/drift

- Verify the drift API is mounted by checking `curl $SERVICE_URL/monitoring/health`
- If 404, the `evidently` package may not be in the Docker image — check the Dockerfile includes it

---

## Checklist

- [ ] `terraform apply` succeeds, `terraform plan` = 0 changes
- [ ] Code committed and pushed
- [ ] API deployed to Cloud Run with new image
- [ ] `curl /health` returns 200
- [ ] `curl /feedback` with test image returns `feedback_id` (not null)
- [ ] `curl /feedback/stats` returns aggregated data
- [ ] `curl /monitoring/health` returns drift config
- [ ] Firestore document visible in console
- [ ] Test PR triggers Terraform plan comment
- [ ] Drift detection workflow runs (manual trigger)
- [ ] Update PHASE_CHECKPOINT.md to mark Phase 4 COMPLETE
