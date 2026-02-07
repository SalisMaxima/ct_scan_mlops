# Terraform Setup Guide

Complete guide for setting up and using Terraform Infrastructure as Code for CT Scan MLOps.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Configuration](#configuration)
4. [Importing Existing Resources](#importing-existing-resources)
5. [Daily Workflow](#daily-workflow)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
```bash
# Terraform >= 1.5.0
terraform --version

# Google Cloud SDK
gcloud --version

# uv (for invoke tasks)
uv --version
```

### Required Permissions
Your GCP user or service account needs:
- `roles/owner` or equivalent for initial setup
- `roles/storage.admin` for state bucket creation
- `roles/iam.serviceAccountAdmin` for service account management

### GCP Project Setup
```bash
# Set your project
export PROJECT_ID="dtu-mlops-data-482907"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  cloudbilling.googleapis.com \
  monitoring.googleapis.com
```

## Initial Setup

### 1. Create Terraform State Bucket (One-Time)

The state bucket must be created manually before running Terraform:

```bash
# Create state bucket
gsutil mb -l europe-west1 gs://dtu-mlops-terraform-state-482907

# Enable versioning for disaster recovery
gsutil versioning set on gs://dtu-mlops-terraform-state-482907

# Set uniform bucket-level access
gsutil uniformbucketlevelaccess set on gs://dtu-mlops-terraform-state-482907

# Optional: Enable encryption with Cloud KMS
# gsutil encryption set -k projects/PROJECT_ID/locations/LOCATION/keyRings/RING/cryptoKeys/KEY gs://BUCKET
```

### 2. Configure Variables

```bash
cd infrastructure/terraform/environments/prod

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
# Required values:
# - project_id
# - project_number (get with: gcloud projects describe PROJECT_ID --format="value(projectNumber)")
# - billing_account (get with: gcloud billing accounts list)
# - github_repository (format: owner/repo)
# - alert_email
# - container_image (Docker image URL)
```

### 3. Initialize Terraform

```bash
# Using invoke
invoke terraform.init --environment=prod

# OR directly
cd infrastructure/terraform/environments/prod
terraform init
```

## Configuration

### Environment Structure

```
infrastructure/terraform/
├── modules/              # Reusable modules
│   ├── storage/         # GCS buckets
│   ├── artifact-registry/
│   ├── cloud-run/
│   ├── secret-manager/
│   ├── iam/
│   ├── workload-identity/
│   ├── firestore/
│   ├── budget/
│   └── monitoring/
├── environments/
│   ├── dev/            # Development environment
│   ├── staging/        # Staging environment
│   └── prod/           # Production environment
└── shared/             # Project-level resources
```

### Key Configuration Files

**`terraform.tfvars`** (per environment, not committed)
```hcl
project_id     = "dtu-mlops-data-482907"
project_number = "123456789012"
alert_email    = "team@example.com"
# ... other required variables
```

**`main.tf`** (environment orchestration)
- Calls all modules
- Wires up dependencies
- Configures backend

**Module variables** can be overridden per environment.

## Importing Existing Resources

### Automated Import (Recommended)

Use the automated import script:

```bash
# Using invoke
invoke terraform.import-all --environment=prod

# OR directly
cd infrastructure/terraform/environments/prod
bash ../../scripts/import-existing.sh
```

The script will:
1. Check for existing resources
2. Import them into Terraform state
3. Run `terraform plan` to verify
4. Report any differences

### Manual Import (if needed)

For specific resources:

```bash
# Using invoke
invoke terraform.import-resource \
  --resource="module.storage.google_storage_bucket.dvc" \
  --id="dtu-mlops-dvc-storage-482907" \
  --environment=prod

# OR directly
terraform import "module.storage.google_storage_bucket.dvc" "dtu-mlops-dvc-storage-482907"
```

### Post-Import Verification

After import, run:

```bash
terraform plan
```

**Expected results:**
- **First time:** Plan will show additions for IAM bindings, labels, lifecycle policies
- **After apply:** Plan should show 0 changes

If plan shows unexpected **deletions**, DO NOT APPLY - investigate first.

### Fixing "Noisy Diffs"

Common issues after import:

1. **IAM bindings order:** Terraform may reorder existing bindings (safe)
2. **Missing labels:** Terraform adds `managed_by=terraform` (safe)
3. **Lifecycle policies:** Terraform adds retention policies (safe)
4. **API defaults:** Some defaults differ between API and Terraform

Review each change carefully before applying.

## Daily Workflow

### 1. Planning Changes

```bash
# Check what would change
invoke terraform.plan --environment=prod

# View specific output
invoke terraform.output --environment=prod --output-name=cloud_run_url
```

### 2. Applying Changes

```bash
# Apply with interactive approval
invoke terraform.apply --environment=prod

# Auto-approve (use with caution)
invoke terraform.apply --environment=prod --auto-approve
```

### 3. Viewing State

```bash
# List all resources
invoke terraform.state-list --environment=prod

# Show specific resource
invoke terraform.state-show \
  --resource="module.cloud_run.google_cloud_run_v2_service.api" \
  --environment=prod
```

### 4. Formatting Code

```bash
# Format all .tf files
invoke terraform.format
```

### 5. Validation

```bash
# Validate configuration
invoke terraform.validate --environment=prod

# Run all checks (validate + plan)
invoke terraform.check --environment=prod
```

## Common Operations

### Adding a New Secret

1. Create secret placeholder in Terraform:
```hcl
# In main.tf (or module)
module "secret_manager" {
  # ...
  generic_secrets = {
    "new-secret" = {
      accessors = ["serviceAccount:my-sa@project.iam.gserviceaccount.com"]
    }
  }
}
```

2. Apply Terraform:
```bash
terraform apply
```

3. Add secret value (NOT in Terraform):
```bash
echo -n "secret-value" | gcloud secrets versions add new-secret --data-file=-
```

### Updating Cloud Run Service

1. Update variables in `terraform.tfvars`:
```hcl
container_image = "europe-west1-docker.pkg.dev/.../api:v2.0"
```

2. Apply:
```bash
terraform apply
```

### Adding a New Alert

1. Edit `monitoring/main.tf`:
```hcl
resource "google_monitoring_alert_policy" "new_alert" {
  # ... configuration
}
```

2. Apply:
```bash
terraform apply
```

### Scaling Resources

```hcl
# Update in terraform.tfvars
cloud_run_max_instances = 20
memory_limit           = "8Gi"
```

## Troubleshooting

### State Lock Errors

If state is locked (concurrent apply):

```bash
# View lock info
gsutil cat gs://dtu-mlops-terraform-state-482907/environments/prod/default.tflock

# Force unlock (use with caution)
terraform force-unlock LOCK_ID
```

### State Drift

If resources were modified outside Terraform:

```bash
# Refresh state to match reality
terraform refresh

# OR import the changed resource again
terraform import <resource> <id>
```

### Plan Shows Unexpected Changes

1. **Check recent GCP changes:**
   ```bash
   gcloud logging read "protoPayload.methodName=~'.*update.*'" --limit=20
   ```

2. **Compare Terraform state with actual resource:**
   ```bash
   # View Terraform state
   terraform state show "module.cloud_run.google_cloud_run_v2_service.api"

   # View actual resource
   gcloud run services describe ct-scan-api --region=europe-west1 --format=json
   ```

3. **Re-import if necessary:**
   ```bash
   terraform import "module.cloud_run.google_cloud_run_v2_service.api" \
     "projects/dtu-mlops-data-482907/locations/europe-west1/services/ct-scan-api"
   ```

### Backend Initialization Errors

If backend config fails:

```bash
# Verify bucket exists
gsutil ls gs://dtu-mlops-terraform-state-482907

# Verify permissions
gsutil iam get gs://dtu-mlops-terraform-state-482907

# Re-initialize
rm -rf .terraform
terraform init
```

### Module Not Found Errors

```bash
# Ensure you're in the correct directory
pwd  # Should be infrastructure/terraform/environments/prod

# Re-initialize to fetch modules
terraform init
```

## Best Practices

### Do's ✅
- Always run `terraform plan` before `apply`
- Commit `terraform.tfvars` to private repo or use encrypted secrets
- Use separate state files per environment (different directories)
- Import existing resources before managing them
- Test changes in dev environment first
- Use meaningful commit messages for Terraform changes

### Don'ts ❌
- Never manually edit Terraform state files
- Never hardcode secrets in `.tf` files
- Never run `terraform apply` without reviewing plan
- Never use `--auto-approve` in prod without thorough testing
- Never delete state bucket or state files
- Never run concurrent `terraform apply` commands

## Getting Help

### Resources
- [Terraform GCP Provider Docs](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Google Cloud Terraform Samples](https://cloud.google.com/docs/terraform)
- Project documentation: `infrastructure/docs/`

### Team Support
- Check runbooks: `infrastructure/terraform/modules/monitoring/runbooks/`
- Ask in team chat
- Review MIGRATION.md for migration strategies
- Check RUNBOOK.md for operations guide

## Next Steps

After successful setup:
1. ✅ Review [MIGRATION.md](./MIGRATION.md) for production migration strategy
2. ✅ Configure CI/CD with [.github/workflows/terraform.yml](../../.github/workflows/terraform.yml)
3. ✅ Set up monitoring alerts (already configured, verify email delivery)
4. ✅ Schedule regular state backups
5. ✅ Train team on Terraform workflows
