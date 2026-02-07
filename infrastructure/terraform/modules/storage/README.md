# Storage Module

Manages GCS buckets for the CT Scan MLOps project.

## Buckets Created

1. **Terraform State Bucket** - Stores Terraform state with versioning and lifecycle policies
2. **DVC Bucket** - Stores dataset versions with 1-year retention
3. **Models Bucket** - Stores model artifacts with 6-month retention
4. **Drift Logs Bucket** - Stores drift detection logs with 90-day retention

## Features

- Versioning enabled on all buckets
- Lifecycle policies for automatic cleanup
- IAM bindings for service account access
- Uniform bucket-level access enabled
- Optional customer-managed encryption (KMS)

## Usage

```hcl
module "storage" {
  source = "./modules/storage"

  project_id                  = "dtu-mlops-data-482907"
  region                      = "europe-west1"
  terraform_state_bucket_name = "dtu-mlops-terraform-state-482907"
  dvc_bucket_name             = "dtu-mlops-dvc-storage-482907"
  models_bucket_name          = "dtu-mlops-data-482907_cloudbuild"
  drift_logs_bucket_name      = "ct-scan-drift-logs-482907"

  dvc_bucket_admins = [
    "serviceAccount:github-actions@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  models_bucket_admins = [
    "serviceAccount:cloud-run@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  drift_logs_writers = [
    "serviceAccount:drift-detection@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
  }
}
```

## Import Existing Buckets

To import existing buckets into Terraform state:

```bash
terraform import module.storage.google_storage_bucket.dvc dtu-mlops-dvc-storage-482907
terraform import module.storage.google_storage_bucket.models dtu-mlops-data-482907_cloudbuild
```

## Notes

- `force_destroy = false` on all buckets (state, DVC, models, drift logs) to prevent accidental deletion
- Lifecycle rules prevent unbounded storage costs
- Versioning provides disaster recovery capability
