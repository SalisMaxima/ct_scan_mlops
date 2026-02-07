# Artifact Registry Module

Manages Docker container registry for CT Scan MLOps.

## Features

- Docker repository for container images
- Cleanup policies to prevent unbounded storage costs
- IAM bindings for service account access
- Cloud Build integration

## Repository URL

Images are stored at:
```
europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops
```

## Cleanup Policies

1. **Delete old images**: Images older than 90 days are automatically deleted
2. **Keep minimum versions**: Always keep at least 5 versions regardless of age

## Usage

```hcl
module "artifact_registry" {
  source = "./modules/artifact-registry"

  project_id     = "dtu-mlops-data-482907"
  project_number = "123456789012"  # Replace with actual project number
  region         = "europe-west1"
  repository_id  = "ct-scan-mlops"

  docker_readers = [
    "serviceAccount:cloud-run@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  docker_writers = [
    "serviceAccount:github-actions@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  enable_cloud_build_access = true
  image_retention_days      = 90
  minimum_versions_to_keep  = 5

  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
  }
}
```

## Import Existing Repository

To import the existing Artifact Registry repository:

```bash
terraform import module.artifact_registry.google_artifact_registry_repository.docker \
  projects/dtu-mlops-data-482907/locations/europe-west1/repositories/ct-scan-mlops
```

## Build and Push Images

```bash
# Authenticate with Artifact Registry
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build image
docker build -t europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops/api:latest .

# Push image
docker push europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops/api:latest
```
