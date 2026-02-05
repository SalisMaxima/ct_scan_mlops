/**
 * Artifact Registry Module
 * Manages Docker container registry for CT Scan MLOps
 */

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Docker repository for container images
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker container repository for CT Scan MLOps"
  format        = "DOCKER"

  labels = merge(
    var.common_labels,
    {
      purpose = "docker-registry"
    }
  )

  # Cleanup policy to prevent unbounded storage costs
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "delete-old-images"
    action = "DELETE"

    condition {
      older_than = "${var.image_retention_days * 86400}s"
    }
  }

  cleanup_policies {
    id     = "keep-minimum-versions"
    action = "KEEP"

    most_recent_versions {
      keep_count = var.minimum_versions_to_keep
    }
  }
}

# IAM binding for Docker repository - Allow service accounts to push/pull
resource "google_artifact_registry_repository_iam_member" "docker_reader" {
  for_each = toset(var.docker_readers)

  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.reader"
  member     = each.value
}

resource "google_artifact_registry_repository_iam_member" "docker_writer" {
  for_each = toset(var.docker_writers)

  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.writer"
  member     = each.value
}

# Service account for Cloud Build (if not using default)
resource "google_artifact_registry_repository_iam_member" "cloud_build" {
  count = var.enable_cloud_build_access ? 1 : 0

  location   = google_artifact_registry_repository.docker.location
  repository = google_artifact_registry_repository.docker.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${var.project_number}@cloudbuild.gserviceaccount.com"
}
