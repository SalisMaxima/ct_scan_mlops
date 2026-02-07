/**
 * Cloud Run Module
 * Manages Cloud Run service for CT Scan API
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

resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = var.service_account_email

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        cpu_idle          = var.cpu_idle
        startup_cpu_boost = var.startup_cpu_boost
      }

      # Environment variables
      dynamic "env" {
        for_each = var.environment_variables
        content {
          name  = env.key
          value = env.value
        }
      }

      # Secret environment variables
      dynamic "env" {
        for_each = var.secret_environment_variables
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value.secret_name
              version = env.value.version
            }
          }
        }
      }

      # Volume mount for GCS bucket (models)
      dynamic "volume_mounts" {
        for_each = var.gcs_volume_mount != null ? [var.gcs_volume_mount] : []
        content {
          name       = volume_mounts.value.name
          mount_path = volume_mounts.value.mount_path
        }
      }
    }

    # GCS volume
    dynamic "volumes" {
      for_each = var.gcs_volume_mount != null ? [var.gcs_volume_mount] : []
      content {
        name = volumes.value.name
        gcs {
          bucket    = volumes.value.bucket
          read_only = volumes.value.read_only
        }
      }
    }

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    max_instance_request_concurrency = var.concurrency

    timeout = "${var.timeout_seconds}s"
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "api-service"
    }
  )
}

# Allow public access (no authentication)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count = var.allow_public_access ? 1 : 0

  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
