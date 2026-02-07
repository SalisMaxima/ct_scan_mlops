/**
 * Storage Module
 * Manages GCS buckets for DVC, models, and Terraform state
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

# Terraform state bucket with encryption and versioning
resource "google_storage_bucket" "terraform_state" {
  name          = var.terraform_state_bucket_name
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  # Optional: Customer-managed encryption (requires KMS key)
  # encryption {
  #   default_kms_key_name = var.kms_key_id
  # }

  # Lifecycle policy to prevent state file bloat
  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "terraform-state"
    }
  )
}

# DVC storage bucket
resource "google_storage_bucket" "dvc" {
  name          = var.dvc_bucket_name
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 365 # Keep data for 1 year
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "dvc-storage"
    }
  )
}

# Model artifacts bucket
resource "google_storage_bucket" "models" {
  name          = var.models_bucket_name
  location      = coalesce(var.models_bucket_location, var.region)
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 180 # Keep models for 6 months
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "model-artifacts"
    }
  )

  # Prevent accidental deletion of production model data
  lifecycle {
    prevent_destroy = true
  }
}

# Drift logs bucket with 90-day retention
resource "google_storage_bucket" "drift_logs" {
  name          = var.drift_logs_bucket_name
  location      = var.region
  force_destroy = false # Protect logs from accidental deletion

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90 # 90-day retention as per plan
    }
    action {
      type = "Delete"
    }
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "drift-logs"
    }
  )
}

# IAM binding for DVC bucket - Allow service accounts to read/write
resource "google_storage_bucket_iam_member" "dvc_admin" {
  for_each = toset(var.dvc_bucket_admins)

  bucket = google_storage_bucket.dvc.name
  role   = "roles/storage.objectAdmin"
  member = each.value
}

# IAM binding for models bucket
resource "google_storage_bucket_iam_member" "models_admin" {
  for_each = toset(var.models_bucket_admins)

  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectAdmin"
  member = each.value
}

# IAM binding for drift logs bucket
resource "google_storage_bucket_iam_member" "drift_logs_writer" {
  for_each = toset(var.drift_logs_writers)

  bucket = google_storage_bucket.drift_logs.name
  role   = "roles/storage.objectCreator"
  member = each.value
}

# Public read access for models bucket (if needed for Cloud Run)
resource "google_storage_bucket_iam_member" "models_public_read" {
  count = var.models_bucket_public_read ? 1 : 0

  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
