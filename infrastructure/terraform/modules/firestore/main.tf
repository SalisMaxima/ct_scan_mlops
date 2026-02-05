/**
 * Firestore Module
 * Manages Firestore database for feedback storage (replaces SQLite)
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

# Firestore database (Native mode)
resource "google_firestore_database" "feedback" {
  project     = var.project_id
  name        = var.database_id
  location_id = var.location_id
  type        = "FIRESTORE_NATIVE"

  # Prevent accidental deletion - database stays if removed from Terraform
  deletion_policy = "ABANDON"

  # Point-in-time recovery
  point_in_time_recovery_enablement = var.enable_pitr ? "POINT_IN_TIME_RECOVERY_ENABLED" : "POINT_IN_TIME_RECOVERY_DISABLED"
}

# Note: Single-field indexes (like timestamp only) are created automatically by Firestore
# Removed google_firestore_index.feedback_by_timestamp as it's not necessary

# Index for feedback queries by prediction_id
resource "google_firestore_index" "feedback_by_prediction_id" {
  project    = var.project_id
  database   = google_firestore_database.feedback.name
  collection = "feedback"

  fields {
    field_path = "prediction_id"
    order      = "ASCENDING"
  }

  fields {
    field_path = "timestamp"
    order      = "DESCENDING"
  }
}

# Index for feedback queries by accuracy
resource "google_firestore_index" "feedback_by_accuracy" {
  project    = var.project_id
  database   = google_firestore_database.feedback.name
  collection = "feedback"

  fields {
    field_path = "is_correct"
    order      = "ASCENDING"
  }

  fields {
    field_path = "timestamp"
    order      = "DESCENDING"
  }
}

# TTL policy for feedback data (90-day retention)
resource "google_firestore_field" "feedback_ttl" {
  project    = var.project_id
  database   = google_firestore_database.feedback.name
  collection = "feedback"
  field      = "expireAt"

  ttl_config {}
}
