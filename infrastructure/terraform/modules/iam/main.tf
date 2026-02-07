/**
 * IAM Module
 * Manages service accounts and IAM bindings
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

# GitHub Actions service account
resource "google_service_account" "github_actions" {
  account_id   = var.github_actions_sa_name
  display_name = "GitHub Actions Service Account"
  description  = "Service account for GitHub Actions CI/CD"
}

# Cloud Run service account
resource "google_service_account" "cloud_run" {
  account_id   = var.cloud_run_sa_name
  display_name = "Cloud Run Service Account"
  description  = "Service account for Cloud Run API service"
}

# Drift detection service account
resource "google_service_account" "drift_detection" {
  account_id   = var.drift_detection_sa_name
  display_name = "Drift Detection Service Account"
  description  = "Service account for drift detection Cloud Function"
}

# Monitoring service account
resource "google_service_account" "monitoring" {
  account_id   = var.monitoring_sa_name
  display_name = "Monitoring Service Account"
  description  = "Service account for monitoring and alerting"
}

# IAM roles for GitHub Actions SA
resource "google_project_iam_member" "github_actions_roles" {
  for_each = toset(var.github_actions_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

# IAM roles for Cloud Run SA
resource "google_project_iam_member" "cloud_run_roles" {
  for_each = toset(var.cloud_run_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

# IAM roles for Drift Detection SA
resource "google_project_iam_member" "drift_detection_roles" {
  for_each = toset(var.drift_detection_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.drift_detection.email}"
}

# IAM roles for Monitoring SA
resource "google_project_iam_member" "monitoring_roles" {
  for_each = toset(var.monitoring_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.monitoring.email}"
}
