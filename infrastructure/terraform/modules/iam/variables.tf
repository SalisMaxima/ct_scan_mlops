variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "github_actions_sa_name" {
  description = "Name for GitHub Actions service account"
  type        = string
  default     = "github-actions"
}

variable "cloud_run_sa_name" {
  description = "Name for Cloud Run service account"
  type        = string
  default     = "cloud-run"
}

variable "drift_detection_sa_name" {
  description = "Name for Drift Detection service account"
  type        = string
  default     = "drift-detection"
}

variable "monitoring_sa_name" {
  description = "Name for Monitoring service account"
  type        = string
  default     = "monitoring"
}

variable "github_actions_roles" {
  description = "IAM roles for GitHub Actions service account"
  type        = list(string)
  default = [
    "roles/storage.admin",
    "roles/artifactregistry.writer",
    "roles/run.admin",
    "roles/iam.serviceAccountUser",
    "roles/cloudfunctions.admin",
  ]
}

variable "cloud_run_roles" {
  description = "IAM roles for Cloud Run service account"
  type        = list(string)
  default = [
    "roles/storage.objectViewer",
    "roles/secretmanager.secretAccessor",
    "roles/datastore.user",
  ]
}

variable "drift_detection_roles" {
  description = "IAM roles for Drift Detection service account"
  type        = list(string)
  default = [
    "roles/storage.objectAdmin",
    "roles/monitoring.metricWriter",
    "roles/datastore.viewer",
  ]
}

variable "monitoring_roles" {
  description = "IAM roles for Monitoring service account"
  type        = list(string)
  default = [
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
  ]
}
