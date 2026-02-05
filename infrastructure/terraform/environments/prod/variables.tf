variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "dtu-mlops-data-482907"
}

variable "project_number" {
  description = "GCP project number (for Workload Identity)"
  type        = string
  # Get with: gcloud projects describe dtu-mlops-data-482907 --format="value(projectNumber)"
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "europe-west1"
}

variable "billing_account" {
  description = "Billing account ID (format: XXXXXX-XXXXXX-XXXXXX)"
  type        = string
  # Get with: gcloud billing accounts list
}

# Storage
variable "terraform_state_bucket_name" {
  description = "Terraform state bucket name"
  type        = string
  default     = "dtu-mlops-terraform-state-482907"
}

variable "dvc_bucket_name" {
  description = "DVC storage bucket name"
  type        = string
  default     = "dtu-mlops-dvc-storage-482907"
}

variable "models_bucket_name" {
  description = "Models storage bucket name"
  type        = string
  default     = "dtu-mlops-data-482907_cloudbuild"
}

variable "drift_logs_bucket_name" {
  description = "Drift logs bucket name"
  type        = string
  default     = "ct-scan-drift-logs-482907"
}

# Artifact Registry
variable "artifact_registry_repository_id" {
  description = "Artifact Registry repository ID"
  type        = string
  default     = "ct-scan-mlops"
}

# Cloud Run
variable "cloud_run_service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "ct-scan-api"
}

variable "container_image" {
  description = "Container image for Cloud Run"
  type        = string
  # Format: europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops/api:latest
}

# GitHub
variable "github_repository" {
  description = "GitHub repository (format: owner/repo)"
  type        = string
  # Example: "yourusername/ct_scan_mlops"
}

# Budget
variable "monthly_budget_amount" {
  description = "Monthly budget in USD"
  type        = number
  default     = 500
}

# Monitoring
variable "alert_email" {
  description = "Primary email for alerts"
  type        = string
}

variable "slack_webhook_url" {
  description = "Slack webhook URL (optional)"
  type        = string
  default     = null
  sensitive   = true
}

variable "pagerduty_key" {
  description = "PagerDuty integration key (optional)"
  type        = string
  default     = null
  sensitive   = true
}
