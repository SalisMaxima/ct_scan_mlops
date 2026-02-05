variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "dtu-mlops-data-482907"
}

variable "project_number" {
  description = "GCP project number"
  type        = string
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "europe-west1"
}

variable "billing_account" {
  description = "Billing account ID"
  type        = string
}

# Storage
variable "terraform_state_bucket_name" {
  description = "Terraform state bucket name"
  type        = string
  default     = "dtu-mlops-terraform-state-482907"
}

variable "dvc_bucket_name" {
  description = "Base DVC storage bucket name (-staging will be appended)"
  type        = string
  default     = "dtu-mlops-dvc-storage-482907"
}

variable "models_bucket_name" {
  description = "Base models storage bucket name (-staging will be appended)"
  type        = string
  default     = "dtu-mlops-models-482907"
}

variable "drift_logs_bucket_name" {
  description = "Base drift logs bucket name (-staging will be appended)"
  type        = string
  default     = "ct-scan-drift-logs-482907"
}

# Artifact Registry
variable "artifact_registry_repository_id" {
  description = "Base artifact registry repository ID"
  type        = string
  default     = "ct-scan-mlops"
}

# Cloud Run
variable "cloud_run_service_name" {
  description = "Base Cloud Run service name (-staging will be appended)"
  type        = string
  default     = "ct-scan-api"
}

variable "container_image" {
  description = "Container image for Cloud Run"
  type        = string
}

# GitHub
variable "github_repository" {
  description = "GitHub repository (format: owner/repo)"
  type        = string
}

# Monitoring
variable "alert_email" {
  description = "Primary email for alerts"
  type        = string
}
