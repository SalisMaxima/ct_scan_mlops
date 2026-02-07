variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_number" {
  description = "GCP project number (required for Cloud Build IAM)"
  type        = string
}

variable "region" {
  description = "GCP region for Artifact Registry"
  type        = string
  default     = "europe-west1"
}

variable "repository_id" {
  description = "ID of the Docker repository"
  type        = string
  default     = "ct-scan-mlops"
}

variable "image_retention_days" {
  description = "Number of days to retain old images"
  type        = number
  default     = 90
}

variable "minimum_versions_to_keep" {
  description = "Minimum number of image versions to keep regardless of age"
  type        = number
  default     = 5
}

variable "docker_readers" {
  description = "Service accounts with read access to Docker repository"
  type        = list(string)
  default     = []
}

variable "docker_writers" {
  description = "Service accounts with write access to Docker repository"
  type        = list(string)
  default     = []
}

variable "enable_cloud_build_access" {
  description = "Grant Cloud Build service account write access"
  type        = bool
  default     = true
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}
