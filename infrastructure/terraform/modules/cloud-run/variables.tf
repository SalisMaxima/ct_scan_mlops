variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run service"
  type        = string
  default     = "europe-west1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "ct-scan-api"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for Cloud Run"
  type        = string
}

variable "cpu_limit" {
  description = "CPU limit for container"
  type        = string
  default     = "2000m"
}

variable "memory_limit" {
  description = "Memory limit for container"
  type        = string
  default     = "4Gi"
}

variable "environment_variables" {
  description = "Environment variables for the container"
  type        = map(string)
  default     = {}
}

variable "secret_environment_variables" {
  description = "Secret environment variables from Secret Manager"
  type = map(object({
    secret_name = string
    version     = string
  }))
  default = {}
}

variable "gcs_volume_mount" {
  description = "GCS volume mount configuration"
  type = object({
    name       = string
    bucket     = string
    mount_path = string
    read_only  = bool
  })
  default = null
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "concurrency" {
  description = "Maximum concurrent requests per container instance"
  type        = number
  default     = 80
}

variable "cpu_idle" {
  description = "Whether CPU should be throttled when no requests are being processed"
  type        = bool
  default     = true
}

variable "startup_cpu_boost" {
  description = "Whether to allocate extra CPU during startup"
  type        = bool
  default     = true
}

variable "timeout_seconds" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}

variable "allow_public_access" {
  description = "Allow public access without authentication"
  type        = bool
  default     = true
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}
