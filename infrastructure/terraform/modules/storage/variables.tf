variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for storage buckets"
  type        = string
  default     = "europe-west1"
}

variable "terraform_state_bucket_name" {
  description = "Name of the Terraform state bucket"
  type        = string
}

variable "dvc_bucket_name" {
  description = "Name of the DVC storage bucket"
  type        = string
}

variable "models_bucket_name" {
  description = "Name of the models storage bucket"
  type        = string
}

variable "models_bucket_location" {
  description = "Location for models bucket (defaults to region if not specified). Use for existing buckets in different locations."
  type        = string
  default     = null
}

variable "drift_logs_bucket_name" {
  description = "Name of the drift logs bucket"
  type        = string
}

variable "dvc_bucket_admins" {
  description = "Service accounts with admin access to DVC bucket"
  type        = list(string)
  default     = []
}

variable "models_bucket_admins" {
  description = "Service accounts with admin access to models bucket"
  type        = list(string)
  default     = []
}

variable "drift_logs_writers" {
  description = "Service accounts with write access to drift logs bucket"
  type        = list(string)
  default     = []
}

variable "models_bucket_public_read" {
  description = "Enable public read access for models bucket"
  type        = bool
  default     = false
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# Optional: KMS key for encryption
# variable "kms_key_id" {
#   description = "KMS key ID for bucket encryption"
#   type        = string
#   default     = null
# }
