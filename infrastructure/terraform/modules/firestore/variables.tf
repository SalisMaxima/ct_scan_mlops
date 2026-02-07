variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "database_id" {
  description = "Firestore database ID"
  type        = string
  default     = "(default)"
}

variable "location_id" {
  description = "Firestore location ID"
  type        = string
  default     = "eur3" # Europe multi-region
}

variable "enable_pitr" {
  description = "Enable point-in-time recovery"
  type        = bool
  default     = true
}

variable "retention_days" {
  description = "Data retention period in days (for documentation purposes)"
  type        = number
  default     = 90
}
