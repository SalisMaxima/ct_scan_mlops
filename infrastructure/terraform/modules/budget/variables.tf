variable "billing_account" {
  description = "Billing account ID (format: XXXXXX-XXXXXX-XXXXXX)"
  type        = string
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_number" {
  description = "GCP project number"
  type        = string
}

variable "budget_display_name" {
  description = "Display name for the budget"
  type        = string
  default     = "CT Scan MLOps Monthly Budget"
}

variable "monthly_budget_amount" {
  description = "Monthly budget amount"
  type        = number
  default     = 500
}

variable "currency_code" {
  description = "Currency code (e.g., USD, EUR)"
  type        = string
  default     = "USD"
}

variable "pubsub_topic_name" {
  description = "Name of Pub/Sub topic for billing alerts"
  type        = string
  default     = "billing-budget-alerts"
}

variable "disable_default_email_recipients" {
  description = "Disable default email recipients (billing admins)"
  type        = bool
  default     = false
}

variable "notification_channels" {
  description = "Cloud Monitoring notification channels for alerts"
  type        = list(string)
  default     = []
}

variable "create_pubsub_subscription" {
  description = "Create Pub/Sub subscription for processing alerts"
  type        = bool
  default     = false
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}
