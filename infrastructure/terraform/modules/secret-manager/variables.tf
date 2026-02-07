variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "wandb_secret_id" {
  description = "Secret ID for W&B API key"
  type        = string
  default     = "wandb-api-key"
}

variable "wandb_secret_accessors" {
  description = "Service accounts with access to W&B secret"
  type        = list(string)
  default     = []
}

variable "create_slack_webhook_secret" {
  description = "Create secret for Slack webhook token"
  type        = bool
  default     = false
}

variable "slack_webhook_secret_id" {
  description = "Secret ID for Slack webhook token"
  type        = string
  default     = "slack-webhook-token"
}

variable "slack_webhook_accessors" {
  description = "Service accounts with access to Slack webhook secret"
  type        = list(string)
  default     = []
}

variable "create_pagerduty_secret" {
  description = "Create secret for PagerDuty integration key"
  type        = bool
  default     = false
}

variable "pagerduty_secret_id" {
  description = "Secret ID for PagerDuty integration key"
  type        = string
  default     = "pagerduty-integration-key"
}

variable "pagerduty_accessors" {
  description = "Service accounts with access to PagerDuty secret"
  type        = list(string)
  default     = []
}

variable "generic_secrets" {
  description = "Map of generic secrets to create with their accessors"
  type = map(object({
    accessors = list(string)
  }))
  default = {}
}

variable "common_labels" {
  description = "Common labels to apply to all resources"
  type        = map(string)
  default     = {}
}
