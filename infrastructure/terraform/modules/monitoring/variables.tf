variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "cloud_run_service_name" {
  description = "Name of the Cloud Run service to monitor"
  type        = string
  default     = "ct-scan-api"
}

variable "alert_email" {
  description = "Primary email for alerts"
  type        = string
}

variable "email_channel_display_name" {
  description = "Display name for email notification channel"
  type        = string
  default     = "CT Scan Team Email"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications (optional)"
  type        = string
  default     = null
  sensitive   = true
}

variable "slack_channel_name" {
  description = "Slack channel name"
  type        = string
  default     = "#ct-scan-alerts"
}

variable "pagerduty_key" {
  description = "PagerDuty integration key (optional)"
  type        = string
  default     = null
  sensitive   = true
}

variable "enable_p1_p2_alerts" {
  description = "Enable P1/P2 alerts (set to false for shadow mode)"
  type        = bool
  default     = false
}

# Threshold configuration
variable "high_error_rate_threshold" {
  description = "High error rate threshold (fraction, e.g., 0.05 = 5%)"
  type        = number
  default     = 0.05
}

variable "elevated_error_rate_threshold" {
  description = "Elevated error rate threshold (fraction, e.g., 0.02 = 2%)"
  type        = number
  default     = 0.02
}

variable "memory_exhaustion_threshold" {
  description = "Memory exhaustion threshold (fraction, e.g., 0.9 = 90%)"
  type        = number
  default     = 0.9
}

variable "high_cpu_threshold" {
  description = "High CPU usage threshold (fraction, e.g., 0.8 = 80%)"
  type        = number
  default     = 0.8
}

variable "high_latency_threshold_ms" {
  description = "High latency threshold in milliseconds"
  type        = number
  default     = 2000
}

variable "crash_loop_restart_threshold" {
  description = "Number of restarts to trigger crash loop alert"
  type        = number
  default     = 3
}
