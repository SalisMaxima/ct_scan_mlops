output "email_channel_id" {
  description = "ID of the email notification channel"
  value       = google_monitoring_notification_channel.email_primary.id
}

output "slack_channel_id" {
  description = "ID of the Slack notification channel (if created)"
  value       = var.slack_webhook_url != null ? google_monitoring_notification_channel.slack[0].id : null
}

output "pagerduty_channel_id" {
  description = "ID of the PagerDuty notification channel (if created)"
  value       = var.pagerduty_key != null ? google_monitoring_notification_channel.pagerduty[0].id : null
}

output "alert_policy_ids" {
  description = "Map of alert policy IDs"
  value = {
    api_downtime         = google_monitoring_alert_policy.api_downtime.id
    high_error_rate      = google_monitoring_alert_policy.high_error_rate.id
    memory_exhaustion    = google_monitoring_alert_policy.memory_exhaustion.id
    crash_loop           = google_monitoring_alert_policy.crash_loop.id
    high_latency         = google_monitoring_alert_policy.high_latency.id
    elevated_error_rate  = google_monitoring_alert_policy.elevated_error_rate.id
    high_cpu             = google_monitoring_alert_policy.high_cpu.id
  }
}
