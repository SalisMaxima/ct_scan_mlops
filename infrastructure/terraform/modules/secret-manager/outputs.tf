output "wandb_secret_id" {
  description = "ID of the W&B API key secret"
  value       = google_secret_manager_secret.wandb_api_key.secret_id
}

output "wandb_secret_name" {
  description = "Full resource name of the W&B API key secret"
  value       = google_secret_manager_secret.wandb_api_key.name
}

output "slack_webhook_secret_id" {
  description = "ID of the Slack webhook secret (if created)"
  value       = var.create_slack_webhook_secret ? google_secret_manager_secret.slack_webhook[0].secret_id : null
}

output "slack_webhook_secret_name" {
  description = "Full resource name of the Slack webhook secret (if created)"
  value       = var.create_slack_webhook_secret ? google_secret_manager_secret.slack_webhook[0].name : null
}

output "pagerduty_secret_id" {
  description = "ID of the PagerDuty secret (if created)"
  value       = var.create_pagerduty_secret ? google_secret_manager_secret.pagerduty_key[0].secret_id : null
}

output "pagerduty_secret_name" {
  description = "Full resource name of the PagerDuty secret (if created)"
  value       = var.create_pagerduty_secret ? google_secret_manager_secret.pagerduty_key[0].name : null
}

output "generic_secret_ids" {
  description = "Map of generic secret IDs"
  value       = { for k, v in google_secret_manager_secret.generic : k => v.secret_id }
}

output "generic_secret_names" {
  description = "Map of generic secret full resource names"
  value       = { for k, v in google_secret_manager_secret.generic : k => v.name }
}
