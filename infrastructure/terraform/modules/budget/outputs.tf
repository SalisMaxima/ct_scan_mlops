output "budget_name" {
  description = "Name of the billing budget"
  value       = google_billing_budget.ct_scan_budget.name
}

output "pubsub_topic_name" {
  description = "Name of the Pub/Sub topic for billing alerts"
  value       = google_pubsub_topic.billing_alerts.name
}

output "pubsub_topic_id" {
  description = "ID of the Pub/Sub topic"
  value       = google_pubsub_topic.billing_alerts.id
}
