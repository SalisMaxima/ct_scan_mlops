output "github_actions_sa_email" {
  description = "Email of GitHub Actions service account"
  value       = google_service_account.github_actions.email
}

output "github_actions_sa_id" {
  description = "ID of GitHub Actions service account"
  value       = google_service_account.github_actions.id
}

output "cloud_run_sa_email" {
  description = "Email of Cloud Run service account"
  value       = google_service_account.cloud_run.email
}

output "cloud_run_sa_id" {
  description = "ID of Cloud Run service account"
  value       = google_service_account.cloud_run.id
}

output "drift_detection_sa_email" {
  description = "Email of Drift Detection service account"
  value       = google_service_account.drift_detection.email
}

output "monitoring_sa_email" {
  description = "Email of Monitoring service account"
  value       = google_service_account.monitoring.email
}
