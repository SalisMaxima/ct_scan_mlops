output "cloud_run_url" {
  description = "URL of the Cloud Run API"
  value       = module.cloud_run.service_url
}

output "artifact_registry_url" {
  description = "URL of the Docker repository"
  value       = module.artifact_registry.repository_url
}

output "dvc_bucket_name" {
  description = "Name of the DVC bucket"
  value       = module.storage.dvc_bucket_name
}

output "models_bucket_name" {
  description = "Name of the models bucket"
  value       = module.storage.models_bucket_name
}

output "github_actions_sa_email" {
  description = "GitHub Actions service account email"
  value       = module.iam.github_actions_sa_email
}

output "workload_identity_provider" {
  description = "Workload Identity Provider for GitHub Actions"
  value       = module.workload_identity.workload_identity_provider
}

output "firestore_database_id" {
  description = "Firestore database ID"
  value       = module.firestore.database_id
}

output "budget_name" {
  description = "Budget name"
  value       = module.budget.budget_name
}

output "monitoring_email_channel_id" {
  description = "Monitoring email notification channel ID"
  value       = module.monitoring.email_channel_id
}

output "alert_policy_ids" {
  description = "Map of alert policy IDs"
  value       = module.monitoring.alert_policy_ids
}
