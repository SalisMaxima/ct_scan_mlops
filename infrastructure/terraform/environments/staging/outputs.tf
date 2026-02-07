output "cloud_run_url" {
  description = "URL of the Staging Cloud Run API"
  value       = module.cloud_run.service_url
}

output "artifact_registry_url" {
  description = "URL of the Staging Docker repository"
  value       = module.artifact_registry.repository_url
}

output "dvc_bucket_name" {
  description = "Name of the Staging DVC bucket"
  value       = module.storage.dvc_bucket_name
}

output "models_bucket_name" {
  description = "Name of the Staging models bucket"
  value       = module.storage.models_bucket_name
}

output "github_actions_sa_email" {
  description = "Staging GitHub Actions service account email"
  value       = module.iam.github_actions_sa_email
}

output "workload_identity_provider" {
  description = "Staging Workload Identity Provider"
  value       = module.workload_identity.workload_identity_provider
}

output "firestore_database_id" {
  description = "Staging Firestore database ID"
  value       = module.firestore.database_id
}

output "budget_name" {
  description = "Staging budget name"
  value       = module.budget.budget_name
}

output "monitoring_email_channel_id" {
  description = "Staging monitoring email channel ID"
  value       = module.monitoring.email_channel_id
}
