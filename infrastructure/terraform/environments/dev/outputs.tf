output "cloud_run_url" {
  description = "URL of the Dev Cloud Run API"
  value       = module.cloud_run.service_url
}

output "artifact_registry_url" {
  description = "URL of the Dev Docker repository"
  value       = module.artifact_registry.repository_url
}

output "dvc_bucket_name" {
  description = "Name of the Dev DVC bucket"
  value       = module.storage.dvc_bucket_name
}

output "models_bucket_name" {
  description = "Name of the Dev models bucket"
  value       = module.storage.models_bucket_name
}

output "github_actions_sa_email" {
  description = "Dev GitHub Actions service account email"
  value       = module.iam.github_actions_sa_email
}

output "workload_identity_provider" {
  description = "Dev Workload Identity Provider"
  value       = module.workload_identity.workload_identity_provider
}

output "budget_name" {
  description = "Dev budget name"
  value       = module.budget.budget_name
}
