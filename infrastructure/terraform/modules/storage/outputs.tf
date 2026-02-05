output "terraform_state_bucket_name" {
  description = "Name of the Terraform state bucket"
  value       = google_storage_bucket.terraform_state.name
}

output "terraform_state_bucket_url" {
  description = "URL of the Terraform state bucket"
  value       = google_storage_bucket.terraform_state.url
}

output "dvc_bucket_name" {
  description = "Name of the DVC storage bucket"
  value       = google_storage_bucket.dvc.name
}

output "dvc_bucket_url" {
  description = "URL of the DVC storage bucket"
  value       = google_storage_bucket.dvc.url
}

output "models_bucket_name" {
  description = "Name of the models storage bucket"
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "URL of the models storage bucket"
  value       = google_storage_bucket.models.url
}

output "drift_logs_bucket_name" {
  description = "Name of the drift logs bucket"
  value       = google_storage_bucket.drift_logs.name
}

output "drift_logs_bucket_url" {
  description = "URL of the drift logs bucket"
  value       = google_storage_bucket.drift_logs.url
}
