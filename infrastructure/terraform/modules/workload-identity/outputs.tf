output "pool_name" {
  description = "Name of the Workload Identity Pool"
  value       = google_iam_workload_identity_pool.github.name
}

output "provider_name" {
  description = "Name of the Workload Identity Provider"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "workload_identity_provider" {
  description = "Full workload identity provider path (for GitHub Actions)"
  value       = "projects/${var.project_number}/locations/global/workloadIdentityPools/${var.pool_id}/providers/${var.provider_id}"
}
