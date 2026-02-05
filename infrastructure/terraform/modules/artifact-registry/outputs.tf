output "repository_id" {
  description = "ID of the Docker repository"
  value       = google_artifact_registry_repository.docker.repository_id
}

output "repository_url" {
  description = "URL of the Docker repository"
  value       = "${google_artifact_registry_repository.docker.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "repository_location" {
  description = "Location of the Docker repository"
  value       = google_artifact_registry_repository.docker.location
}

output "repository_name" {
  description = "Full name of the Docker repository"
  value       = google_artifact_registry_repository.docker.name
}
