output "database_id" {
  description = "ID of the Firestore database"
  value       = google_firestore_database.feedback.name
}

output "database_location" {
  description = "Location of the Firestore database"
  value       = google_firestore_database.feedback.location_id
}

output "database_type" {
  description = "Type of the Firestore database"
  value       = google_firestore_database.feedback.type
}
