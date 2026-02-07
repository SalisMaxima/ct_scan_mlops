/**
 * Budget Module
 * Manages GCP billing budgets and cost monitoring
 */

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Pub/Sub topic for budget alerts
resource "google_pubsub_topic" "billing_alerts" {
  name = var.pubsub_topic_name

  labels = merge(
    var.common_labels,
    {
      purpose = "billing-alerts"
    }
  )
}

# Budget with multiple threshold alerts
resource "google_billing_budget" "ct_scan_budget" {
  billing_account = var.billing_account
  display_name    = var.budget_display_name

  budget_filter {
    projects = ["projects/${var.project_number}"]

    # Optional: Filter by specific services
    # services = ["services/24E6-581D-38E5"] # Cloud Run
  }

  amount {
    specified_amount {
      currency_code = var.currency_code
      units         = tostring(var.monthly_budget_amount)
    }
  }

  # Alert at 25% threshold (early warning - Gemini recommendation)
  threshold_rules {
    threshold_percent = 0.25
    spend_basis       = "CURRENT_SPEND"
  }

  # Alert at 50% threshold
  threshold_rules {
    threshold_percent = 0.5
    spend_basis       = "CURRENT_SPEND"
  }

  # Alert at 90% threshold
  threshold_rules {
    threshold_percent = 0.9
    spend_basis       = "CURRENT_SPEND"
  }

  # Alert at 100% threshold
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "CURRENT_SPEND"
  }

  # Alert at 110% threshold (overspend warning)
  threshold_rules {
    threshold_percent = 1.1
    spend_basis       = "CURRENT_SPEND"
  }

  all_updates_rule {
    pubsub_topic                     = google_pubsub_topic.billing_alerts.id
    schema_version                   = "1.0"
    disable_default_iam_recipients   = var.disable_default_email_recipients
    monitoring_notification_channels = var.notification_channels
  }
}

# Pub/Sub subscription for processing billing alerts (optional)
resource "google_pubsub_subscription" "billing_alerts_subscription" {
  count = var.create_pubsub_subscription ? 1 : 0

  name  = "${var.pubsub_topic_name}-subscription"
  topic = google_pubsub_topic.billing_alerts.name

  # Messages retained for 7 days
  message_retention_duration = "604800s"

  # Acknowledge deadline
  ack_deadline_seconds = 60

  # Push to Cloud Function or webhook (optional)
  # push_config {
  #   push_endpoint = var.webhook_url
  # }
}
