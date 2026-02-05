/**
 * Secret Manager Module
 * Manages secrets for CT Scan MLOps
 *
 * IMPORTANT: This module creates secret PLACEHOLDERS only.
 * Secret values must be injected via GitHub Actions secrets or gcloud CLI.
 * Never store secret values in Terraform state.
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

# W&B API Key secret (placeholder)
resource "google_secret_manager_secret" "wandb_api_key" {
  secret_id = var.wandb_secret_id

  replication {
    auto {}
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "wandb-api-key"
    }
  )
}

# IAM binding for W&B secret - Allow service accounts to access
resource "google_secret_manager_secret_iam_member" "wandb_accessor" {
  for_each = toset(var.wandb_secret_accessors)

  secret_id = google_secret_manager_secret.wandb_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# Slack webhook token secret (placeholder)
resource "google_secret_manager_secret" "slack_webhook" {
  count     = var.create_slack_webhook_secret ? 1 : 0
  secret_id = var.slack_webhook_secret_id

  replication {
    auto {}
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "slack-webhook"
    }
  )
}

resource "google_secret_manager_secret_iam_member" "slack_webhook_accessor" {
  for_each = var.create_slack_webhook_secret ? toset(var.slack_webhook_accessors) : []

  secret_id = google_secret_manager_secret.slack_webhook[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# PagerDuty integration key secret (placeholder)
resource "google_secret_manager_secret" "pagerduty_key" {
  count     = var.create_pagerduty_secret ? 1 : 0
  secret_id = var.pagerduty_secret_id

  replication {
    auto {}
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "pagerduty-key"
    }
  )
}

resource "google_secret_manager_secret_iam_member" "pagerduty_accessor" {
  for_each = var.create_pagerduty_secret ? toset(var.pagerduty_accessors) : []

  secret_id = google_secret_manager_secret.pagerduty_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
}

# Generic secrets (for custom use cases)
resource "google_secret_manager_secret" "generic" {
  for_each = var.generic_secrets

  secret_id = each.key

  replication {
    auto {}
  }

  labels = merge(
    var.common_labels,
    {
      purpose = "generic-secret"
    }
  )
}

resource "google_secret_manager_secret_iam_member" "generic_accessor" {
  for_each = merge([
    for secret_id, config in var.generic_secrets : {
      for accessor in config.accessors :
      "${secret_id}-${accessor}" => {
        secret_id = secret_id
        accessor  = accessor
      }
    }
  ]...)

  secret_id = google_secret_manager_secret.generic[each.value.secret_id].id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value.accessor
}
