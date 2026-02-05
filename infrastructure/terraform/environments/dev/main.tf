/**
 * Development Environment Configuration
 * Smaller resources, lower costs, more permissive for testing
 */

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "dtu-mlops-terraform-state-482907"
    prefix = "environments/dev"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Local variables
locals {
  common_labels = {
    project     = "ct-scan-mlops"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Storage Module
module "storage" {
  source = "../../modules/storage"

  project_id                  = var.project_id
  region                      = var.region
  terraform_state_bucket_name = var.terraform_state_bucket_name
  dvc_bucket_name             = "${var.dvc_bucket_name}-dev"
  models_bucket_name          = "${var.models_bucket_name}-dev"
  drift_logs_bucket_name      = "${var.drift_logs_bucket_name}-dev"

  dvc_bucket_admins = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
  ]

  models_bucket_admins = [
    "serviceAccount:${module.iam.cloud_run_sa_email}",
    "serviceAccount:${module.iam.github_actions_sa_email}",
  ]

  drift_logs_writers = [
    "serviceAccount:${module.iam.drift_detection_sa_email}",
  ]

  models_bucket_public_read = false
  common_labels             = local.common_labels
}

# Artifact Registry Module
module "artifact_registry" {
  source = "../../modules/artifact-registry"

  project_id     = var.project_id
  project_number = var.project_number
  region         = var.region
  repository_id  = "${var.artifact_registry_repository_id}-dev"

  docker_readers = [
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  docker_writers = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
  ]

  enable_cloud_build_access = true
  image_retention_days      = 30  # Shorter retention for dev
  minimum_versions_to_keep  = 3   # Fewer versions for dev

  common_labels = local.common_labels
}

# IAM Module
module "iam" {
  source = "../../modules/iam"

  project_id = var.project_id

  github_actions_sa_name  = "github-actions-dev"
  cloud_run_sa_name       = "cloud-run-dev"
  drift_detection_sa_name = "drift-detection-dev"
  monitoring_sa_name      = "monitoring-dev"

  # Same roles as prod
  github_actions_roles = [
    "roles/storage.admin",
    "roles/artifactregistry.writer",
    "roles/run.admin",
    "roles/iam.serviceAccountUser",
  ]

  cloud_run_roles = [
    "roles/storage.objectViewer",
    "roles/secretmanager.secretAccessor",
    "roles/datastore.user",
  ]

  drift_detection_roles = [
    "roles/storage.objectAdmin",
    "roles/monitoring.metricWriter",
    "roles/datastore.viewer",
  ]

  monitoring_roles = [
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
  ]
}

# Secret Manager Module
module "secret_manager" {
  source = "../../modules/secret-manager"

  project_id = var.project_id

  wandb_secret_id = "wandb-api-key-dev"

  wandb_secret_accessors = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  create_slack_webhook_secret = false  # Not needed for dev
  create_pagerduty_secret     = false  # Not needed for dev

  common_labels = local.common_labels
}

# Workload Identity Module (optional for dev)
module "workload_identity" {
  source = "../../modules/workload-identity"

  project_id     = var.project_id
  project_number = var.project_number

  service_account_id = module.iam.github_actions_sa_id
  github_repository  = var.github_repository
  repository_filter  = var.github_repository

  pool_id              = "github-pool-dev"
  provider_id          = "github-provider-dev"
  pool_display_name    = "GitHub Actions Pool (Dev)"
  provider_display_name = "GitHub OIDC Provider (Dev)"
}

# Firestore Module (share with prod or create separate dev database)
# Note: Firestore database name must be unique per project
# Option 1: Share prod database (default)
# Option 2: Create separate dev database (uncomment below)

# module "firestore" {
#   source = "../../modules/firestore"
#
#   project_id   = var.project_id
#   database_id  = "dev"  # Separate dev database
#   location_id  = "eur3"
#   enable_pitr  = false  # Save costs in dev
# }

# Budget Module
module "budget" {
  source = "../../modules/budget"

  billing_account       = var.billing_account
  project_id            = var.project_id
  project_number        = var.project_number
  monthly_budget_amount = 100  # Lower budget for dev
  currency_code         = "USD"

  budget_display_name = "CT Scan MLOps Dev Budget"

  disable_default_email_recipients = false
  create_pubsub_subscription      = false

  common_labels = local.common_labels
}

# Cloud Run Module
module "cloud_run" {
  source = "../../modules/cloud-run"

  project_id            = var.project_id
  region                = var.region
  service_name          = "${var.cloud_run_service_name}-dev"
  container_image       = var.container_image
  service_account_email = module.iam.cloud_run_sa_email

  cpu_limit    = "1000m"  # Smaller for dev
  memory_limit = "2Gi"    # Smaller for dev

  environment_variables = {
    PROJECT_ID  = var.project_id
    REGION      = var.region
    ENVIRONMENT = "dev"
  }

  secret_environment_variables = {
    WANDB_API_KEY = {
      secret_name = module.secret_manager.wandb_secret_id
      version     = "latest"
    }
  }

  gcs_volume_mount = {
    name       = "models"
    bucket     = module.storage.models_bucket_name
    mount_path = "/models"
    read_only  = true
  }

  min_instances        = 0  # Scale to zero in dev
  max_instances        = 3  # Lower max for dev
  timeout_seconds      = 300
  allow_public_access  = true

  common_labels = local.common_labels
}

# Monitoring Module (optional for dev - enable for testing alerts)
module "monitoring" {
  source = "../../modules/monitoring"

  depends_on = [module.cloud_run]

  project_id             = var.project_id
  cloud_run_service_name = "${var.cloud_run_service_name}-dev"
  alert_email            = var.alert_email

  # Disable most alerts in dev to avoid noise
  enable_p1_p2_alerts = false

  # Higher thresholds for dev (more permissive)
  high_error_rate_threshold     = 0.10 # 10% (vs 5% in prod)
  elevated_error_rate_threshold = 0.05 # 5% (vs 2% in prod)
  memory_exhaustion_threshold   = 0.95 # 95% (vs 90% in prod)
  high_cpu_threshold            = 0.90 # 90% (vs 80% in prod)
  high_latency_threshold_ms     = 5000 # 5s (vs 2s in prod)
  crash_loop_restart_threshold  = 5    # 5 restarts (vs 3 in prod)
}
