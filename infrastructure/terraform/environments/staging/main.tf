/**
 * Staging Environment Configuration
 * Production-like resources for final testing before prod deployment
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
    prefix = "environments/staging"
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
    environment = "staging"
    managed_by  = "terraform"
  }
}

# Storage Module
module "storage" {
  source = "../../modules/storage"

  project_id                  = var.project_id
  region                      = var.region
  terraform_state_bucket_name = var.terraform_state_bucket_name
  dvc_bucket_name             = "${var.dvc_bucket_name}-staging"
  models_bucket_name          = "${var.models_bucket_name}-staging"
  drift_logs_bucket_name      = "${var.drift_logs_bucket_name}-staging"

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
  repository_id  = "${var.artifact_registry_repository_id}-staging"

  docker_readers = [
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  docker_writers = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
  ]

  enable_cloud_build_access = true
  image_retention_days      = 60  # Between dev and prod
  minimum_versions_to_keep  = 4

  common_labels = local.common_labels
}

# IAM Module
module "iam" {
  source = "../../modules/iam"

  project_id = var.project_id

  github_actions_sa_name  = "github-actions-staging"
  cloud_run_sa_name       = "cloud-run-staging"
  drift_detection_sa_name = "drift-detection-staging"
  monitoring_sa_name      = "monitoring-staging"

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

  wandb_secret_id = "wandb-api-key-staging"

  wandb_secret_accessors = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  create_slack_webhook_secret = true  # Test notifications in staging
  slack_webhook_accessors = [
    "serviceAccount:${module.iam.monitoring_sa_email}",
  ]

  create_pagerduty_secret = false  # Not needed for staging

  common_labels = local.common_labels
}

# Workload Identity Module
module "workload_identity" {
  source = "../../modules/workload-identity"

  project_id     = var.project_id
  project_number = var.project_number

  service_account_id = module.iam.github_actions_sa_id
  github_repository  = var.github_repository
  repository_filter  = var.github_repository

  pool_id              = "github-pool-staging"
  provider_id          = "github-provider-staging"
  pool_display_name    = "GitHub Actions Pool (Staging)"
  provider_display_name = "GitHub OIDC Provider (Staging)"
}

# Firestore Module (optional separate staging database)
module "firestore" {
  source = "../../modules/firestore"

  project_id   = var.project_id
  database_id  = "staging"
  location_id  = "eur3"
  enable_pitr  = true  # Production-like
}

# Budget Module
module "budget" {
  source = "../../modules/budget"

  billing_account       = var.billing_account
  project_id            = var.project_id
  project_number        = var.project_number
  monthly_budget_amount = 250  # Between dev and prod
  currency_code         = "USD"

  budget_display_name = "CT Scan MLOps Staging Budget"

  disable_default_email_recipients = false
  create_pubsub_subscription      = false

  common_labels = local.common_labels
}

# Cloud Run Module
module "cloud_run" {
  source = "../../modules/cloud-run"

  project_id            = var.project_id
  region                = var.region
  service_name          = "${var.cloud_run_service_name}-staging"
  container_image       = var.container_image
  service_account_email = module.iam.cloud_run_sa_email

  cpu_limit    = "2000m"  # Same as prod
  memory_limit = "4Gi"    # Same as prod

  environment_variables = {
    PROJECT_ID  = var.project_id
    REGION      = var.region
    ENVIRONMENT = "staging"
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

  min_instances        = 1  # Keep warm like prod
  max_instances        = 5  # Lower max than prod
  timeout_seconds      = 300
  allow_public_access  = true

  common_labels = local.common_labels
}

# Monitoring Module
module "monitoring" {
  source = "../../modules/monitoring"

  depends_on = [module.cloud_run]

  project_id             = var.project_id
  cloud_run_service_name = "${var.cloud_run_service_name}-staging"
  alert_email            = var.alert_email

  # Enable P1/P2 alerts in staging to test notification flow
  enable_p1_p2_alerts = true

  # Production-like thresholds
  high_error_rate_threshold     = 0.05
  elevated_error_rate_threshold = 0.02
  memory_exhaustion_threshold   = 0.85  # Use recommended 85%
  high_cpu_threshold            = 0.80
  high_latency_threshold_ms     = 2000
  crash_loop_restart_threshold  = 3
}
