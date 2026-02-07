/**
 * Production Environment Configuration
 * Orchestrates all Terraform modules for CT Scan MLOps
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
    prefix = "environments/prod"
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}

# Local variables
locals {
  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
  }
}

# Storage Module
module "storage" {
  source = "../../modules/storage"

  project_id                  = var.project_id
  region                      = var.region
  terraform_state_bucket_name = var.terraform_state_bucket_name
  dvc_bucket_name             = var.dvc_bucket_name
  models_bucket_name          = var.models_bucket_name
  models_bucket_location      = "US" # Existing bucket is in US multi-region
  drift_logs_bucket_name      = var.drift_logs_bucket_name

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
  repository_id  = var.artifact_registry_repository_id

  docker_readers = [
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  docker_writers = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
  ]

  enable_cloud_build_access = true
  image_retention_days      = 90
  minimum_versions_to_keep  = 5

  common_labels = local.common_labels
}

# IAM Module
module "iam" {
  source = "../../modules/iam"

  project_id = var.project_id

  github_actions_sa_name  = "github-actions"
  cloud_run_sa_name       = "cloud-run"
  drift_detection_sa_name = "drift-detection"
  monitoring_sa_name      = "monitoring"

  # Explicitly configure IAM roles (reviewed and approved)
  github_actions_roles = [
    "roles/storage.admin",                   # DVC and model bucket access
    "roles/artifactregistry.writer",         # Push Docker images
    "roles/run.admin",                       # Deploy Cloud Run services
    "roles/iam.serviceAccountUser",          # Impersonate service accounts
    "roles/secretmanager.admin",             # Manage secrets (Terraform CI/CD)
    "roles/monitoring.admin",                # Manage alert policies (Terraform CI/CD)
    "roles/datastore.owner",                 # Manage Firestore (Terraform CI/CD)
    "roles/iam.serviceAccountAdmin",         # Manage service accounts (Terraform CI/CD)
    "roles/resourcemanager.projectIamAdmin", # Manage IAM bindings (Terraform CI/CD)
    "roles/pubsub.admin",                    # Manage PubSub topics (Terraform CI/CD)
  ]

  cloud_run_roles = [
    "roles/storage.objectViewer",         # Read models bucket
    "roles/secretmanager.secretAccessor", # Access W&B API key
    "roles/datastore.user",               # Read/write Firestore
  ]

  drift_detection_roles = [
    "roles/storage.objectAdmin",     # Write drift logs
    "roles/monitoring.metricWriter", # Publish custom metrics
    "roles/datastore.viewer",        # Read Firestore for analysis
  ]

  monitoring_roles = [
    "roles/monitoring.metricWriter",      # Publish monitoring metrics
    "roles/logging.logWriter",            # Write logs
    "roles/secretmanager.secretAccessor", # Access Slack/PagerDuty secrets
  ]
}

# Secret Manager Module
module "secret_manager" {
  source = "../../modules/secret-manager"

  project_id = var.project_id

  wandb_secret_accessors = [
    "serviceAccount:${module.iam.github_actions_sa_email}",
    "serviceAccount:${module.iam.cloud_run_sa_email}",
  ]

  create_slack_webhook_secret = true
  slack_webhook_accessors = [
    "serviceAccount:${module.iam.monitoring_sa_email}",
  ]

  create_pagerduty_secret = false # Enable when PagerDuty is configured

  common_labels = local.common_labels
}

# Workload Identity Module (for GitHub Actions keyless auth)
module "workload_identity" {
  source = "../../modules/workload-identity"

  project_id     = var.project_id
  project_number = var.project_number

  service_account_id = module.iam.github_actions_sa_id
  github_repository  = var.github_repository
  repository_filter  = var.github_repository

  pool_id               = "github-pool"
  provider_id           = "github-provider"
  pool_display_name     = "GitHub Actions Pool"
  provider_display_name = "GitHub OIDC Provider"
}

# Firestore Module (feedback database)
module "firestore" {
  source = "../../modules/firestore"

  project_id  = var.project_id
  database_id = "(default)"
  location_id = "eur3" # Europe multi-region
  enable_pitr = true
}

# Budget Module
module "budget" {
  source = "../../modules/budget"

  billing_account       = var.billing_account
  project_id            = var.project_id
  project_number        = var.project_number
  monthly_budget_amount = var.monthly_budget_amount
  currency_code         = "USD"

  disable_default_email_recipients = false
  create_pubsub_subscription       = false

  common_labels = local.common_labels
}

# Cloud Run Module
# NOTE: Configuration matches existing production service to avoid drift.
# Service account uses default compute SA for Phase 2 import compatibility.
# Can migrate to dedicated SA in a future phase.
module "cloud_run" {
  source = "../../modules/cloud-run"

  project_id      = var.project_id
  region          = var.region
  service_name    = var.cloud_run_service_name
  container_image = var.container_image
  # Using default compute SA to match existing service (Phase 2 import)
  service_account_email = "${var.project_number}-compute@developer.gserviceaccount.com"

  cpu_limit    = "2"
  memory_limit = "4Gi"

  # Environment variables matching actual production service
  environment_variables = {
    MODEL_PATH            = "/gcs/outputs/checkpoints/model.pt"
    FEATURE_METADATA_PATH = "/gcs/outputs/checkpoints/feature_metadata.json"
    CONFIG_PATH           = "/gcs/configs/config.yaml"
    DEPLOYMENT_ID         = "20260131-102227"
    RELOAD_TIME           = "1769851883"
    FEATURES_UPDATED      = "1769852116"
    GCP_PROJECT_ID        = var.project_id
    USE_FIRESTORE         = "1"
    DRIFT_REFERENCE_PATH  = "/gcs/drift/reference.csv"
    DRIFT_CURRENT_PATH    = "/gcs/drift/current.csv"
    DRIFT_REPORT_PATH     = "/gcs/drift/drift_report.html"
  }

  # Disabled for Phase 2 - existing service doesn't use secrets from Secret Manager
  # secret_environment_variables = {
  #   WANDB_API_KEY = {
  #     secret_name = module.secret_manager.wandb_secret_id
  #     version     = "latest"
  #   }
  # }

  # GCS volume mount matching actual service
  gcs_volume_mount = {
    name       = "gcs-bucket"
    bucket     = "dtu-mlops-data-482907_cloudbuild"
    mount_path = "/gcs"
    read_only  = false
  }

  min_instances       = 0  # Actual service config
  max_instances       = 20 # Actual service config
  concurrency         = 80 # Actual service config
  timeout_seconds     = 300
  allow_public_access = true

  common_labels = local.common_labels
}

# Monitoring Module
module "monitoring" {
  source = "../../modules/monitoring"

  # Ensure Cloud Run exists before creating monitoring alerts
  depends_on = [module.cloud_run]

  project_id             = var.project_id
  cloud_run_service_name = var.cloud_run_service_name
  alert_email            = var.alert_email

  # Slack and PagerDuty are optional
  # slack_webhook_url = var.slack_webhook_url
  # pagerduty_key     = var.pagerduty_key

  # Start with P1/P2 alerts disabled (shadow mode)
  enable_p1_p2_alerts = false

  # Threshold configuration
  high_error_rate_threshold     = 0.05 # 5%
  elevated_error_rate_threshold = 0.02 # 2%
  memory_exhaustion_threshold   = 0.85 # 85% (Gemini recommendation - prevents OOM)
  high_cpu_threshold            = 0.8  # 80%
  high_latency_threshold_ms     = 2000 # 2 seconds
  crash_loop_restart_threshold  = 3
}
