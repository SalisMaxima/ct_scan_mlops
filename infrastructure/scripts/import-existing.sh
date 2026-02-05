#!/bin/bash
# Import existing GCP resources into Terraform state
# This script should be run from infrastructure/terraform/environments/prod/

set -e

PROJECT_ID="dtu-mlops-data-482907"
REGION="europe-west1"

echo "================================================================"
echo "Terraform Import Script for CT Scan MLOps"
echo "================================================================"
echo ""
echo "⚠️  WARNING: This script will import existing GCP resources"
echo "   into Terraform state. Make sure you have:"
echo "   1. Initialized Terraform (terraform init)"
echo "   2. Configured terraform.tfvars with correct values"
echo "   3. Backed up any existing state"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting import process..."
echo ""

# Function to import resource with error handling
import_resource() {
    local resource_address=$1
    local resource_id=$2
    local resource_name=$3

    echo "Importing $resource_name..."
    if terraform import "$resource_address" "$resource_id" 2>&1 | tee /tmp/terraform_import.log; then
        echo "✅ Successfully imported $resource_name"
    else
        if grep -q "already managed" /tmp/terraform_import.log; then
            echo "⏭️  $resource_name already in state, skipping"
        else
            echo "❌ Failed to import $resource_name"
            echo "   You may need to import this manually"
        fi
    fi
    echo ""
}

# ================================================================
# Storage Buckets
# ================================================================
echo "===== Importing Storage Buckets ====="
import_resource \
    "module.storage.google_storage_bucket.dvc" \
    "dtu-mlops-dvc-storage-482907" \
    "DVC Storage Bucket"

import_resource \
    "module.storage.google_storage_bucket.models" \
    "dtu-mlops-data-482907_cloudbuild" \
    "Models Storage Bucket"

# Note: Terraform state bucket should already exist (created manually first)
# import_resource \
#     "module.storage.google_storage_bucket.terraform_state" \
#     "dtu-mlops-terraform-state-482907" \
#     "Terraform State Bucket"

# ================================================================
# Artifact Registry
# ================================================================
echo "===== Importing Artifact Registry ====="
import_resource \
    "module.artifact_registry.google_artifact_registry_repository.docker" \
    "projects/${PROJECT_ID}/locations/${REGION}/repositories/ct-scan-mlops" \
    "Docker Repository"

# ================================================================
# Secret Manager
# ================================================================
echo "===== Importing Secret Manager Secrets ====="
import_resource \
    "module.secret_manager.google_secret_manager_secret.wandb_api_key" \
    "projects/${PROJECT_ID}/secrets/wandb-api-key" \
    "W&B API Key Secret"

# ================================================================
# IAM Service Accounts
# ================================================================
echo "===== Importing IAM Service Accounts ====="
# Note: Only import if they already exist. If they don't exist, Terraform will create them.

# Check if service accounts exist before importing
echo "Checking for existing service accounts..."

if gcloud iam service-accounts describe github-actions@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    import_resource \
        "module.iam.google_service_account.github_actions" \
        "projects/${PROJECT_ID}/serviceAccounts/github-actions@${PROJECT_ID}.iam.gserviceaccount.com" \
        "GitHub Actions Service Account"
else
    echo "⏭️  GitHub Actions SA doesn't exist, will be created"
fi

if gcloud iam service-accounts describe cloud-run@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    import_resource \
        "module.iam.google_service_account.cloud_run" \
        "projects/${PROJECT_ID}/serviceAccounts/cloud-run@${PROJECT_ID}.iam.gserviceaccount.com" \
        "Cloud Run Service Account"
else
    echo "⏭️  Cloud Run SA doesn't exist, will be created"
fi

# ================================================================
# Workload Identity Pool
# ================================================================
echo "===== Importing Workload Identity Pool ====="
# Check if pool exists
if gcloud iam workload-identity-pools describe github-pool --location=global &>/dev/null; then
    import_resource \
        "module.workload_identity.google_iam_workload_identity_pool.github" \
        "projects/${PROJECT_ID}/locations/global/workloadIdentityPools/github-pool" \
        "Workload Identity Pool"

    import_resource \
        "module.workload_identity.google_iam_workload_identity_pool_provider.github" \
        "projects/${PROJECT_ID}/locations/global/workloadIdentityPools/github-pool/providers/github-provider" \
        "Workload Identity Provider"
else
    echo "⏭️  Workload Identity Pool doesn't exist, will be created"
fi

# ================================================================
# Cloud Run Service
# ================================================================
echo "===== Importing Cloud Run Service ====="
if gcloud run services describe ct-scan-api --region=${REGION} &>/dev/null; then
    import_resource \
        "module.cloud_run.google_cloud_run_v2_service.api" \
        "projects/${PROJECT_ID}/locations/${REGION}/services/ct-scan-api" \
        "Cloud Run API Service"
else
    echo "⏭️  Cloud Run service doesn't exist, will be created"
fi

# ================================================================
# Validation
# ================================================================
echo ""
echo "================================================================"
echo "Import Complete! Running terraform plan to verify..."
echo "================================================================"
echo ""

terraform plan -detailed-exitcode
PLAN_EXIT_CODE=$?

echo ""
if [ $PLAN_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! No changes needed - all resources imported correctly"
elif [ $PLAN_EXIT_CODE -eq 2 ]; then
    echo "⚠️  Changes detected after import."
    echo "   This is NORMAL for first import - Terraform will adjust some attributes."
    echo "   Review the plan carefully and apply if it looks correct."
    echo ""
    echo "   Common differences to expect:"
    echo "   - IAM bindings added/removed (will be corrected)"
    echo "   - Label additions (managed_by=terraform)"
    echo "   - Lifecycle policies added"
    echo ""
    echo "   Next steps:"
    echo "   1. Review the plan above"
    echo "   2. If acceptable, run: terraform apply"
    echo "   3. After apply, run 'terraform plan' again - it should show 0 changes"
else
    echo "❌ terraform plan failed with exit code $PLAN_EXIT_CODE"
    exit 1
fi

echo ""
echo "Import script complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Review the terraform plan output above"
echo "   2. If plan looks correct, apply: terraform apply"
echo "   3. Verify 'terraform plan' shows 0 changes after apply"
echo "   4. Commit the .terraform directory and state to version control (if using local state)"
echo ""
