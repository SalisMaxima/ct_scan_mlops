# Terraform Operations Runbook

Day-to-day operational guide for managing Terraform infrastructure.

## Quick Reference

```bash
# Common commands
invoke terraform.plan --environment=prod
invoke terraform.apply --environment=prod
invoke terraform.output --environment=prod
invoke terraform.state-list --environment=prod

# Emergency rollback
cd infrastructure/terraform/environments/prod
terraform plan -target=module.cloud_run
terraform apply -target=module.cloud_run
```

## Daily Operations

### 1. Checking Infrastructure State

**View all managed resources:**
```bash
invoke terraform.state-list --environment=prod
```

**Check for drift:**
```bash
invoke terraform.plan --environment=prod
```

**Expected:** "No changes. Your infrastructure matches the configuration."

**If drift detected:** Review the changes. Common causes:
- Manual changes in GCP console
- Auto-scaling changes
- GCP API defaults changing

### 2. Making Changes

**Process:**
1. Edit Terraform files or `terraform.tfvars`
2. Run plan to preview changes
3. Review changes carefully
4. Apply if acceptable
5. Verify in GCP console

**Example - Scale Cloud Run:**
```hcl
# Edit terraform.tfvars
max_instances = 20
memory_limit  = "8Gi"
```

```bash
terraform plan
# Review: Should show updates to Cloud Run service
terraform apply
```

### 3. Viewing Outputs

**All outputs:**
```bash
invoke terraform.output --environment=prod
```

**Specific output:**
```bash
invoke terraform.output --environment=prod --output-name=cloud_run_url
# Use in scripts:
API_URL=$(cd infrastructure/terraform/environments/prod && terraform output -raw cloud_run_url)
```

## Common Scenarios

### Scenario 1: Deploy New Container Image

**Context:** New Docker image built and pushed to Artifact Registry

**Steps:**
1. Update image tag in `terraform.tfvars`:
   ```hcl
   container_image = "europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops/api:v2.1.0"
   ```

2. Plan and apply:
   ```bash
   cd infrastructure/terraform/environments/prod
   terraform plan
   # Review: Should show Cloud Run service update
   terraform apply
   ```

3. Verify deployment:
   ```bash
   curl "$(terraform output -raw cloud_run_url)/health"
   ```

**Rollback if needed:**
```bash
# Update to previous image tag
# Re-run terraform apply
```

### Scenario 2: Add New Environment Variable

**Steps:**
1. Edit `main.tf`:
   ```hcl
   module "cloud_run" {
     # ...
     environment_variables = {
       PROJECT_ID = var.project_id
       REGION     = var.region
       NEW_VAR    = "value"  # Add this
     }
   }
   ```

2. Apply:
   ```bash
   terraform apply
   ```

### Scenario 3: Add New Secret

**Steps:**
1. Create secret placeholder in Terraform:
   ```hcl
   module "secret_manager" {
     # ...
     generic_secrets = {
       "api-key" = {
         accessors = ["serviceAccount:my-sa@project.iam.gserviceaccount.com"]
       }
     }
   }
   ```

2. Apply Terraform:
   ```bash
   terraform apply
   ```

3. **Separately** add secret value (NOT in Terraform):
   ```bash
   echo -n "actual-secret-value" | gcloud secrets versions add api-key --data-file=-
   ```

4. Grant Cloud Run access to secret:
   ```hcl
   secret_environment_variables = {
     API_KEY = {
       secret_name = "api-key"  # pragma: allowlist secret
       version     = "latest"
     }
   }
   ```

### Scenario 4: Update Alert Thresholds

**Steps:**
1. Edit `main.tf` monitoring module:
   ```hcl
   module "monitoring" {
     # ...
     high_error_rate_threshold = 0.03  # Change from 0.05 to 0.03
   }
   ```

2. Apply:
   ```bash
   terraform apply
   ```

3. Verify alert policy updated:
   ```bash
   gcloud alpha monitoring policies list --project=dtu-mlops-data-482907
   ```

### Scenario 5: Scale Budget

**Steps:**
1. Edit `terraform.tfvars`:
   ```hcl
   monthly_budget_amount = 750  # Increase from 500
   ```

2. Apply:
   ```bash
   terraform apply
   ```

3. Verify budget:
   ```bash
   gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID
   ```

### Scenario 6: Enable P1/P2 Alerts (Post Shadow Mode)

**Steps:**
1. After 1 week of shadow mode monitoring, enable alerts:
   ```hcl
   module "monitoring" {
     # ...
     enable_p1_p2_alerts = true  # Change from false
   }
   ```

2. Apply:
   ```bash
   terraform apply
   ```

3. Test alert delivery:
   ```bash
   # Trigger test alert (use staging or controlled prod test)
   ```

## Emergency Procedures

### Emergency 1: Rollback Cloud Run Deployment

**Scenario:** New deployment causes API errors

**Immediate Action (Fastest - Use GCP CLI):**
```bash
# List revisions
gcloud run revisions list \
  --service=ct-scan-api \
  --region=europe-west1 \
  --project=dtu-mlops-data-482907

# Rollback to previous revision
gcloud run services update-traffic ct-scan-api \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region=europe-west1 \
  --project=dtu-mlops-data-482907
```

**Fix in Terraform:**
```hcl
# Update terraform.tfvars to previous image
container_image = "...previous-version..."
```

```bash
terraform apply  # Update Terraform state to match reality
```

### Emergency 2: State Lock Stuck

**Scenario:** Terraform commands hang with "Acquiring state lock..."

**Solution:**
```bash
# View lock info
gsutil cat gs://dtu-mlops-terraform-state-482907/environments/prod/default.tflock

# If safe to unlock (confirm no other applies running):
cd infrastructure/terraform/environments/prod
terraform force-unlock LOCK_ID
```

### Emergency 3: State File Corrupted

**Scenario:** Terraform commands fail with state errors

**Solution:**
```bash
# List state file versions
gsutil ls -la gs://dtu-mlops-terraform-state-482907/environments/prod/

# Restore previous version
gsutil cp gs://dtu-mlops-terraform-state-482907/environments/prod/default.tfstate#VERSION \
  ./terraform.tfstate.backup

# Copy to current state
gsutil cp ./terraform.tfstate.backup \
  gs://dtu-mlops-terraform-state-482907/environments/prod/default.tfstate

# Verify
terraform plan
```

### Emergency 4: Resource Deleted in GCP (Not Terraform)

**Scenario:** Someone manually deleted a resource in GCP console

**Solution:**
```bash
# Remove from Terraform state (it's already gone)
terraform state rm "module.storage.google_storage_bucket.drift_logs"

# Recreate with Terraform
terraform plan
# Should show creation of missing resource
terraform apply
```

## Maintenance Tasks

### Weekly: Check for Drift

```bash
cd infrastructure/terraform/environments/prod
terraform plan

# If drift detected:
# 1. Investigate cause
# 2. Fix in GCP or Terraform
# 3. Re-import or apply as needed
```

### Monthly: Update Terraform Provider

```bash
# Update provider version in main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.1"  # Update version
    }
  }
}

# Re-initialize
terraform init -upgrade

# Test in dev first
cd ../dev
terraform init -upgrade
terraform plan
terraform apply

# If successful, apply to prod
cd ../prod
terraform init -upgrade
terraform plan
terraform apply
```

### Quarterly: Review Alert Thresholds

Based on 3 months of production data:
1. Analyze false positive rate
2. Adjust thresholds if needed
3. Update in Terraform
4. Deploy with `terraform apply`

### Quarterly: Budget Review

```bash
# Check actual spend vs budget
gcloud billing accounts get-spend --billing-account=BILLING_ACCOUNT_ID

# Adjust budget if needed in terraform.tfvars
```

## Troubleshooting

### Issue: "Error acquiring state lock"

**Cause:** Concurrent Terraform operations

**Solution:** Wait for other operation to complete, or force-unlock if stale

### Issue: "Resource already exists"

**Cause:** Resource exists in GCP but not in Terraform state

**Solution:** Import the resource

```bash
terraform import "module.X.resource" "resource-id"
```

### Issue: Plan shows unexpected changes

**Cause:** Drift between Terraform state and reality

**Investigation:**
```bash
# Compare Terraform state with actual resource
terraform state show "module.cloud_run.google_cloud_run_v2_service.api"
gcloud run services describe ct-scan-api --region=europe-west1 --format=json

# If difference is acceptable, apply
# If not, adjust Terraform config
```

### Issue: Apply fails with permission error

**Cause:** Insufficient IAM permissions

**Solution:**
```bash
# Check current permissions
gcloud projects get-iam-policy dtu-mlops-data-482907 \
  --flatten="bindings[].members" \
  --filter="bindings.members:YOUR_USER"

# Grant required role (requires admin)
gcloud projects add-iam-policy-binding dtu-mlops-data-482907 \
  --member="user:YOUR_EMAIL" \
  --role="roles/editor"
```

## Best Practices

### Do's ✅
- Always run `terraform plan` before `apply`
- Review plan output carefully
- Test changes in dev environment first
- Use version control for all Terraform changes
- Document all manual changes
- Keep terraform.tfvars in sync with reality

### Don'ts ❌
- Never manually edit state files
- Never use `--auto-approve` in prod without reviewing plan
- Never commit secrets to version control
- Never run concurrent `terraform apply` commands
- Never delete state bucket or files

## Useful Commands Reference

```bash
# Plan and apply
terraform plan
terraform apply
terraform apply -auto-approve

# Targeting specific resources
terraform plan -target=module.cloud_run
terraform apply -target=module.cloud_run

# State management
terraform state list
terraform state show "resource"
terraform state rm "resource"
terraform state mv "source" "destination"

# Outputs
terraform output
terraform output -raw output_name

# Import
terraform import "resource" "id"

# Validation and formatting
terraform validate
terraform fmt -recursive

# Workspace (not used - we use directories)
# terraform workspace list
```

## Contacts

- **Terraform Issues:** Team Slack #infrastructure
- **GCP Issues:** Cloud console support
- **Urgent/P0:** Page on-call via PagerDuty (if configured)

## Related Documentation

- [Setup Guide](./TERRAFORM_SETUP.md) - Initial setup
- [Migration Guide](./MIGRATION.md) - Migration strategy
- [Monitoring Runbooks](../terraform/modules/monitoring/runbooks/) - Alert response
