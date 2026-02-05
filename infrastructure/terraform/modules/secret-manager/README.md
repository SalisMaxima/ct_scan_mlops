# Secret Manager Module

Manages secrets for CT Scan MLOps project.

## ⚠️ IMPORTANT: Secret Value Management

**This module creates secret PLACEHOLDERS only. Secret values are NOT stored in Terraform state.**

Secret values must be injected separately via:
- GitHub Actions secrets
- `gcloud` CLI commands
- Google Cloud Console

## Secrets Created

1. **W&B API Key** (`wandb-api-key`) - For Weights & Biases authentication
2. **Slack Webhook** (`slack-webhook-token`) - Optional, for Slack notifications
3. **PagerDuty Key** (`pagerduty-integration-key`) - Optional, for PagerDuty alerts
4. **Generic Secrets** - Custom secrets as needed

## Usage

```hcl
module "secret_manager" {
  source = "./modules/secret-manager"

  project_id = "dtu-mlops-data-482907"

  wandb_secret_accessors = [
    "serviceAccount:github-actions@dtu-mlops-data-482907.iam.gserviceaccount.com",
    "serviceAccount:cloud-run@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  create_slack_webhook_secret = true
  slack_webhook_accessors = [
    "serviceAccount:monitoring@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  create_pagerduty_secret = true
  pagerduty_accessors = [
    "serviceAccount:monitoring@dtu-mlops-data-482907.iam.gserviceaccount.com"
  ]

  generic_secrets = {
    "api-auth-token" = {
      accessors = [
        "serviceAccount:api@dtu-mlops-data-482907.iam.gserviceaccount.com"
      ]
    }
  }

  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
  }
}
```

## Import Existing Secrets

To import the existing W&B API key secret:

```bash
terraform import module.secret_manager.google_secret_manager_secret.wandb_api_key \
  projects/dtu-mlops-data-482907/secrets/wandb-api-key
```

## Adding Secret Values

### Via gcloud CLI

```bash
# Add W&B API key
echo -n "your-wandb-api-key" | gcloud secrets versions add wandb-api-key --data-file=-

# Add Slack webhook token
echo -n "your-slack-webhook-url" | gcloud secrets versions add slack-webhook-token --data-file=-

# Add PagerDuty integration key
echo -n "your-pagerduty-key" | gcloud secrets versions add pagerduty-integration-key --data-file=-
```

### Via GitHub Actions

```yaml
- name: Add secret to Secret Manager
  run: |
    echo -n "${{ secrets.WANDB_API_KEY }}" | \
      gcloud secrets versions add wandb-api-key --data-file=-
```

## Secret Rotation Strategy

1. **Manual Rotation** (Recommended for low-frequency rotation)
   - Create new secret version via gcloud or Console
   - Old versions are automatically disabled after new version is active
   - Delete old versions after validation period

2. **Automated Rotation** (For high-frequency rotation)
   - Use Secret Manager rotation feature (requires Cloud Functions)
   - Implement rotation handler function
   - Configure rotation schedule

Example rotation:
```bash
# Add new version
echo -n "new-api-key" | gcloud secrets versions add wandb-api-key --data-file=-

# Verify new version works (test API deployment)

# Disable old version
gcloud secrets versions disable 1 --secret=wandb-api-key

# Delete old version (after retention period)
gcloud secrets versions destroy 1 --secret=wandb-api-key
```

## Accessing Secrets in Applications

### Cloud Run

```yaml
env:
  - name: WANDB_API_KEY
    valueFrom:
      secretKeyRef:
        name: wandb-api-key
        key: latest
```

### Python Application

```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(request={"name": name})
secret_value = response.payload.data.decode("UTF-8")
```

## Security Best Practices

1. **Never commit secret values** to version control
2. **Use service accounts** for programmatic access (no user credentials)
3. **Enable audit logging** for secret access
4. **Rotate secrets regularly** (quarterly minimum)
5. **Use IAM conditions** for fine-grained access control
6. **Monitor secret access** via Cloud Logging
