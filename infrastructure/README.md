# CT Scan MLOps Infrastructure

**Infrastructure as Code** and **Monitoring** implementation for the CT Scan MLOps project.

## Overview

This directory contains comprehensive Terraform modules and monitoring configurations to manage all GCP infrastructure as code, enabling:
- ✅ Disaster recovery through code
- ✅ Environment consistency (dev, staging, prod)
- ✅ Team collaboration via version control
- ✅ Comprehensive alerting and monitoring
- ✅ Automated deployments via CI/CD

## Directory Structure

```
infrastructure/
├── terraform/
│   ├── modules/              # Reusable Terraform modules
│   │   ├── storage/         # GCS buckets (DVC, models, state, drift logs)
│   │   ├── artifact-registry/ # Docker container registry
│   │   ├── cloud-run/       # API service deployment
│   │   ├── secret-manager/  # Secrets management (W&B, Slack, PagerDuty)
│   │   ├── iam/             # Service accounts and IAM bindings
│   │   ├── workload-identity/ # Keyless GitHub Actions authentication
│   │   ├── firestore/       # Feedback database (replaces SQLite)
│   │   ├── budget/          # Cost monitoring and alerts
│   │   └── monitoring/      # Comprehensive alerting (P0-P3 priorities)
│   ├── environments/
│   │   ├── dev/            # Development environment config
│   │   ├── staging/        # Staging environment config
│   │   └── prod/           # Production environment config (main reference)
│   └── shared/             # Project-level shared resources
├── scripts/
│   └── import-existing.sh  # Automated import of existing GCP resources
└── docs/
    ├── TERRAFORM_SETUP.md  # Complete setup guide
    ├── MIGRATION.md        # 8-week migration strategy
    └── RUNBOOK.md          # Day-to-day operations guide
```

## Quick Start

### Prerequisites
```bash
# Install Terraform >= 1.5.0
brew install terraform  # macOS
# OR
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip

# Authenticate with GCP
gcloud auth application-default login
gcloud config set project dtu-mlops-data-482907
```

### Initial Setup

1. **Create Terraform state bucket** (one-time):
   ```bash
   gsutil mb -l europe-west1 gs://dtu-mlops-terraform-state-482907
   gsutil versioning set on gs://dtu-mlops-terraform-state-482907
   ```

2. **Configure variables**:
   ```bash
   cd infrastructure/terraform/environments/prod
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your values
   ```

3. **Initialize Terraform**:
   ```bash
   invoke terraform.init --environment=prod
   ```

4. **Import existing resources**:
   ```bash
   invoke terraform.import-all --environment=prod
   ```

5. **Verify (should show 0 changes)**:
   ```bash
   invoke terraform.plan --environment=prod
   ```

## Using Invoke Tasks

All Terraform operations are available via invoke tasks:

```bash
# Planning and applying
invoke terraform.plan --environment=prod
invoke terraform.apply --environment=prod

# Viewing outputs
invoke terraform.output --environment=prod
invoke terraform.output --output-name=cloud_run_url

# State management
invoke terraform.state-list --environment=prod
invoke terraform.state-show --resource="module.cloud_run.google_cloud_run_v2_service.api"

# Validation and formatting
invoke terraform.validate --environment=prod
invoke terraform.format

# Importing resources
invoke terraform.import-resource \
  --resource="module.storage.google_storage_bucket.dvc" \
  --id="bucket-name" \
  --environment=prod

# Full check (validate + plan)
invoke terraform.check --environment=prod
```

## Resources Managed

### Storage
- **DVC Bucket:** `dtu-mlops-dvc-storage-482907` (1-year retention)
- **Models Bucket:** `dtu-mlops-data-482907_cloudbuild` (6-month retention)
- **Drift Logs Bucket:** `ct-scan-drift-logs-482907` (90-day retention)
- **Terraform State Bucket:** `dtu-mlops-terraform-state-482907` (10 versions)

### Container Registry
- **Artifact Registry:** `europe-west1-docker.pkg.dev/dtu-mlops-data-482907/ct-scan-mlops`
- **Cleanup Policy:** 90-day retention, keep minimum 5 versions

### Cloud Run
- **Service:** `ct-scan-api`
- **Resources:** 2 CPU, 4Gi RAM
- **Scaling:** 1-10 instances
- **Volume Mount:** Models bucket at `/models`

### IAM
- **GitHub Actions SA:** For CI/CD deployments
- **Cloud Run SA:** For API service execution
- **Drift Detection SA:** For scheduled drift detection
- **Monitoring SA:** For alerting and notifications

### Secrets
- **W&B API Key:** `wandb-api-key`
- **Slack Webhook:** `slack-webhook-token` (optional)
- **PagerDuty Key:** `pagerduty-integration-key` (optional)

### Database
- **Firestore:** Feedback storage (replaces SQLite for Cloud Run compatibility)
- **Retention:** 90-day TTL via expireAt field
- **Indexes:** Optimized queries for timestamp, prediction_id, accuracy

### Monitoring & Alerting

#### Critical Alerts (P0)
1. **API Downtime** - Health check fails for 1 minute → Email + PagerDuty
2. **High Error Rate** - >5% errors over 5 minutes → Email + Slack
3. **Memory Exhaustion** - >90% memory for 3 minutes → Email + Slack
4. **Container Crash Loop** - >3 restarts in 10 minutes → Email + PagerDuty

#### High Priority Alerts (P1)
5. **High Latency** - P95 >2s for 5 minutes → Email + Slack
6. **Data Drift** - >30% features drifted (daily check) → Email + Slack
7. **Low Confidence** - Avg confidence <0.7 over 100 predictions → Email

#### Medium Priority Alerts (P2)
8. **Elevated Error Rate** - >2% errors over 15 minutes
9. **Model Accuracy Degradation** - <85% accuracy from feedback
10. **High CPU Usage** - >80% for 10 minutes
11. **Disk Space Warning** - Feedback DB >500MB or logs >1GB

#### Runbooks
- All alerts have detailed runbooks with diagnostic and resolution steps
- Located in: `terraform/modules/monitoring/runbooks/`

### Budget
- **Monthly Budget:** $500 USD (configurable)
- **Alert Thresholds:** 50%, 90%, 100%, 110%
- **Notifications:** Pub/Sub topic for automation

## Module Documentation

Each module has a README with:
- Purpose and features
- Usage examples
- Import commands for existing resources
- Configuration options

### Example Module Usage

```hcl
module "storage" {
  source = "../../modules/storage"

  project_id     = "dtu-mlops-data-482907"
  region         = "europe-west1"

  dvc_bucket_name = "dtu-mlops-dvc-storage-482907"

  dvc_bucket_admins = [
    "serviceAccount:github-actions@project.iam.gserviceaccount.com"
  ]

  common_labels = {
    project     = "ct-scan-mlops"
    environment = "prod"
    managed_by  = "terraform"
  }
}
```

## 📍 Current Phase Status

**Last Updated:** 2026-02-02
**Current Phase:** Phase 2 - Import Existing Resources (JUST STARTED)
**Next Milestone:** Terraform state populated with all GCP resources

```
Phase 1: Terraform Foundation        ✅ COMPLETED (100%)
Phase 2: Import Resources            🔄 STARTING (0%)
Phase 3-8: Deployment & Monitoring   ⏸️  PENDING
```

**See detailed checkpoint:** `PHASE_CHECKPOINT.md`

## Implementation Status

### ✅ Completed (Phase 1: Terraform Foundation)
- [x] Infrastructure directory structure
- [x] All Terraform modules (9 modules, 27 .tf files)
- [x] Production environment configuration
- [x] Dev and staging environment configurations
- [x] Monitoring alert policies and runbooks
- [x] Import automation script
- [x] Comprehensive documentation (2,484 lines)
- [x] Invoke task integration (13 tasks)
- [x] All 7 critical issues fixed (Gemini review)
- [x] 3 bonus improvements applied

**Grade:** A (Gemini Architectural Review)
**Completed:** 2026-02-02

### 🔄 In Progress (Phase 2: Import & Validation)
- [ ] Terraform state bucket creation (manual prerequisite) - **NEXT**
- [ ] Configure terraform.tfvars with project values - **BLOCKED: awaiting user input**
- [ ] Import existing GCP resources (Week 2-3)
- [ ] Verify terraform plan shows 0 changes

### ⏳ Pending (Phase 3-8: Deployment & Monitoring)
- [ ] Firestore migration from SQLite (Week 4)
- [ ] Monitoring API endpoints (Week 4-5)
- [ ] Drift detection Cloud Function (Week 5)
- [ ] CI/CD GitHub Actions workflow (Week 4)
- [ ] Chaos testing implementation (Week 7-8)

## Migration Timeline

**Total Duration:** 8 weeks

```
Week 1: Terraform Foundation (COMPLETED)
Week 2-3: Import Existing Resources (CRITICAL - allow 2 weeks)
Week 3: Validation & Testing
Week 4: CI/CD + Firestore Migration
Week 5-6: Monitoring Shadow Mode (P1/P2 tuning)
Week 7: Monitoring Production (all alerts enabled)
Week 8: Finalization & Training
```

See [MIGRATION.md](docs/MIGRATION.md) for detailed strategy.

## Key Features

### 🔒 Security
- Secrets managed via Secret Manager (values NOT in Terraform state)
- Workload Identity Federation for keyless GitHub Actions
- Service account least-privilege IAM
- Uniform bucket-level access

### 💰 Cost Management
- Budget alerts at multiple thresholds
- Automatic lifecycle policies on all buckets
- Image retention policies in Artifact Registry
- Resource right-sizing

### 🔍 Observability
- 11 alert policies across 4 priority levels
- Notification routing (email, Slack, PagerDuty)
- Comprehensive runbooks for incident response
- Dashboard as code (ready for deployment)

### 🔄 Disaster Recovery
- Terraform state versioning (10 versions)
- Bucket versioning on critical data
- Point-in-time recovery for Firestore
- Documented rollback procedures

### 🏗️ Architecture Decisions

#### Why Firestore (not SQLite)?
- Cloud Run is stateless - SQLite files lost on restart
- Firestore is serverless, managed, supports concurrent writes
- No code changes needed, just database adapter swap

#### Why Separate Directories (not Workspaces)?
- Clearer state isolation (prevents accidental prod changes)
- Different backend configs per environment
- Easier team collaboration and PR reviews

#### Why Shadow Mode for P1/P2 Alerts?
- Tune thresholds with real production data
- Avoid alert fatigue from false positives
- Achieve <2% false positive rate before enabling notifications

## Documentation

### For Getting Started
- **[TERRAFORM_SETUP.md](docs/TERRAFORM_SETUP.md)** - Complete setup guide with prerequisites, configuration, and troubleshooting

### For Migration
- **[MIGRATION.md](docs/MIGRATION.md)** - 8-week phased migration strategy with rollback procedures

### For Daily Operations
- **[RUNBOOK.md](docs/RUNBOOK.md)** - Common scenarios, emergency procedures, maintenance tasks

### For Incident Response
- **Alert Runbooks:** `terraform/modules/monitoring/runbooks/`
  - api-downtime.md
  - high-error-rate.md
  - memory-exhaustion.md
  - crash-loop.md
  - high-latency.md
  - elevated-error-rate.md

## CI/CD Integration

### Planned GitHub Actions Workflow
- ✅ Auto-plan on pull requests
- ✅ Post plan as PR comment
- ✅ Auto-apply to dev on merge
- ✅ Manual approval for prod
- ✅ OIDC authentication (no long-lived credentials)

Workflow file: `.github/workflows/terraform.yml` (to be created)

## Next Steps

1. **Week 1 (Now):**
   - [ ] Review implementation with team
   - [ ] Create Terraform state bucket
   - [ ] Configure `terraform.tfvars` with actual values
   - [ ] Initialize Terraform

2. **Week 2-3:**
   - [ ] Run import script
   - [ ] Fix noisy diffs iteratively
   - [ ] Achieve `terraform plan` = 0 changes

3. **Week 4:**
   - [ ] Create GitHub Actions workflow
   - [ ] Migrate SQLite to Firestore
   - [ ] Deploy monitoring foundation

4. **Week 5-8:**
   - [ ] Tune alerts in shadow mode
   - [ ] Enable all alerts with notifications
   - [ ] Conduct chaos testing
   - [ ] Train team

## Getting Help

- **Documentation:** Start with [TERRAFORM_SETUP.md](docs/TERRAFORM_SETUP.md)
- **Issues:** Check [RUNBOOK.md](docs/RUNBOOK.md) troubleshooting section
- **Team:** Slack #infrastructure channel
- **Urgent:** Follow incident response procedures in alert runbooks

## Contributing

When making infrastructure changes:
1. Create feature branch
2. Make changes to Terraform files
3. Run `invoke terraform.format`
4. Run `invoke terraform.validate`
5. Test in dev environment first
6. Create PR with plan output
7. After approval, apply to prod

## License

Internal project - DTU MLOps Course

## Acknowledgments

Architecture designed based on:
- Gemini Pro architectural review
- GCP best practices
- Terraform registry documentation
- Production incident learnings
