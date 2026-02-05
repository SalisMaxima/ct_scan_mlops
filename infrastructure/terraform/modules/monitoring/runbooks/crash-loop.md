# Crash Loop Runbook

## Alert: Container Crash Loop Detected

### Symptoms
- Container repeatedly starting and crashing
- Service unavailable or intermittently responding
- Multiple revision failures in Cloud Run

### Immediate Actions
1. Check container logs for crash reason
2. Identify the failing revision: `gcloud run revisions list --service=ct-scan-api --region=europe-west1`
3. Check startup probe failures
4. Verify environment variables and secrets are correctly set

### Common Causes
- Missing or invalid environment variables
- Model file not found at expected path
- GCS FUSE mount failure
- Python dependency import errors
- Insufficient startup time for model loading

### Rollback Command
```bash
gcloud run services update-traffic ct-scan-api --to-revisions=STABLE_REVISION=100 --region=europe-west1
```

### Escalation
If crash continues after rollback, escalate for infrastructure review.
