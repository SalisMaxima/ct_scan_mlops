# API Downtime Runbook

## Alert: CT Scan API Service Down

### Symptoms
- Cloud Run service not responding to health checks
- HTTP 502/503 errors on all endpoints

### Immediate Actions
1. Check Cloud Run service status: `gcloud run services describe ct-scan-api --region=europe-west1`
2. Check recent logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ct-scan-api" --limit=50`
3. Verify container image exists in Artifact Registry
4. Check if service account has required permissions

### Rollback
If recent deployment caused the issue:
```bash
gcloud run services update-traffic ct-scan-api --to-revisions=PREVIOUS_REVISION=100 --region=europe-west1
```

### Escalation
If unresolved after 15 minutes, escalate to on-call engineer.
