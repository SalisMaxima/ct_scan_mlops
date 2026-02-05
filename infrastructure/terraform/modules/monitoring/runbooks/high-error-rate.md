# High Error Rate Runbook

## Alert: Error Rate Exceeded Threshold

### Symptoms
- Error rate > 5% over 5 minutes
- Increased 4xx/5xx HTTP responses

### Immediate Actions
1. Check error logs: `gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=100`
2. Identify error patterns (model loading, GCS access, memory issues)
3. Check if traffic spike is causing resource exhaustion
4. Verify GCS bucket accessibility and model file integrity

### Common Causes
- Model file corrupted or missing
- GCS bucket permissions changed
- Memory limit exceeded during inference
- Invalid input data format

### Mitigation
- If model issue: redeploy with known-good model checkpoint
- If memory issue: increase Cloud Run memory limits
- If traffic spike: adjust max instances scaling

### Escalation
If error rate persists > 10% for 10 minutes, escalate immediately.
