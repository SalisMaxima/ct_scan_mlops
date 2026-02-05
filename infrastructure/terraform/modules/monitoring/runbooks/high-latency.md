# High Latency Runbook

## Alert: Response Latency Exceeded Threshold

### Symptoms
- P95 latency > 5 seconds
- Timeout errors from clients
- Slow inference responses

### Immediate Actions
1. Check Cloud Run metrics for latency distribution
2. Identify if cold starts are contributing (check instance count)
3. Review concurrent request load
4. Check GCS read latency for model/config files

### Common Causes
- Cold starts (min instances = 0)
- Model inference time increased
- GCS cross-region latency
- CPU throttling under load

### Mitigation
1. **Cold starts**: Set minimum instances > 0
   ```bash
   gcloud run services update ct-scan-api --min-instances=1 --region=europe-west1
   ```
2. **CPU throttling**: Increase CPU allocation
3. **GCS latency**: Consider caching model in container

### Escalation
If latency persists after scaling adjustments, escalate for model optimization review.
