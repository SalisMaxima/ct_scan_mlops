# Memory Exhaustion Runbook

## Alert: Memory Usage Critical

### Symptoms
- Container memory usage > 85% threshold
- OOMKilled container restarts
- Slow response times or timeouts

### Immediate Actions
1. Check current memory usage in Cloud Run metrics
2. Review recent changes to model size or batch processing
3. Check for memory leaks in application logs
4. Verify concurrent request handling

### Mitigation
1. **Short-term**: Increase memory limit in Cloud Run config
2. **Medium-term**: Implement request queuing or reduce concurrency
3. **Long-term**: Optimize model memory footprint or implement streaming

### Configuration Change
```bash
gcloud run services update ct-scan-api --memory=8Gi --region=europe-west1
```

### Escalation
If OOMKilled events continue after memory increase, escalate for code review.
