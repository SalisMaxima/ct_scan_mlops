# Elevated Error Rate Runbook

## Alert: Error Rate Above Normal (Warning)

### Symptoms
- Error rate between 1-5% (warning threshold)
- Intermittent failures on some requests
- Partial service degradation

### Immediate Actions
1. Monitor trend - is it increasing or stable?
2. Check recent deployments or config changes
3. Review error logs for patterns
4. Check external dependencies (GCS, any external APIs)

### Common Causes
- Malformed input requests from specific clients
- Transient GCS connectivity issues
- Resource contention during traffic spikes
- Partial model loading issues

### Mitigation
- If client-side: Add input validation, return helpful error messages
- If resource-related: Adjust scaling or resource limits
- If transient: Monitor and document for pattern analysis

### Escalation
Escalate if error rate trends toward 5% or persists > 30 minutes.
