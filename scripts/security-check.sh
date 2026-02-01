#!/bin/bash
# Security check hook for Claude Code
# Runs BEFORE Bash commands execute - exit 2 to block
#
# Receives JSON on stdin with tool_input.command

set -euo pipefail

# Read JSON from stdin
INPUT=$(cat)

# Extract command from JSON
COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cmd = data.get('tool_input', {}).get('command', '')
    print(cmd)
except:
    pass
" 2>/dev/null)

# If we couldn't extract command, allow (fail open for non-Bash tools)
if [[ -z "$COMMAND" ]]; then
    exit 0
fi

# =============================================================================
# BLOCKED PATTERNS
# =============================================================================

# --- Destructive File Operations ---
# Block rm -rf on root, home, or without path safety
if echo "$COMMAND" | grep -qE "rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|(-[a-zA-Z]+\s+)*)(\/|~|\.\.|\/home|\/etc|\/usr|\/var|\/root)\b"; then
    echo "❌ BLOCKED: Destructive rm command targeting sensitive directory"
    echo "   Command: $COMMAND"
    echo "   Tip: Use specific paths like ./build/ instead of /"
    exit 2
fi

# Block dd writing to block devices
if echo "$COMMAND" | grep -qE "dd\s+.*of=\/dev\/(sd|hd|nvme|vd)"; then
    echo "❌ BLOCKED: dd write to block device"
    exit 2
fi

# Block mkfs (formatting)
if echo "$COMMAND" | grep -qE "mkfs"; then
    echo "❌ BLOCKED: Filesystem format command"
    exit 2
fi

# --- Credential Exposure ---
# Block reading common secret files
if echo "$COMMAND" | grep -qE "(cat|head|tail|less|more|bat)\s+.*(id_rsa|id_ed25519|\.env|\.secret|credentials|\.pem|\.key|password|token|api.key)"; then
    echo "❌ BLOCKED: Potential credential exposure"
    echo "   Command: $COMMAND"
    exit 2
fi

# Block echoing environment secrets
if echo "$COMMAND" | grep -qE "echo\s+.*\\\$(API_KEY|SECRET|PASSWORD|TOKEN|PRIVATE|CREDENTIAL)"; then
    echo "❌ BLOCKED: Echoing sensitive environment variable"
    exit 2
fi

# Block printenv/env for secrets
if echo "$COMMAND" | grep -qE "(printenv|env)\s*\|.*grep.*(KEY|SECRET|TOKEN|PASSWORD)"; then
    echo "❌ BLOCKED: Extracting secrets from environment"
    exit 2
fi

# --- Dangerous Git Operations ---
# Block force push to main/master
if echo "$COMMAND" | grep -qE "git\s+push\s+.*(-f|--force).*\s+(origin\s+)?(main|master)\b"; then
    echo "❌ BLOCKED: Force push to main/master"
    echo "   This can destroy team history. Use a PR workflow instead."
    exit 2
fi

# Block git reset --hard without specific ref
if echo "$COMMAND" | grep -qE "git\s+reset\s+--hard\s*$"; then
    echo "❌ BLOCKED: git reset --hard without specific ref"
    echo "   Tip: Specify a commit or use 'git stash' to save changes"
    exit 2
fi

# Block git clean -fd on root
if echo "$COMMAND" | grep -qE "git\s+clean\s+-[a-zA-Z]*f[a-zA-Z]*d"; then
    echo "⚠️  WARNING: git clean -fd removes untracked files permanently"
    # Allow but warn - not blocking as it's sometimes needed
fi

# --- Network Exfiltration ---
# Block curl/wget POSTing local files to external servers
if echo "$COMMAND" | grep -qE "(curl|wget|http)\s+.*(-X\s*POST|-d\s*@|--data-binary\s*@|--upload-file)"; then
    echo "❌ BLOCKED: Uploading local files to external server"
    echo "   Command: $COMMAND"
    exit 2
fi

# Block netcat reverse shells
if echo "$COMMAND" | grep -qE "(nc|netcat|ncat)\s+.*-e\s*(\/bin\/(ba)?sh|sh)"; then
    echo "❌ BLOCKED: Potential reverse shell"
    exit 2
fi

# --- Resource Exhaustion ---
# Block fork bombs
if echo "$COMMAND" | grep -qE ":\(\)\s*\{.*:\|:.*\}"; then
    echo "❌ BLOCKED: Fork bomb detected"
    exit 2
fi

# Block infinite loops without limits
if echo "$COMMAND" | grep -qE "while\s+(true|1|:)\s*;\s*do.*done" | grep -vqE "(sleep|break|exit)"; then
    echo "⚠️  WARNING: Infinite loop detected - ensure it has an exit condition"
fi

# --- Dangerous Permission Changes ---
# Block chmod 777 on sensitive directories
if echo "$COMMAND" | grep -qE "chmod\s+(-R\s+)?777\s+(\/|~|\.\.|\.)"; then
    echo "❌ BLOCKED: chmod 777 on sensitive directory"
    echo "   Tip: Use more restrictive permissions (755 for dirs, 644 for files)"
    exit 2
fi

# Block chown to root
if echo "$COMMAND" | grep -qE "chown\s+(-R\s+)?root"; then
    echo "❌ BLOCKED: Changing ownership to root"
    exit 2
fi

# --- System Modification ---
# Block modifying system files
if echo "$COMMAND" | grep -qE "(echo|cat|tee)\s+.*>\s*\/etc\/"; then
    echo "❌ BLOCKED: Writing to /etc/"
    exit 2
fi

# Block adding cron jobs
if echo "$COMMAND" | grep -qE "crontab|\/etc\/cron"; then
    echo "❌ BLOCKED: Modifying cron jobs"
    exit 2
fi

# Block systemctl/service modifications
if echo "$COMMAND" | grep -qE "(systemctl|service)\s+(enable|disable|mask|stop)"; then
    echo "❌ BLOCKED: Modifying system services"
    exit 2
fi

# --- Docker Dangers ---
# Block docker run with privileged mode
if echo "$COMMAND" | grep -qE "docker\s+run\s+.*--privileged"; then
    echo "❌ BLOCKED: Docker privileged mode"
    exit 2
fi

# Block mounting host root in docker
if echo "$COMMAND" | grep -qE "docker\s+run\s+.*-v\s+\/:\/?"; then
    echo "❌ BLOCKED: Mounting host root in Docker"
    exit 2
fi

# --- Python/Pip Dangers ---
# Block pip install from untrusted sources
if echo "$COMMAND" | grep -qE "pip\s+install\s+.*--index-url\s+(?!https://(pypi\.org|files\.pythonhosted\.org))"; then
    echo "⚠️  WARNING: pip install from non-PyPI source"
fi

# =============================================================================
# PASSED ALL CHECKS
# =============================================================================
exit 0
