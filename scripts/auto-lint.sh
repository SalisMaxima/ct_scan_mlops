#!/bin/bash
# Auto-lint hook for Claude Code
# Runs ruff on Python files after Edit/Write operations
# Receives JSON on stdin with tool_input.file_path

# Read JSON from stdin
INPUT=$(cat)

# Extract file path from JSON (handles both Edit and Write tool formats)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Try tool_input.file_path first, then direct file_path
    path = data.get('tool_input', {}).get('file_path', data.get('file_path', ''))
    print(path)
except:
    pass
" 2>/dev/null)

# Only run on Python files
if [[ "$FILE_PATH" == *.py ]]; then
    # Run ruff check with auto-fix (quiet mode, only show if issues)
    uv run ruff check "$FILE_PATH" --fix --quiet 2>/dev/null
    # Run ruff format (quiet mode)
    uv run ruff format "$FILE_PATH" --quiet 2>/dev/null

    # Exit 0 to allow the operation to proceed
    exit 0
fi

# Non-Python files: do nothing, allow operation
exit 0
