#!/bin/bash
################################################################################
# Check Tier 1 Progress
################################################################################
#
# Quick script to check if Tier 1 experiments are complete
#
# Usage: ./scripts/check_tier1_progress.sh
#
################################################################################

LOG_DIR="/Users/dkMatHLu/Desktop/ct_scan_mlops/logs/tier1_20260129_152548"  # pragma: allowlist secret

echo "=================================="
echo "TIER 1 PROGRESS CHECK"
echo "=================================="
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    exit 1
fi

# Count completed experiments
completed=$(grep -c "COMPLETED ✓" "$LOG_DIR/results_summary.txt" 2>/dev/null || echo 0)
failed=$(grep -c "FAILED ✗" "$LOG_DIR/results_summary.txt" 2>/dev/null || echo 0)
total=7

echo "📊 STATUS:"
echo "  Completed: $completed / $total"
echo "  Failed: $failed"
echo ""

# Check if all done
if [ "$completed" -eq "$total" ]; then
    echo "✅ ALL EXPERIMENTS COMPLETE!"
    echo ""
    echo "Run analysis:"
    echo "  uv run python scripts/analyze_tier1_results.py"
    echo ""
    exit 0
fi

# Show current experiment
current_log=$(ls -t "$LOG_DIR"/T1_*.log 2>/dev/null | head -1)
if [ -n "$current_log" ]; then
    current_name=$(basename "$current_log" .log)
    echo "🔄 CURRENT EXPERIMENT: $current_name"

    # Check epoch progress
    last_epoch=$(grep -oE "Epoch [0-9]+/[0-9]+" "$current_log" 2>/dev/null | tail -1)
    if [ -n "$last_epoch" ]; then
        echo "  Progress: $last_epoch"
    fi
    echo ""
fi

# Estimate time remaining
if [ "$completed" -gt 0 ] && [ -f "$LOG_DIR/durations.txt" ]; then
    avg_time=$(awk '{sum+=$NF; count++} END {print sum/count}' "$LOG_DIR/durations.txt" 2>/dev/null)
    if [ -n "$avg_time" ]; then
        remaining=$((total - completed))
        est_min=$(echo "$avg_time * $remaining" | bc 2>/dev/null)
        echo "⏱️  ESTIMATE:"
        echo "  Average time per experiment: ${avg_time} min"
        echo "  Remaining experiments: $remaining"
        echo "  Estimated time remaining: ~${est_min} min"
        echo ""
    fi
fi

# Show W&B link
echo "📈 MONITOR ON W&B:"
echo "  https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps"
echo ""

echo "Run this script again to check progress"
echo ""
