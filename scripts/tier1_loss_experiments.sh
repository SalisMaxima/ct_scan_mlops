#!/bin/bash
################################################################################
# Tier 1: Loss Function Experiments
################################################################################
#
# Quick test of different loss functions to improve adenocarcinoma classification
# Focus: Better handling of hard cases and reducing overconfidence
#
# Estimated runtime: ~48 minutes (7 experiments × ~7 min each)
#
# Usage:
#   chmod +x scripts/tier1_loss_experiments.sh
#   ./scripts/tier1_loss_experiments.sh
#
# Or run in background:
#   nohup ./scripts/tier1_loss_experiments.sh > logs/tier1_run.log 2>&1 &
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

################################################################################
# Configuration
################################################################################

PROJECT_DIR="/Users/dkMatHLu/Desktop/ct_scan_mlops"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/tier1_$TIMESTAMP"
COOLDOWN=60  # 1 minute cooldown between experiments (shorter since they're fast)

################################################################################
# Setup
################################################################################

cd "$PROJECT_DIR" || exit 1

# Activate environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    exit 1
fi

source .venv/bin/activate

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize results summary
RESULTS_FILE="$LOG_DIR/results_summary.txt"
touch "$RESULTS_FILE"

################################################################################
# Helper Functions
################################################################################

print_header() {
    local text="$1"
    echo ""
    echo "$(printf '=%.0s' $(seq 1 78))"
    echo "$text"
    echo "$(printf '=%.0s' $(seq 1 78))"
}

print_section() {
    local text="$1"
    echo ""
    echo "$(printf -- '-%.0s' $(seq 1 78))"
    echo "$text"
    echo "$(printf -- '-%.0s' $(seq 1 78))"
}

run_experiment() {
    local name=$1
    local args=$2
    local log_file="$LOG_DIR/${name}.log"

    print_section "EXPERIMENT: $name"
    echo "Time: $(date)"
    echo "Args: $args"
    echo ""

    local start_time=$(date +%s)

    # Run training
    if invoke train --args "$args" 2>&1 | tee "$log_file"; then
        echo "$name: COMPLETED ✓" >> "$RESULTS_FILE"
        echo "Status: SUCCESS ✓"
    else
        local exit_code=$?
        echo "$name: FAILED ✗ (exit code: $exit_code)" >> "$RESULTS_FILE"
        echo "WARNING: Experiment $name failed with exit code $exit_code"
        echo "Continuing to next experiment..."
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))

    echo "Duration: ${duration_min} min $((duration % 60)) sec"
    echo "$name: ${duration_min} min" >> "$LOG_DIR/durations.txt"

    echo ""
    echo "Cooling down for $COOLDOWN seconds..."
    sleep $COOLDOWN
}

################################################################################
# Main Experiment Suite
################################################################################

print_header "TIER 1: LOSS FUNCTION EXPERIMENTS"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo ""
echo "Objective: Test different loss functions to:"
echo "  - Better handle hard adenocarcinoma cases"
echo "  - Reduce overconfidence (current: 83.3% on errors)"
echo "  - Improve calibration"
echo ""
echo "Estimated runtime: ~48 minutes (7 experiments)"
print_header ""

################################################################################
# Baseline Configuration
################################################################################

BASE_CONFIG="model=dual_pathway_top_features features=top_features \
train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
data.batch_size=32 train.max_epochs=25 train.profiling.enabled=false"

################################################################################
# Experiments
################################################################################

echo "Running 7 loss function experiments..."
echo ""

# 1. Baseline with standard cross-entropy
print_section "Experiment 1/7: Baseline (Cross-Entropy)"
run_experiment "T1_baseline_crossentropy" \
    "$BASE_CONFIG"

# 2. Focal Loss gamma=2.0 (recommended starting point)
print_section "Experiment 2/7: Focal Loss (gamma=2.0)"
run_experiment "T1_focal_loss_gamma2.0" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0"

# 3. Focal Loss gamma=2.5
print_section "Experiment 3/7: Focal Loss (gamma=2.5)"
run_experiment "T1_focal_loss_gamma2.5" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.5"

# 4. Focal Loss gamma=3.0 (more aggressive)
print_section "Experiment 4/7: Focal Loss (gamma=3.0)"
run_experiment "T1_focal_loss_gamma3.0" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=3.0"

# 5. Label Smoothing 0.1
print_section "Experiment 5/7: Label Smoothing (0.1)"
run_experiment "T1_label_smoothing_0.1" \
    "$BASE_CONFIG train.loss.type=label_smoothing train.loss.smoothing=0.1"

# 6. Label Smoothing 0.15
print_section "Experiment 6/7: Label Smoothing (0.15)"
run_experiment "T1_label_smoothing_0.15" \
    "$BASE_CONFIG train.loss.type=label_smoothing train.loss.smoothing=0.15"

# 7. Focal Loss + Class Weights (combined approach)
print_section "Experiment 7/7: Focal Loss + Class Weights"
run_experiment "T1_focal_plus_class_weights" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 'train.loss.class_weights=[1.3,1.0,1.0,0.8]'"

################################################################################
# Summary and Next Steps
################################################################################

print_header "TIER 1 EXPERIMENTS COMPLETED"
echo "End time: $(date)"
echo ""

# Calculate total runtime
if [ -f "$LOG_DIR/durations.txt" ]; then
    total_min=$(awk '{sum+=$NF} END {print sum}' "$LOG_DIR/durations.txt" 2>/dev/null || echo 0)
    echo "Total runtime: ${total_min} minutes"
    echo ""
fi

# Display results summary
print_section "RESULTS SUMMARY"
if [ -f "$RESULTS_FILE" ]; then
    cat "$RESULTS_FILE"
else
    echo "No results recorded."
fi
echo ""

# Count successes and failures
if [ -f "$RESULTS_FILE" ]; then
    total=$(wc -l < "$RESULTS_FILE")
    completed=$(grep -c "COMPLETED ✓" "$RESULTS_FILE" 2>/dev/null || echo 0)
    failed=$(grep -c "FAILED ✗" "$RESULTS_FILE" 2>/dev/null || echo 0)

    echo "Total experiments: $total"
    echo "Completed: $completed"
    echo "Failed: $failed"
    echo ""
fi

print_section "LOG FILES"
echo "Main log directory: $LOG_DIR"
echo "Individual experiment logs: $LOG_DIR/T1_*.log"
echo "Results summary: $RESULTS_FILE"
echo "Experiment durations: $LOG_DIR/durations.txt"
echo ""

print_section "NEXT STEPS"
echo ""
echo "1. Review W&B dashboard to compare all 7 experiments:"
echo "   https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps"
echo ""
echo "2. Look for the best performing loss function by:"
echo "   - Test accuracy (target: >97%)"
echo "   - Adenocarcinoma recall (target: >95%)"
echo "   - Confusion matrix (target: <4 adeno-squamous confusions)"
echo ""
echo "3. Quick comparison command:"
echo "   uv run python -c \\"
echo "   import wandb"
echo "   api = wandb.Api()"
echo "   runs = api.runs('mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps')"
echo "   recent = sorted([r for r in runs[:20] if 'T1_' in r.name],"
echo "                   key=lambda r: r.summary.get('test_acc', 0), reverse=True)"
echo "   for i, r in enumerate(recent[:5], 1):"
echo "       print(f'{i}. {r.name:40s} Test Acc: {r.summary.get(\\\"test_acc\\\", 0):.4f}')"
echo "   \\"
echo ""
echo "4. Evaluate best model:"
echo "   # Replace <run_id> with the best run from W&B"
echo "   invoke evaluate --checkpoint outputs/ct_scan_classifier/*/checkpoints/best_model.ckpt"
echo ""
echo "5. If results are promising, proceed to Tier 2:"
echo "   # Edit scripts/adenocarcinoma_improvement_experiments.sh"
echo "   # Uncomment Tier 2 section and update BEST_LOSS_T1 variable"
echo "   # Or run full suite: ./scripts/adenocarcinoma_improvement_experiments.sh"
echo ""
echo "6. If Focal Loss gamma=2.0 wins (most likely):"
echo "   BEST_LOSS='train.loss.type=focal train.loss.gamma=2.0'"
echo "   # Use this in all future Tier 2-5 experiments"
echo ""

print_header "ANALYSIS TIPS"
echo ""
echo "Key metrics to check in W&B:"
echo "  • test_acc - Overall test accuracy"
echo "  • test_loss - Lower is better, indicates calibration"
echo "  • Per-class metrics (if logged)"
echo ""
echo "Questions to answer:"
echo "  1. Which loss function achieves highest test accuracy?"
echo "  2. Does focal loss reduce overconfidence compared to baseline?"
echo "  3. Does label smoothing improve calibration?"
echo "  4. Is there a trade-off between accuracy and calibration?"
echo ""
echo "Expected outcomes:"
echo "  • Baseline: ~95.24% (same as current model)"
echo "  • Focal Loss: 95.5-96.5% (improved hard case handling)"
echo "  • Label Smoothing: 95.0-96.0% (better calibration)"
echo "  • Focal + Weights: 96.0-97.0% (best combination)"
echo ""

print_header "EXPERIMENT COMPLETE"
echo ""

exit 0
