#!/bin/bash
################################################################################
# Phase 2: Fine-tuning Experiments with Analysis and Shutdown
################################################################################
#
# This script runs all Phase 2 adenocarcinoma improvement experiments, analyzes
# results, generates a summary report, and then shuts down the computer.
#
# Goal: Improve from 95.24% baseline to >97% test accuracy
#
# Experiments:
#   2.1 - Best Config Fine-tune (label_smooth=0.1, dropout=0.1, radiomics=1024)
#   2.2 - Label Smoothing Only (same architecture as checkpoint)
#   2.3 - Label Smoothing + Dropout 0.1
#   2.4 - Investigate 768 Anomaly (from scratch, 25 epochs)
#   2.5 - Extended Fine-tuning (30 epochs, lower LR)
#
# Usage:
#   chmod +x scripts/phase2_run_and_shutdown.sh
#   sudo ./scripts/phase2_run_and_shutdown.sh  # Requires sudo for shutdown
#
# Or to run in background:
#   nohup sudo ./scripts/phase2_run_and_shutdown.sh > /tmp/phase2_run.log 2>&1 &
#
################################################################################

set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure
# Note: We don't use 'set -e' to allow graceful handling of failed experiments

################################################################################
# Configuration
################################################################################

PROJECT_DIR="/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/ct_scan_mlops"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/phase2_run_$TIMESTAMP"
COOLDOWN=10  # Seconds between experiments
CHECKPOINT="models/dual_pathway_bn_finetune_kygevxv0.pt"
BASELINE_ACC="95.24"
TARGET_ACC="97.00"

################################################################################
# Setup and Validation
################################################################################

cd "$PROJECT_DIR" || {
    echo "ERROR: Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

# Validate environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    exit 1
fi

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize result files
RESULTS_FILE="$LOG_DIR/results_summary.txt"
ANALYSIS_FILE="$LOG_DIR/analysis_report.md"
DURATIONS_FILE="$LOG_DIR/durations.txt"

touch "$RESULTS_FILE"
touch "$DURATIONS_FILE"

################################################################################
# Helper Functions
################################################################################

# Print formatted header
print_header() {
    local text="$1"
    local width=78
    echo ""
    echo "$(printf '=%.0s' $(seq 1 $width))"
    echo "$text"
    echo "$(printf '=%.0s' $(seq 1 $width))"
}

# Print formatted section
print_section() {
    local text="$1"
    local width=78
    echo ""
    echo "$(printf -- '-%.0s' $(seq 1 $width))"
    echo "$text"
    echo "$(printf -- '-%.0s' $(seq 1 $width))"
}

# Extract test accuracy from log file
# Looks for patterns like "test_acc 0.93968" or "test_acc         0.9396825432777405"
extract_test_acc() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        # Try multiple patterns to find test accuracy
        local acc=""

        # Pattern 1: wandb summary line (e.g., "test_acc 0.93968")
        acc=$(grep -oE "test_acc\s+0\.[0-9]+" "$log_file" | tail -1 | grep -oE "0\.[0-9]+")

        # Pattern 2: Table format (e.g., "test_acc         0.9396825432777405")
        if [ -z "$acc" ]; then
            acc=$(grep -E "test_acc\s+" "$log_file" | grep -oE "0\.[0-9]{4,}" | tail -1)
        fi

        if [ -n "$acc" ]; then
            echo "$acc"
        else
            echo "N/A"
        fi
    else
        echo "N/A"
    fi
}

# Convert decimal accuracy to percentage string (e.g., 0.9524 -> 95.24)
to_percentage() {
    local decimal="$1"
    if [ "$decimal" = "N/A" ]; then
        echo "N/A"
    else
        echo "scale=2; $decimal * 100" | bc
    fi
}

# Compare accuracy against baseline
compare_to_baseline() {
    local acc_pct="$1"
    if [ "$acc_pct" = "N/A" ]; then
        echo "N/A"
    else
        local delta
        delta=$(echo "scale=2; $acc_pct - $BASELINE_ACC" | bc)
        if (( $(echo "$delta >= 0" | bc -l) )); then
            echo "+$delta%"
        else
            echo "$delta%"
        fi
    fi
}

# Check if accuracy meets target
meets_target() {
    local acc_pct="$1"
    if [ "$acc_pct" = "N/A" ]; then
        echo "UNKNOWN"
    elif (( $(echo "$acc_pct >= $TARGET_ACC" | bc -l) )); then
        echo "YES"
    else
        echo "NO"
    fi
}

# Run a training experiment with error handling
run_experiment() {
    local name="$1"
    local args="$2"
    local log_file="$LOG_DIR/${name}.log"
    local exit_code=0

    print_section "EXPERIMENT: $name"
    echo "Time: $(date)"
    echo "Log: $log_file"
    echo ""
    echo "Command: invoke train --args \"$args\""
    echo ""

    # Record start time
    local start_time
    start_time=$(date +%s)

    # Run training with error capture
    if invoke train --args "$args" 2>&1 | tee "$log_file"; then
        exit_code=0
    else
        exit_code=$?
    fi

    # Record end time and duration
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))

    # Extract test accuracy
    local test_acc
    test_acc=$(extract_test_acc "$log_file")
    local test_acc_pct
    test_acc_pct=$(to_percentage "$test_acc")
    local delta
    delta=$(compare_to_baseline "$test_acc_pct")
    local target_met
    target_met=$(meets_target "$test_acc_pct")

    # Record results
    if [ $exit_code -eq 0 ] && [ "$test_acc" != "N/A" ]; then
        echo "$name: ${test_acc_pct}% ($delta) [Target: $target_met] - COMPLETED" >> "$RESULTS_FILE"
        echo "Status: SUCCESS"
        echo "Test Accuracy: ${test_acc_pct}% ($delta vs baseline)"
        echo "Target Met (>${TARGET_ACC}%): $target_met"
    else
        echo "$name: FAILED (exit code: $exit_code, acc: $test_acc)" >> "$RESULTS_FILE"
        echo "Status: FAILED (exit code: $exit_code)"
    fi

    echo "Duration: ${duration_min} minutes (${duration} seconds)"
    echo "$name: ${duration_min} min" >> "$DURATIONS_FILE"

    # Cooldown between experiments (skip if this was the last one or it failed)
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Cooling down for $COOLDOWN seconds..."
        sleep $COOLDOWN
    fi

    return $exit_code
}

# Generate analysis report
generate_analysis_report() {
    print_header "GENERATING ANALYSIS REPORT"

    cat > "$ANALYSIS_FILE" << 'HEADER'
# Phase 2 Experiment Analysis Report

**Generated:** TIMESTAMP_PLACEHOLDER
**Baseline Model:** dual_pathway_bn_finetune_kygevxv0.pt
**Baseline Accuracy:** 95.24%
**Target Accuracy:** >97%

---

## Experiment Results

HEADER

    # Replace timestamp placeholder
    sed -i "s/TIMESTAMP_PLACEHOLDER/$(date '+%Y-%m-%d %H:%M:%S')/" "$ANALYSIS_FILE"

    # Add results table header
    cat >> "$ANALYSIS_FILE" << 'TABLE_HEADER'
| Experiment | Test Accuracy | Delta vs Baseline | Target Met | Status |
|------------|---------------|-------------------|------------|--------|
TABLE_HEADER

    # Parse results and add to table
    local best_acc=0
    local best_exp=""
    local experiments_passed=0
    local experiments_total=0

    while IFS= read -r line; do
        if [ -z "$line" ]; then
            continue
        fi

        experiments_total=$((experiments_total + 1))

        # Parse the line format: "exp_name: XX.XX% (+Y.YY%) [Target: YES/NO] - STATUS"
        local exp_name
        exp_name=$(echo "$line" | cut -d':' -f1 | xargs)

        if echo "$line" | grep -q "FAILED"; then
            echo "| $exp_name | FAILED | N/A | N/A | FAILED |" >> "$ANALYSIS_FILE"
        else
            local acc_pct
            acc_pct=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+%" | head -1 | tr -d '%')
            local delta
            delta=$(echo "$line" | grep -oE "[+-][0-9]+\.[0-9]+%" | head -1)
            local target
            target=$(echo "$line" | grep -oE "Target: (YES|NO)" | cut -d' ' -f2)

            if [ -n "$acc_pct" ]; then
                echo "| $exp_name | ${acc_pct}% | $delta | $target | COMPLETED |" >> "$ANALYSIS_FILE"

                # Track best result
                if (( $(echo "$acc_pct > $best_acc" | bc -l) )); then
                    best_acc=$acc_pct
                    best_exp=$exp_name
                fi

                # Count experiments that met target
                if [ "$target" = "YES" ]; then
                    experiments_passed=$((experiments_passed + 1))
                fi
            fi
        fi
    done < "$RESULTS_FILE"

    # Add summary section
    cat >> "$ANALYSIS_FILE" << SUMMARY

---

## Summary

- **Total Experiments:** $experiments_total
- **Best Accuracy:** ${best_acc}% ($best_exp)
- **Experiments Meeting Target (>97%):** $experiments_passed / $experiments_total
- **Improvement over Baseline:** $(echo "scale=2; $best_acc - $BASELINE_ACC" | bc)%

---

## Analysis

SUMMARY

    # Add success/failure analysis
    if (( $(echo "$best_acc >= $TARGET_ACC" | bc -l) )); then
        cat >> "$ANALYSIS_FILE" << SUCCESS
### SUCCESS - Target Achieved!

The experiment **$best_exp** achieved ${best_acc}% test accuracy, exceeding the target of ${TARGET_ACC}%.

**Next Steps:**
1. Evaluate the best model on adenocarcinoma-specific metrics
2. Generate confusion matrix comparison with baseline
3. Consider running additional fine-tuning sweeps to push accuracy further
4. Save the best checkpoint for production use

SUCCESS
    elif (( $(echo "$best_acc > $BASELINE_ACC" | bc -l) )); then
        cat >> "$ANALYSIS_FILE" << PARTIAL_SUCCESS
### PARTIAL SUCCESS - Improvement Achieved

The best result (${best_acc}%) improved over the baseline (${BASELINE_ACC}%) but did not reach the target (${TARGET_ACC}%).

**Recommendations:**
1. Run a W&B Bayesian sweep focusing on the winning configuration
2. Try longer training (40-50 epochs) with cosine annealing
3. Experiment with learning rate warmup
4. Consider ensemble methods combining top models

PARTIAL_SUCCESS
    else
        cat >> "$ANALYSIS_FILE" << NO_IMPROVEMENT
### ATTENTION - No Improvement Over Baseline

None of the experiments improved over the baseline of ${BASELINE_ACC}%.

**Possible Causes:**
1. Checkpoint loading issues with architecture mismatches
2. Learning rate too high for fine-tuning (try 1e-5 to 5e-5)
3. Need longer training with early stopping
4. Consider different regularization strategies

**Next Steps:**
1. Verify checkpoint is loading correctly
2. Check training logs for learning instabilities
3. Run diagnostic evaluation on the baseline model
4. Try training from scratch with best hyperparameters

NO_IMPROVEMENT
    fi

    # Add experiment details
    cat >> "$ANALYSIS_FILE" << 'DETAILS'

---

## Experiment Details

### Exp 2.1 - Best Config Fine-tune
- **Config:** Label smoothing 0.1, dropout 0.1, radiomics_hidden=1024, fusion_hidden=512
- **LR:** 0.00005, 15 epochs with checkpoint

### Exp 2.2 - Label Smoothing Only
- **Config:** Label smoothing 0.1, dropout 0.05 (original), radiomics_hidden=512, fusion_hidden=256 (original)
- **Rationale:** Test if label smoothing alone helps fine-tuning

### Exp 2.3 - Label Smoothing + Dropout 0.1
- **Config:** Label smoothing 0.1, dropout 0.1, original architecture
- **Rationale:** Isolate dropout improvement from architecture changes

### Exp 2.4 - Investigate 768 Anomaly
- **Config:** radiomics_hidden=768, fusion_hidden=384, from scratch (no checkpoint)
- **Purpose:** Verify if the 82.86% result from Phase 1 was a fluke

### Exp 2.5 - Extended Fine-tuning
- **Config:** Best fine-tuning config with 30 epochs, lr=0.00003
- **Rationale:** Allow more gradual convergence with lower learning rate

---

## Log Files

DETAILS

    # Add log file listing
    for log in "$LOG_DIR"/*.log; do
        if [ -f "$log" ]; then
            echo "- $(basename "$log")" >> "$ANALYSIS_FILE"
        fi
    done

    cat >> "$ANALYSIS_FILE" << FOOTER

---

**Report Generated By:** phase2_run_and_shutdown.sh
**Log Directory:** $LOG_DIR
FOOTER

    echo ""
    echo "Analysis report saved to: $ANALYSIS_FILE"
}

# Print final summary to console
print_final_summary() {
    print_header "PHASE 2 FINAL SUMMARY"

    echo ""
    echo "Completion Time: $(date)"
    echo "Log Directory: $LOG_DIR"
    echo ""

    # Print results summary
    print_section "Results"
    if [ -f "$RESULTS_FILE" ]; then
        cat "$RESULTS_FILE"
    else
        echo "No results recorded."
    fi

    echo ""
    echo "Baseline was: ${BASELINE_ACC}%"
    echo "Target was: >${TARGET_ACC}%"
    echo ""

    # Calculate total runtime
    if [ -f "$DURATIONS_FILE" ]; then
        local total_min
        total_min=$(awk -F': ' '{sum+=$2} END {print sum}' "$DURATIONS_FILE" | sed 's/ min//')
        if [ -n "$total_min" ] && [ "$total_min" != "0" ]; then
            local total_hrs
            total_hrs=$(echo "scale=1; $total_min / 60" | bc)
            echo "Total GPU time: ${total_hrs} hours (${total_min} minutes)"
        fi
    fi

    # Find best result
    if [ -f "$RESULTS_FILE" ]; then
        local best_line
        best_line=$(grep -v "FAILED" "$RESULTS_FILE" | sort -t':' -k2 -rn | head -1)
        if [ -n "$best_line" ]; then
            echo ""
            echo "Best Result: $best_line"
        fi
    fi

    print_section "Analysis Report"
    echo "Full analysis available at: $ANALYSIS_FILE"
    echo ""

    # Print key recommendations
    print_section "Next Actions (After System Restart)"
    echo "1. Review analysis report: cat $ANALYSIS_FILE"
    echo "2. Check W&B dashboard: https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps"
    echo "3. If best result >97%: Run adenocarcinoma evaluation"
    echo "4. If best result <97%: Plan sweep experiments"
    echo ""
}

################################################################################
# Main Experiment Suite
################################################################################

print_header "PHASE 2: ADENOCARCINOMA IMPROVEMENT EXPERIMENTS"
echo "Start Time: $(date)"
echo "Log Directory: $LOG_DIR"
echo ""
echo "Baseline Model: $CHECKPOINT"
echo "Baseline Accuracy: ${BASELINE_ACC}%"
echo "Target Accuracy: >${TARGET_ACC}%"
echo ""

# Record system info
print_section "System Information"
echo "Hostname: $(hostname)"
echo "OS: $(uname -s) $(uname -r)"
echo "Python: $(python --version 2>&1)"
nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "N/A")
nvidia_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")
echo "CUDA Driver: $nvidia_driver"
echo "GPU: $nvidia_gpu"
echo ""

################################################################################
# Experiment 2.1: Full best config fine-tune
# Note: This may fail if architecture dimensions don't match checkpoint
################################################################################

run_experiment "exp2.1_best_config_finetune" \
    "model=dual_pathway_top_features features=top_features \
    train.loss.type=label_smoothing train.loss.smoothing=0.1 \
    train.checkpoint=$CHECKPOINT \
    model.dropout=0.1 \
    model.radiomics_hidden=1024 model.fusion_hidden=512 \
    train.optimizer.lr=0.00005 \
    train.optimizer.weight_decay=2.06e-05 \
    data.batch_size=16 \
    train.max_epochs=15"

################################################################################
# Experiment 2.2: Label smoothing only (matching checkpoint architecture)
################################################################################

run_experiment "exp2.2_label_smoothing_finetune" \
    "model=dual_pathway_top_features features=top_features \
    train.loss.type=label_smoothing train.loss.smoothing=0.1 \
    train.checkpoint=$CHECKPOINT \
    model.dropout=0.05 \
    model.radiomics_hidden=512 model.fusion_hidden=256 \
    train.optimizer.lr=0.00005 \
    train.optimizer.weight_decay=2.06e-05 \
    data.batch_size=16 \
    train.max_epochs=15"

################################################################################
# Experiment 2.3: Label smoothing + higher dropout
################################################################################

run_experiment "exp2.3_label_smooth_dropout_finetune" \
    "model=dual_pathway_top_features features=top_features \
    train.loss.type=label_smoothing train.loss.smoothing=0.1 \
    train.checkpoint=$CHECKPOINT \
    model.dropout=0.1 \
    model.radiomics_hidden=512 model.fusion_hidden=256 \
    train.optimizer.lr=0.00005 \
    train.optimizer.weight_decay=2.06e-05 \
    data.batch_size=16 \
    train.max_epochs=15"

################################################################################
# Experiment 2.4: Investigate 768 anomaly (from scratch, no checkpoint)
################################################################################

run_experiment "exp2.4_768_anomaly_rerun" \
    "model=dual_pathway_top_features features=top_features \
    train.loss.type=label_smoothing train.loss.smoothing=0.1 \
    model.dropout=0.05 \
    model.radiomics_hidden=768 model.fusion_hidden=384 \
    train.optimizer.lr=0.000115 \
    train.optimizer.weight_decay=2.06e-05 \
    data.batch_size=16 \
    train.max_epochs=25"

################################################################################
# Experiment 2.5: Extended fine-tuning with lower LR
################################################################################

run_experiment "exp2.5_extended_finetune" \
    "model=dual_pathway_top_features features=top_features \
    train.loss.type=label_smoothing train.loss.smoothing=0.1 \
    train.checkpoint=$CHECKPOINT \
    model.dropout=0.1 \
    model.radiomics_hidden=512 model.fusion_hidden=256 \
    train.optimizer.lr=0.00003 \
    train.optimizer.weight_decay=2.06e-05 \
    data.batch_size=16 \
    train.max_epochs=30 \
    train.scheduler.T_max=30"

################################################################################
# Analysis and Summary
################################################################################

# Generate detailed analysis report
generate_analysis_report

# Print final summary to console and logs
print_final_summary

# Save summary to separate file for easy access after restart
cp "$ANALYSIS_FILE" "$PROJECT_DIR/logs/LATEST_PHASE2_ANALYSIS.md"
echo ""
echo "Quick access after restart: cat $PROJECT_DIR/logs/LATEST_PHASE2_ANALYSIS.md"

################################################################################
# Shutdown Sequence
################################################################################

print_header "INITIATING SHUTDOWN SEQUENCE"

echo ""
echo "All experiments completed successfully."
echo "Analysis report saved to: $ANALYSIS_FILE"
echo ""
echo "System will shut down in 60 seconds..."
echo "Press Ctrl+C to cancel shutdown."
echo ""

# Give time to cancel if needed
for i in {60..1}; do
    printf "\rShutdown in %2d seconds... " $i
    sleep 1
done

echo ""
echo ""
echo "Shutting down now..."

# Sync filesystem before shutdown
sync

# Shutdown the system
# Note: Requires script to be run with sudo or user to have shutdown privileges
if command -v shutdown &> /dev/null; then
    shutdown -h now "Phase 2 experiments completed. Shutting down."
else
    echo "WARNING: 'shutdown' command not found. Please shut down manually."
    exit 0
fi

exit 0
