#!/bin/bash
################################################################################
# Adenocarcinoma Classification Improvement Experiments
################################################################################
#
# Comprehensive experiment suite to improve adenocarcinoma classification
# from 95.24% to >97% test accuracy
#
# Estimated runtime: 20-30 hours on RTX 3080
# Designed for overnight/weekend unattended execution
#
# Usage:
#   chmod +x scripts/adenocarcinoma_improvement_experiments.sh
#   nohup ./scripts/adenocarcinoma_improvement_experiments.sh > logs/experiment_run.log 2>&1 &
#
# Monitor progress:
#   tail -f logs/experiment_run.log
#   watch -n 60 "tail -20 logs/adeno_improvement_*/results_summary.txt"
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
LOG_DIR="$PROJECT_DIR/logs/adeno_improvement_$TIMESTAMP"
COOLDOWN=180  # 3 minutes between experiments to allow GPU to cool down
DRY_RUN=false  # Set to true to test script without actually running experiments

################################################################################
# Setup and Validation
################################################################################

cd "$PROJECT_DIR" || exit 1

# Validate environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    exit 1
fi

source .venv/bin/activate

# Create log directory
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/scripts"

# Initialize results summary
RESULTS_FILE="$LOG_DIR/results_summary.txt"
touch "$RESULTS_FILE"

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

# Run a training experiment
run_experiment() {
    local name=$1
    local args=$2
    local log_file="$LOG_DIR/${name}.log"

    print_section "EXPERIMENT: $name"
    echo "Time: $(date)"
    echo "Args: $args"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: invoke train --args \"$args\""
        echo "$name: DRY_RUN" >> "$RESULTS_FILE"
        return 0
    fi

    # Record start time
    local start_time=$(date +%s)

    # Run training
    if invoke train --args "$args" 2>&1 | tee "$log_file"; then
        local exit_code=0
        echo "$name: COMPLETED ✓" >> "$RESULTS_FILE"
        echo "Status: SUCCESS"
    else
        local exit_code=$?
        echo "$name: FAILED ✗ (exit code: $exit_code)" >> "$RESULTS_FILE"
        echo "WARNING: Experiment $name failed with exit code $exit_code"
        echo "Check log file: $log_file"
    fi

    # Record end time and duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))

    echo "Duration: ${duration_min} minutes (${duration} seconds)"
    echo "$name: ${duration_min} min" >> "$LOG_DIR/durations.txt"

    # Cooldown
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Cooling down for $COOLDOWN seconds..."
        sleep $COOLDOWN
    fi

    return $exit_code
}

# Extract best result from W&B (requires wandb API)
get_best_result() {
    echo ""
    echo "Fetching best results from W&B..."
    python3 - <<'EOF'
import wandb
import sys

try:
    api = wandb.Api()
    runs = api.runs('mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps')

    # Find runs from today
    import datetime
    today = datetime.datetime.now().date()

    recent_runs = []
    for run in runs[:100]:
        if 'test_acc' in run.summary:
            created_at = datetime.datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
            if created_at.date() == today:
                recent_runs.append((run.name, run.summary['test_acc'], run.id))

    if recent_runs:
        recent_runs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{'='*78}")
        print("TOP 5 EXPERIMENTS TODAY:")
        print(f"{'='*78}")
        for i, (name, acc, run_id) in enumerate(recent_runs[:5], 1):
            print(f"{i}. {name:50s} {acc:.4f} ({run_id})")
        print(f"{'='*78}\n")
    else:
        print("No runs found for today.")

except Exception as e:
    print(f"Warning: Could not fetch W&B results: {e}", file=sys.stderr)
EOF
}

################################################################################
# Main Experiment Suite
################################################################################

print_header "ADENOCARCINOMA CLASSIFICATION IMPROVEMENT EXPERIMENTS"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo "Dry run: $DRY_RUN"
echo ""
echo "Current model: dual_pathway_bn_finetune_kygevxv0.pt"
echo "Baseline accuracy: 95.238%"
echo "Target accuracy: >97%"
echo ""
echo "Focus: Reduce adenocarcinoma error rate from 8.33% to <5%"
print_header ""

# Record system info
print_section "System Information"
echo "Hostname: $(hostname)"
echo "OS: $(uname -s) $(uname -r)"
echo "Python: $(python --version)"
echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

################################################################################
# Baseline Configuration
################################################################################

# Best hyperparameters from previous sweep
BASE_CONFIG="model=dual_pathway_top_features features=top_features \
train.optimizer.lr=0.000115 train.optimizer.weight_decay=2.06e-05 \
model.dropout=0.05 model.radiomics_hidden=512 model.fusion_hidden=256 \
data.batch_size=16 train.max_epochs=25 train.profiling.enabled=false"

################################################################################
# TIER 1: Loss Function Experiments
# Objective: Improve handling of hard cases and reduce overconfidence
# Expected runtime: 5-7 hours
################################################################################

print_header "TIER 1: LOSS FUNCTION EXPERIMENTS (5-7 hours)"
echo "Objective: Test different loss functions to:"
echo "  - Better handle hard adenocarcinoma cases"
echo "  - Reduce overconfidence (current: 83.3% on errors)"
echo "  - Improve calibration"
echo ""

# Baseline with standard cross-entropy
run_experiment "T1_baseline_crossentropy" \
    "$BASE_CONFIG"

# Focal Loss variants
run_experiment "T1_focal_loss_gamma2.0" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0"

run_experiment "T1_focal_loss_gamma2.5" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.5"

run_experiment "T1_focal_loss_gamma3.0" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=3.0"

# Label Smoothing variants
run_experiment "T1_label_smoothing_0.1" \
    "$BASE_CONFIG train.loss.type=label_smoothing train.loss.smoothing=0.1"

run_experiment "T1_label_smoothing_0.15" \
    "$BASE_CONFIG train.loss.type=label_smoothing train.loss.smoothing=0.15"

# Combined approach: Focal + Class Weights
run_experiment "T1_focal_plus_class_weights" \
    "$BASE_CONFIG train.loss.type=focal train.loss.gamma=2.0 'train.loss.class_weights=[1.3,1.0,1.0,0.8]'"

print_section "Tier 1 Complete"
echo "Experiments: 7"
echo "Check W&B dashboard to identify best loss function"
echo ""

################################################################################
# TIER 2: Weighted Sampling Experiments
# Objective: Oversample adenocarcinoma to improve representation
# Expected runtime: 2-3 hours
################################################################################

print_header "TIER 2: WEIGHTED SAMPLING EXPERIMENTS (2-3 hours)"
echo "Objective: Balance class representation during training"
echo "  - Oversample adenocarcinoma (class weight 1.5-2.0)"
echo "  - Undersample normal (class weight 0.6-0.7)"
echo ""

# Weighted sampling variants
run_experiment "T2_weighted_sampling_moderate" \
    "$BASE_CONFIG data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]'"

run_experiment "T2_weighted_sampling_aggressive" \
    "$BASE_CONFIG data.sampling.weighted=true 'data.sampling.class_weights=[2.0,1.0,1.0,0.6]'"

# Combine best loss from Tier 1 with weighted sampling
# NOTE: Update this after Tier 1 completes if a different loss performs best
BEST_LOSS_T1="train.loss.type=focal train.loss.gamma=2.0"

run_experiment "T2_focal_plus_weighted_moderate" \
    "$BASE_CONFIG $BEST_LOSS_T1 data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]'"

run_experiment "T2_focal_plus_weighted_aggressive" \
    "$BASE_CONFIG $BEST_LOSS_T1 data.sampling.weighted=true 'data.sampling.class_weights=[2.0,1.0,1.0,0.6]'"

print_section "Tier 2 Complete"
echo "Experiments: 4"
echo ""

################################################################################
# TIER 3: Data Augmentation Experiments
# Objective: Improve shape feature robustness
# Expected runtime: 6-8 hours
################################################################################

print_header "TIER 3: DATA AUGMENTATION EXPERIMENTS (6-8 hours)"
echo "Objective: Improve robustness to shape variations"
echo "  - Elastic transforms (distort tumor shapes)"
echo "  - Grid distortion (local geometric perturbations)"
echo "  - Coarse dropout (force distributed feature usage)"
echo ""

# Individual augmentation techniques
run_experiment "T3_elastic_transform" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.elastic_transform=true \
    data.augmentation.train.elastic_alpha=120 \
    data.augmentation.train.elastic_sigma=6 \
    data.augmentation.train.elastic_p=0.3"

run_experiment "T3_grid_distortion" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.grid_distortion=true \
    data.augmentation.train.grid_steps=5 \
    data.augmentation.train.grid_distort_limit=0.3 \
    data.augmentation.train.grid_p=0.3"

run_experiment "T3_coarse_dropout" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.coarse_dropout=true \
    data.augmentation.train.dropout_max_holes=8 \
    data.augmentation.train.dropout_max_height=32 \
    data.augmentation.train.dropout_max_width=32 \
    data.augmentation.train.dropout_p=0.3"

run_experiment "T3_gaussian_noise" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.gaussian_noise=true \
    data.augmentation.train.noise_std=0.02"

# Combined augmentations
run_experiment "T3_combined_shape_aug" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.elastic_transform=true \
    data.augmentation.train.grid_distortion=true \
    data.augmentation.train.coarse_dropout=true"

run_experiment "T3_all_augmentations" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.augmentation.train.elastic_transform=true \
    data.augmentation.train.grid_distortion=true \
    data.augmentation.train.coarse_dropout=true \
    data.augmentation.train.gaussian_noise=true"

print_section "Tier 3 Complete"
echo "Experiments: 6"
echo ""

################################################################################
# TIER 4: Architecture Modifications
# Objective: Better leverage radiomics features for edge cases
# Expected runtime: 8-12 hours
################################################################################

print_header "TIER 4: ARCHITECTURE MODIFICATIONS (8-12 hours)"
echo "Objective: Improve feature extraction and fusion"
echo "  - Larger radiomics pathway"
echo "  - Full 50 features (vs 16 top features)"
echo "  - Different dropout rates"
echo ""

# Determine best augmentation from Tier 3
BEST_AUG_T3="data.augmentation.train.elastic_transform=true"

# Larger radiomics pathway
run_experiment "T4_larger_radiomics_hidden_768" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.radiomics_hidden=768 model.fusion_hidden=384"

run_experiment "T4_larger_radiomics_hidden_1024" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.radiomics_hidden=1024 model.fusion_hidden=512"

# Dropout variations
run_experiment "T4_lower_dropout_0.02" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.dropout=0.02"

run_experiment "T4_higher_dropout_0.1" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.dropout=0.1"

# Full 50 features (requires feature extraction first)
print_section "Extracting full 50 features..."
if [ "$DRY_RUN" = false ]; then
    invoke extract-features --args "features=default" || echo "Warning: Feature extraction failed"
fi

run_experiment "T4_full_50_features" \
    "model=dual_pathway features=default $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.radiomics_dim=50 model.radiomics_hidden=512 model.fusion_hidden=256 \
    train.optimizer.lr=0.0001 train.optimizer.weight_decay=2.06e-05 \
    model.dropout=0.05 data.batch_size=16 train.max_epochs=25"

print_section "Tier 4 Complete"
echo "Experiments: 5"
echo ""

################################################################################
# TIER 5: Extended Training & Best Combined Configuration
# Objective: Allow precise convergence on hard examples
# Expected runtime: 6-10 hours
################################################################################

print_header "TIER 5: EXTENDED TRAINING (6-10 hours)"
echo "Objective: Extended training with best configurations"
echo "  - 40-50 epochs for more precise learning"
echo "  - Combined best approaches from Tiers 1-4"
echo ""

# Extended training with current best config
run_experiment "T5_extended_training_40_epochs" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    model.radiomics_hidden=768 model.fusion_hidden=384 \
    train.max_epochs=40 train.scheduler.T_max=40"

run_experiment "T5_extended_training_50_epochs" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    train.max_epochs=50 train.scheduler.T_max=50"

# Best combined configuration with weighted sampling
run_experiment "T5_best_combined_config" \
    "$BASE_CONFIG $BEST_LOSS_T1 $BEST_AUG_T3 \
    data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]' \
    model.radiomics_hidden=768 model.fusion_hidden=384 \
    model.dropout=0.05 \
    train.max_epochs=40 train.scheduler.T_max=40"

# Conservative best (fewer changes, likely more stable)
run_experiment "T5_conservative_best" \
    "$BASE_CONFIG $BEST_LOSS_T1 \
    data.sampling.weighted=true 'data.sampling.class_weights=[1.5,1.0,1.0,0.7]' \
    train.max_epochs=35 train.scheduler.T_max=35"

print_section "Tier 5 Complete"
echo "Experiments: 4"
echo ""

################################################################################
# Final Summary and Cleanup
################################################################################

print_header "EXPERIMENT SUITE COMPLETED"
echo "End time: $(date)"
echo ""

# Calculate total runtime
if [ -f "$LOG_DIR/durations.txt" ]; then
    total_min=$(awk '{sum+=$NF} END {print sum}' "$LOG_DIR/durations.txt")
    total_hrs=$(echo "scale=1; $total_min / 60" | bc)
    echo "Total GPU time: ${total_hrs} hours (${total_min} minutes)"
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
    completed=$(grep -c "COMPLETED ✓" "$RESULTS_FILE" || echo 0)
    failed=$(grep -c "FAILED ✗" "$RESULTS_FILE" || echo 0)

    echo "Total experiments: $total"
    echo "Completed: $completed"
    echo "Failed: $failed"
    echo ""
fi

print_section "LOG FILES"
echo "Main log directory: $LOG_DIR"
echo "Individual experiment logs: $LOG_DIR/*.log"
echo "Results summary: $RESULTS_FILE"
echo "Experiment durations: $LOG_DIR/durations.txt"
echo ""

print_section "NEXT STEPS"
echo "1. Review W&B dashboard for detailed metrics comparison:"
echo "   https://wandb.ai/mathiashl-danmarks-tekniske-universitet-dtu/CT_Scan_MLOps"
echo ""
echo "2. Identify best performing model and evaluate:"
echo "   invoke evaluate --checkpoint outputs/ct_scan_classifier/sweeps/<run_id>/best_model.ckpt"
echo ""
echo "3. Compare with baseline:"
echo "   invoke compare-baselines \\"
echo "     --baseline models/dual_pathway_bn_finetune_kygevxv0.pt \\"
echo "     --improved outputs/ct_scan_classifier/sweeps/<best_run_id>/best_model.ckpt"
echo ""
echo "4. Analyze remaining errors:"
echo "   invoke analyze-errors --checkpoint outputs/ct_scan_classifier/sweeps/<best_run_id>/best_model.ckpt"
echo ""
echo "5. Review improvement plan:"
echo "   cat docs/AdenocarcinomaImprovementPlan.md"
echo ""

# Try to fetch and display best results from W&B
get_best_result

print_header "EXPERIMENT SUITE FINISHED"
echo "Timestamp: $(date)"
echo "Log directory preserved at: $LOG_DIR"
echo ""
echo "🎯 Target: >97% accuracy, <5% adenocarcinoma error rate"
echo "📊 Check W&B for detailed results and confusion matrices"
print_header ""

exit 0
