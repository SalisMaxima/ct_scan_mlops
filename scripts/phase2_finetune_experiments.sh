#!/bin/bash
################################################################################
# Phase 2: Fine-tuning Experiments
################################################################################
#
# Apply winning hyperparameters from Phase 1 screening to fine-tune the
# 95.24% baseline model toward >97% accuracy.
#
# Usage:
#   chmod +x scripts/phase2_finetune_experiments.sh
#   ./scripts/phase2_finetune_experiments.sh
#
################################################################################

set -e
set -u
set -o pipefail

PROJECT_DIR="/media/salismaxima/41827d46-03ee-4c8d-9636-12e2cf1281c3/Projects/ct_scan_mlops"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/phase2_finetune_$TIMESTAMP"
COOLDOWN=10
CHECKPOINT="models/dual_pathway_bn_finetune_kygevxv0.pt"

cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p "$LOG_DIR"

RESULTS_FILE="$LOG_DIR/results_summary.txt"
touch "$RESULTS_FILE"

run_experiment() {
    local name=$1
    local args=$2
    local log_file="$LOG_DIR/${name}.log"

    echo ""
    echo "============================================================"
    echo "EXPERIMENT: $name"
    echo "Time: $(date)"
    echo "============================================================"

    local start_time=$(date +%s)

    if invoke train --args "$args" 2>&1 | tee "$log_file"; then
        local acc=$(grep -oE "test_acc 0\.[0-9]+" "$log_file" | tail -1 | awk '{print $2}')
        echo "$name: $acc ✓" >> "$RESULTS_FILE"
        echo "Result: $acc"
    else
        echo "$name: FAILED ✗" >> "$RESULTS_FILE"
        echo "FAILED"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Duration: $duration seconds"

    sleep $COOLDOWN
}

echo "============================================================"
echo "PHASE 2: FINE-TUNING EXPERIMENTS"
echo "============================================================"
echo "Start time: $(date)"
echo "Baseline: 95.24%"
echo "Target: >97%"
echo "Checkpoint: $CHECKPOINT"
echo "Log directory: $LOG_DIR"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

################################################################################
# Experiment 2.1: Full best config (may fail due to architecture mismatch)
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
# Experiment 2.2: Label smoothing only (same architecture as checkpoint)
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
# Experiment 2.5: Extended fine-tuning with best simple config
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
# Summary
################################################################################

echo ""
echo "============================================================"
echo "PHASE 2 COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "RESULTS:"
echo "--------"
cat "$RESULTS_FILE"
echo ""
echo "Baseline was: 95.24%"
echo "Target: >97%"
echo ""
echo "Log directory: $LOG_DIR"
echo "============================================================"
