#!/usr/bin/env bash
# v5 full TMC grid: 5 seeds x 5 Dirichlet alphas x 6 algorithms x 20 rounds
# = 150 cells. Estimated wall-clock ~2.3 h on a single RTX 4080.

set -euo pipefail

REPO="/home/thc1006/dev/fl-oran-tmc"
PYTHON="/home/thc1006/dev/colosseum-oran-federated-slicing/.venv/bin/python"
LOG_DIR="$REPO/artifacts/v5_sweep/_fullsweep_logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "full sweep start: $(date -Iseconds)" | tee "$LOG_DIR/_summary.txt"
T0=$(date +%s)

"$PYTHON" experiments/run_v5_sweep_matrix.py \
    --seeds 42 43 44 45 46 \
    --alphas 0.05 0.1 0.5 1.0 10.0 \
    --partition-mode dirichlet \
    --n-clients 5 \
    --num-rounds 20 \
    --clients-per-round 5 \
    --max-steps-per-round 50 \
    --batch-size 256 \
    --lr 5e-4 \
    --lr-warmup-rounds 3 \
    --sample-ratio 1.0 \
    --seq-len 5 \
    --device cuda \
    --mixed-precision bf16 \
    --compile-model reduce-overhead \
    --output-dir "$REPO/artifacts/v5_sweep" \
    --algo-spec 'fedavg:{}' \
    --algo-spec 'fedprox:{"mu": 0.01}' \
    --algo-spec 'fedadam:{"server_lr": 0.01, "bias_correction": true}' \
    --algo-spec 'scaffold:{}' \
    --algo-spec 'feddyn:{"alpha": 0.01}' \
    --algo-spec 'moon:{"mu": 0.1, "tau": 1.0}' \
    2>&1 | tee "$LOG_DIR/matrix.log"

TOTAL=$(( $(date +%s) - T0 ))
echo "" | tee -a "$LOG_DIR/_summary.txt"
echo "full sweep end: $(date -Iseconds) total=${TOTAL}s ($(( TOTAL / 60 )) min)" | tee -a "$LOG_DIR/_summary.txt"
echo "" | tee -a "$LOG_DIR/_summary.txt"
echo "cells completed:" | tee -a "$LOG_DIR/_summary.txt"
wc -l "$REPO/artifacts/v5_sweep/_matrix_summary_latest.csv" 2>/dev/null | tee -a "$LOG_DIR/_summary.txt" || true
