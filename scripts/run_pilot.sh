#!/usr/bin/env bash
# v5 validation pilot (post-adversarial-review). 6 algorithms x 1 seed x 1
# alpha x 20 rounds — long enough for SCAFFOLD / FedDyn / MOON to stabilise
# under Option-II / contrastive corrections. Uses the scientifically-correct
# defaults baked into V5Config: pos_weight_split=train, cudnn_deterministic=True,
# FedDyn update_mode=option_ii. FedAdam is called with bias_correction=True
# to avoid the t<=5 momentum-ramp under-training the paper-default incurs.

set -euo pipefail

REPO="/home/thc1006/dev/fl-oran-tmc"
PYTHON="/home/thc1006/dev/colosseum-oran-federated-slicing/.venv/bin/python"
LOG_DIR="$REPO/artifacts/v5_sweep/_pilot_logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "pilot start: $(date -Iseconds)" | tee "$LOG_DIR/_summary.txt"
T0=$(date +%s)

"$PYTHON" experiments/run_v5_sweep_matrix.py \
    --seeds 42 \
    --alphas 0.5 \
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
    --algo-spec 'moon:{"mu": 1.0, "tau": 0.5}' \
    2>&1 | tee "$LOG_DIR/matrix.log"

TOTAL=$(( $(date +%s) - T0 ))
echo "" | tee -a "$LOG_DIR/_summary.txt"
echo "pilot end: $(date -Iseconds) total=${TOTAL}s" | tee -a "$LOG_DIR/_summary.txt"
echo "" | tee -a "$LOG_DIR/_summary.txt"
cat "$REPO/artifacts/v5_sweep/_matrix_summary_latest.csv" 2>/dev/null \
    | tee -a "$LOG_DIR/_summary.txt" || true
