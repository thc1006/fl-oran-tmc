#!/usr/bin/env bash
# MOON hparam sweep at α=0.5, seed=42. 15 cells (mu × tau = 5 × 3) ≈ 14 min.
# After completion, pick best (mu*, tau*) and update run_full_sweep.sh.

set -euo pipefail

REPO="/home/thc1006/dev/fl-oran-tmc"
PYTHON="/home/thc1006/dev/colosseum-oran-federated-slicing/.venv/bin/python"
LOG_DIR="$REPO/artifacts/v5_sweep/_moon_hpo_logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "MOON HPO start: $(date -Iseconds)" | tee "$LOG_DIR/_summary.txt"
T0=$(date +%s)

"$PYTHON" experiments/run_moon_hpo.py \
    --seed 42 \
    --alpha 0.5 \
    --mus 0.1 0.5 1.0 5.0 10.0 \
    --taus 0.1 0.5 1.0 \
    --num-rounds 20 \
    --max-steps-per-round 50 \
    --batch-size 256 \
    --lr 5e-4 \
    --output-dir "$REPO/artifacts/v5_sweep/_moon_hpo" \
    2>&1 | tee "$LOG_DIR/hpo.log"

TOTAL=$(( $(date +%s) - T0 ))
echo "" | tee -a "$LOG_DIR/_summary.txt"
echo "MOON HPO end: $(date -Iseconds) total=${TOTAL}s ($(( TOTAL / 60 ))m)" | tee -a "$LOG_DIR/_summary.txt"
echo "" | tee -a "$LOG_DIR/_summary.txt"
echo "best 5 trials:" | tee -a "$LOG_DIR/_summary.txt"
grep -E "BEST:|MOON|RESCUED|IMPROVED|UNDERPERFORMS" "$LOG_DIR/hpo.log" | tee -a "$LOG_DIR/_summary.txt"
