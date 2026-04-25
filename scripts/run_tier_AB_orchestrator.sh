#!/usr/bin/env bash
# Tier A + B post-Option-B orchestrator.
#
# Sequenced runs (single GPU). Each step writes a per-step done flag
# under artifacts/logs/v6_tier_*_done.flag so wake-up logic and
# external monitors can resume from any partial completion.
#
# Total expected wall-clock: ~6-7 hr on RTX 4080.
#   Tier A.1 LSTM 100k × 10 seeds  ~85 min
#   Tier A.1 Mamba 100k × 10 seeds ~100 min
#   Tier B.2 spiking_expand2 25k × 10 seeds ~70 min
#   Tier B.2 spiking_expand2 50k × 10 seeds ~140 min  (only if 25k pilot looks competitive)
#   Tier A.2 NVML measurement on all cells ~30 min
#
# Run with:
#   nohup ./scripts/run_tier_AB_orchestrator.sh > artifacts/logs/tier_AB_orchestrator.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
LOGDIR="artifacts/logs"
SWEEP_DIR="artifacts/v6_arch_sweep"
mkdir -p "$LOGDIR"

run_v6 () {
    PYTHONPATH=src "$PY" experiments/run_v6_arch_sweep.py "$@"
}

echo "[$(date +%H:%M)] === Tier A.1: LSTM/Mamba 100k × 10 seeds (matched convergence) ==="
if [ ! -f "$LOGDIR/v6_tierA1_done.flag" ]; then
    run_v6 \
        --arch lstm,mamba \
        --seeds 42,0,1,2,3,7,11,13,17,23 \
        --total-steps 100000 \
        --val-every 4000 \
        --sample-ratio 1.0 \
        --output-suffix _100k \
        --output-dir "$SWEEP_DIR" \
        > "$LOGDIR/v6_tierA1_lstm_mamba_100k.log" 2>&1
    echo TIER_A1_DONE > "$LOGDIR/v6_tierA1_done.flag"
fi
echo "[$(date +%H:%M)] tier A.1 complete"

echo "[$(date +%H:%M)] === Tier B.2: spiking_expand2 25k × 10 seeds ==="
if [ ! -f "$LOGDIR/v6_tierB2_25k_done.flag" ]; then
    run_v6 \
        --arch spiking_expand2 \
        --seeds 42,0,1,2,3,7,11,13,17,23 \
        --total-steps 25000 \
        --val-every 1000 \
        --sample-ratio 1.0 \
        --output-suffix "" \
        --output-dir "$SWEEP_DIR" \
        > "$LOGDIR/v6_tierB2_expand2_25k.log" 2>&1
    echo TIER_B2_25K_DONE > "$LOGDIR/v6_tierB2_25k_done.flag"
fi
echo "[$(date +%H:%M)] tier B.2 (25k) complete"

echo "[$(date +%H:%M)] === Tier A.2: NVML wattage measurement on all cells ==="
if [ ! -f "$LOGDIR/v6_tierA2_done.flag" ]; then
    PYTHONPATH=src "$PY" scripts/measure_v6_gpu_energy.py \
        --sweep-dir "$SWEEP_DIR" \
        --n-inferences 128000 \
        --batch-size 64 \
        > "$LOGDIR/v6_tierA2_nvml.log" 2>&1
    echo TIER_A2_DONE > "$LOGDIR/v6_tierA2_done.flag"
fi
echo "[$(date +%H:%M)] tier A.2 complete"

echo "[$(date +%H:%M)] === Aggregator + paper-data refresh ==="
PYTHONPATH=src "$PY" scripts/recompute_v6_energy.py > "$LOGDIR/v6_recompute_energy_postAB.log" 2>&1
PYTHONPATH=src "$PY" scripts/aggregate_v6_results.py > "$LOGDIR/v6_aggregate_postAB.log" 2>&1

echo TIER_AB_ALL_DONE > "$LOGDIR/v6_tier_AB_done.flag"
echo "[$(date +%H:%M)] === Tier A + B all complete; flag at $LOGDIR/v6_tier_AB_done.flag ==="
