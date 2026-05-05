#!/usr/bin/env bash
# Phase 6 Rank 1: SLA threshold sensitivity sweep
#
# Cells: LSTM × FedAvg × IID natural-by-BS × {threshold=0.05, 0.15, 0.20} × 5 seeds = 15
# Hardware: 4× Tesla V100-SXM2-32GB
# Optimisations vs Phase A:
#   * OMP_NUM_THREADS=4 (4 procs × 4 = 16 cores per container, 1:1)
#   * MKL_NUM_THREADS=4 (same)
#   * Parquet symlink to /dev/shm (avoid 16× NFS reads)
#   * Cell duration estimate: ~600s/cell on V100 LSTM
# Wall-clock target: 4 cells/GPU × ~10 min ≈ 40 min total wall
#
# Output: artifacts/v7_phase6_threshold/v7_<arch>_<algo>_iid_n7_s<seed>_t<thr>/

set -e
cd ~/fl-oran-tmc
source .venv/bin/activate

OUTDIR="artifacts/v7_phase6_threshold"
LOGDIR="logs"
PARQ="/dev/shm/coloran_raw_unified.parquet"  # tmpfs path, NOT NFS
mkdir -p "$OUTDIR" "$LOGDIR"

# Verify parquet on /dev/shm
if [ ! -f "$PARQ" ]; then
    echo "ERROR: $PARQ missing — copy from \$HOME/data first"
    exit 1
fi

# CPU optimisation env
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Schedule: round-robin (threshold, seed) → GPU
declare -a SCHED_GPU0=("0.05:42" "0.15:42" "0.20:42" "0.05:0")
declare -a SCHED_GPU1=("0.15:0" "0.20:0" "0.05:1" "0.15:1")
declare -a SCHED_GPU2=("0.20:1" "0.05:2" "0.15:2" "0.20:2")
declare -a SCHED_GPU3=("0.05:3" "0.15:3" "0.20:3")

run_cell () {
    local gpu=$1 thr=$2 seed=$3
    # Encode threshold in cell name: 0.05 → t05, 0.15 → t15, 0.20 → t20
    local thr_tag=$(echo "$thr" | sed 's/\.//' | sed 's/^0//' | head -c 2)
    local name="v7_lstm_fedavg_iid_n7_s${seed}_t${thr_tag}"
    local cell_dir="$OUTDIR/$name"
    if [[ -f "$cell_dir/summary.json" ]]; then
        echo "[gpu$gpu] SKIP existing $name"
        return
    fi
    echo "[gpu$gpu] START $name (thr=$thr seed=$seed)"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    python experiments/run_v7_fl_arch_sweep.py \
        --arch lstm \
        --algorithm fedavg \
        --partition-mode iid \
        --n-clients 7 \
        --num-rounds 100 \
        --clients-per-round 5 \
        --max-steps-per-round 50 \
        --batch-size 64 \
        --lr 5.0e-4 \
        --lr-warmup-rounds 3 \
        --sample-ratio 1.0 \
        --threshold "$thr" \
        --seq-len 5 \
        --seed "$seed" \
        --unified-parquet "$PARQ" \
        --output-dir "$OUTDIR" \
        --pos-weight-split train \
        --mixed-precision bf16 \
        --name "$name" \
        > "$LOGDIR/v100_p6r1_gpu${gpu}_${name}.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [[ $rc -eq 0 ]]; then
        echo "[gpu$gpu] DONE $name in ${dt}s"
    else
        echo "[gpu$gpu] FAIL $name rc=$rc in ${dt}s"
    fi
}

run_gpu_chain () {
    local gpu=$1
    shift
    local cells=("$@")
    for spec in "${cells[@]}"; do
        local thr="${spec%%:*}"
        local seed="${spec##*:}"
        run_cell "$gpu" "$thr" "$seed"
    done
    echo "[gpu$gpu] CHAIN COMPLETE"
}

echo "=== Phase 6 Rank 1: launching 4 parallel chains ==="
date

run_gpu_chain 0 "${SCHED_GPU0[@]}" > "$LOGDIR/v100_p6r1_chain0.log" 2>&1 &
PID0=$!
run_gpu_chain 1 "${SCHED_GPU1[@]}" > "$LOGDIR/v100_p6r1_chain1.log" 2>&1 &
PID1=$!
run_gpu_chain 2 "${SCHED_GPU2[@]}" > "$LOGDIR/v100_p6r1_chain2.log" 2>&1 &
PID2=$!
run_gpu_chain 3 "${SCHED_GPU3[@]}" > "$LOGDIR/v100_p6r1_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: $PID0 $PID1 $PID2 $PID3"
# Per-PID `wait || true` so a single chain failure (set -e in run_cell)
# doesn't make the launcher exit before the other 3 chains are reaped.
# Without this, any chain failure → wait returns non-zero → set -e fires
# → script exits → orphaned python procs continue holding GPUs while a
# chain-watcher polling pgrep <launcher.sh> mistakenly fires the next
# phase. Discovered during Phase 6 Rank 3 audit 2026-05-03.
for p in $PID0 $PID1 $PID2 $PID3; do wait "$p" || echo "[wait] chain pid=$p exited non-zero"; done
echo "=== Phase 6 Rank 1 complete ==="
date
ls -la "$OUTDIR" | head -20
