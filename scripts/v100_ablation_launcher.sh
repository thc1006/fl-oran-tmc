#!/usr/bin/env bash
# V100 ablation parallel launcher — splits 15 cells across 4 GPUs.
#
# Cells: 3 archs × 1 algo (FedAvg) × 1 partition (random_split) × 5 seeds = 15
# Per-GPU schedule: GPU0=4, GPU1=4, GPU2=4, GPU3=3 (round-robin)
# Per-cell estimate: ~8 min on V100 (LSTM/Mamba), ~12 min (Spiking)
# Wall-clock target: ~30-40 min (4 cells/GPU × 8-12 min each)
#
# Output: artifacts/v7_ablation_random_split/v7_<arch>_fedavg_randsplit_n7_s<seed>/
# Logs:   logs/v100_ablation_gpu<N>.log

set -e
cd ~/fl-oran-tmc
source .venv/bin/activate

OUTDIR="artifacts/v7_ablation_random_split"
LOGDIR="logs"
PARQ="$HOME/data/coloran_raw_unified.parquet"
mkdir -p "$OUTDIR" "$LOGDIR"

# Cell schedule: (arch, seed) tuples assigned round-robin to GPUs.
# Order matters for tail latency: put longer cells (Spiking) first to spread.
declare -a SCHED_GPU0=("spiking_expand2:42" "lstm:42" "mamba:42" "lstm:0")
declare -a SCHED_GPU1=("spiking_expand2:0" "mamba:0" "lstm:1" "mamba:1")
declare -a SCHED_GPU2=("spiking_expand2:1" "lstm:2" "mamba:2" "lstm:3")
declare -a SCHED_GPU3=("spiking_expand2:2" "spiking_expand2:3" "mamba:3")

run_cell () {
    local gpu=$1 arch=$2 seed=$3
    local lr_warmup=3
    if [[ "$arch" == "spiking_expand2" ]]; then lr_warmup=5; fi
    local name="v7_${arch}_fedavg_randsplit_n7_s${seed}"
    local cell_dir="$OUTDIR/$name"
    if [[ -f "$cell_dir/summary.json" ]]; then
        echo "[gpu$gpu] SKIP existing $name"
        return
    fi
    echo "[gpu$gpu] START $name"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" python experiments/run_v7_fl_arch_sweep.py \
        --arch "$arch" \
        --algorithm fedavg \
        --partition-mode random_split \
        --n-clients 7 \
        --num-rounds 100 \
        --clients-per-round 5 \
        --max-steps-per-round 50 \
        --batch-size 64 \
        --lr 5.0e-4 \
        --lr-warmup-rounds "$lr_warmup" \
        --sample-ratio 1.0 \
        --threshold 0.10 \
        --seq-len 5 \
        --seed "$seed" \
        --unified-parquet "$PARQ" \
        --output-dir "$OUTDIR" \
        --pos-weight-split train \
        --mixed-precision bf16 \
        > "$LOGDIR/v100_ablation_gpu${gpu}_${name}.log" 2>&1
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
        local arch="${spec%%:*}"
        local seed="${spec##*:}"
        run_cell "$gpu" "$arch" "$seed"
    done
    echo "[gpu$gpu] CHAIN COMPLETE"
}

echo "=== launching 4 parallel chains ==="
date

run_gpu_chain 0 "${SCHED_GPU0[@]}" > "$LOGDIR/v100_ablation_chain0.log" 2>&1 &
PID0=$!
run_gpu_chain 1 "${SCHED_GPU1[@]}" > "$LOGDIR/v100_ablation_chain1.log" 2>&1 &
PID1=$!
run_gpu_chain 2 "${SCHED_GPU2[@]}" > "$LOGDIR/v100_ablation_chain2.log" 2>&1 &
PID2=$!
run_gpu_chain 3 "${SCHED_GPU3[@]}" > "$LOGDIR/v100_ablation_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: $PID0 $PID1 $PID2 $PID3"
echo "logs: $LOGDIR/v100_ablation_chain{0..3}.log"
wait $PID0 $PID1 $PID2 $PID3
echo "=== all chains complete ==="
date
ls -la "$OUTDIR" | head -20
