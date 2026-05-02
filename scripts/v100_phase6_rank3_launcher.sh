#!/usr/bin/env bash
# Phase 6 Rank 3: per-BS Dirichlet mechanism disambiguation
#
# Cells: 3 archs × FedAvg × per_bs_dirichlet × {α=0.05, 0.50, 5.00, 10.0} × 5 seeds = 60
# (NOT 300; FUTURE_WORK 300-cell estimate assumed all 5 algos × all archs;
# we restrict to FedAvg + 3 archs + 4 alphas + 5 seeds for the M9 confound
# closure — the mechanism question is per-BS bs-grouping preservation, which
# is partition-axis not algorithm-axis. 60 cells = same scope as Rank 1
# T-ABLATION random_split.)
#
# Hardware: 4× Tesla V100-SXM2-32GB; OMP=4 to match 16-core container
# Output: artifacts/v7_phase6_per_bs_dirichlet/

set -e
cd ~/fl-oran-tmc
source .venv/bin/activate

OUTDIR="artifacts/v7_phase6_per_bs_dirichlet"
LOGDIR="logs"
PARQ="/dev/shm/coloran_raw_unified.parquet"
mkdir -p "$OUTDIR" "$LOGDIR"

if [ ! -f "$PARQ" ]; then
    echo "ERROR: $PARQ missing"
    exit 1
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# 60 cells distributed across 4 GPUs (15 cells/GPU).
# Spec key: arch:alpha:seed. Spiking goes first per chain (slowest).
ARCHS=("spiking_expand2" "mamba" "lstm")
ALPHAS=("0.05" "0.50" "5.00" "10.00")
SEEDS=("42" "0" "1" "2" "3")

# Build full cell list, then deal across GPUs round-robin.
ALL_CELLS=()
for arch in "${ARCHS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ALL_CELLS+=("$arch:$alpha:$seed")
        done
    done
done
N_CELLS=${#ALL_CELLS[@]}
echo "total cells: $N_CELLS (expected 60)"

# Build per-GPU schedules
declare -a SCHED_GPU0=()
declare -a SCHED_GPU1=()
declare -a SCHED_GPU2=()
declare -a SCHED_GPU3=()
for i in "${!ALL_CELLS[@]}"; do
    case $((i % 4)) in
        0) SCHED_GPU0+=("${ALL_CELLS[$i]}") ;;
        1) SCHED_GPU1+=("${ALL_CELLS[$i]}") ;;
        2) SCHED_GPU2+=("${ALL_CELLS[$i]}") ;;
        3) SCHED_GPU3+=("${ALL_CELLS[$i]}") ;;
    esac
done

run_cell () {
    local gpu=$1 arch=$2 alpha=$3 seed=$4
    local lr_warmup=3
    if [[ "$arch" == "spiking_expand2" ]]; then lr_warmup=5; fi
    local alpha_tag=$(echo "$alpha" | sed 's/\./p/' | head -c 4)
    local name="v7_${arch}_fedavg_perbsdir_a${alpha_tag}_s${seed}"
    local cell_dir="$OUTDIR/$name"
    if [[ -f "$cell_dir/summary.json" ]]; then
        echo "[gpu$gpu] SKIP $name"
        return
    fi
    echo "[gpu$gpu] START $name"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    python experiments/run_v7_fl_arch_sweep.py \
        --arch "$arch" \
        --algorithm fedavg \
        --partition-mode per_bs_dirichlet \
        --alpha "$alpha" \
        --algo-kwargs '{"sub_per_bs": 2}' \
        --n-clients 14 \
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
        --name "$name" \
        > "$LOGDIR/v100_p6r3_gpu${gpu}_${name}.log" 2>&1
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
        IFS=':' read -r arch alpha seed <<< "$spec"
        run_cell "$gpu" "$arch" "$alpha" "$seed"
    done
    echo "[gpu$gpu] CHAIN COMPLETE"
}

echo "=== Phase 6 Rank 3: per-BS Dirichlet ablation ==="
date

run_gpu_chain 0 "${SCHED_GPU0[@]}" > "$LOGDIR/v100_p6r3_chain0.log" 2>&1 &
PID0=$!
run_gpu_chain 1 "${SCHED_GPU1[@]}" > "$LOGDIR/v100_p6r3_chain1.log" 2>&1 &
PID1=$!
run_gpu_chain 2 "${SCHED_GPU2[@]}" > "$LOGDIR/v100_p6r3_chain2.log" 2>&1 &
PID2=$!
run_gpu_chain 3 "${SCHED_GPU3[@]}" > "$LOGDIR/v100_p6r3_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: $PID0 $PID1 $PID2 $PID3"
# Per-PID `wait || true` so a single chain failure (set -e in run_cell)
# doesn't make the launcher exit before the other 3 chains are reaped.
# Without this, any chain failure → wait returns non-zero → set -e fires
# → script exits → orphaned python procs continue holding GPUs while a
# chain-watcher polling pgrep <launcher.sh> mistakenly fires the next
# phase. Discovered during Phase 6 Rank 3 audit 2026-05-03.
for p in $PID0 $PID1 $PID2 $PID3; do wait "$p" || echo "[wait] chain pid=$p exited non-zero"; done
echo "=== Phase 6 Rank 3 complete ==="
date
ls -la "$OUTDIR" | head -30
