#!/usr/bin/env bash
# SAM-family Phase 6 V100 launcher.
#
# Sweep: 1 arch (lstm) × 2 algos (fedscam, fedgmt) × 6 partitions
#         (iid + 5 Dirichlet alphas) × 5 seeds = 60 cells
# Hardware: 4× Tesla V100-SXM2-32GB
# Output: artifacts/v7_sam_family/<cell_name>/
#
# Per-cell wall on V100 (LSTM at sample_ratio=1.0, 100 rounds × 5 clients
#   × 50 steps): ~5 min including data prep (NO SharedSplits cache; each
#   cell re-runs data prep from /dev/shm parquet which is fast).
# 4 chains × 15 cells × 5 min ≈ 75 min wall.
#
# Algorithm hyperparameters pinned to paper defaults:
#   FedSCAM (arXiv:2601.00853 §3 + §4.1.3 mid-of-tested):
#     rho_max=0.05, alpha_rho=1.0, gamma=1.0, beta_align=0.8, kappa=1.0
#   FedGMT (ICML 2025; harrylee999/FL-SAM README example):
#     alpha_ema=0.95, gamma_kl=1.0, tau=3.0, beta=10.0, n_total_clients=7
#
# NaN/Inf early-stop: _local_loop.py + fedscam._sam_train now raise
# NonFiniteLossError on divergent loss. With --continue-on-cell-failure
# semantics, a divergent cell's chain continues with the next cell.

set -e
cd ~/fl-oran-tmc
source .venv/bin/activate

OUTDIR="artifacts/v7_sam_family"
LOGDIR="logs"
PARQ="/dev/shm/coloran_raw_unified.parquet"
mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -f "$PARQ" ]]; then
    echo "ERROR: $PARQ missing — copy from \$HOME/data first"
    exit 1
fi

# CPU per-cell threading: 16 cores / 4 cells = 4 threads each
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Disable torch.compile / dynamo / inductor.
# V100 here has torch 2.4.0+cu121 from the standard PyTorch wheel WITHOUT
# triton bundled — `reduce-overhead` mode (fl_v7's LSTM arch-conditional
# default) hits ``RuntimeError: Cannot find a working triton installation``
# in the inductor scheduler ~140s into each cell (audit 2026-05-17).
# Disabling dynamo is preferable to installing triton because:
#   (a) Triton wheel for SM 7.0 + cu121 is not guaranteed compatible.
#   (b) Eager-mode loss is ~20-30% wall, acceptable for a 60-cell sweep.
#   (c) SAM 2-backward + EMA deepcopy under torch.compile have known
#       graph-break risks anyway — eager is the safer reference.
export TORCHDYNAMO_DISABLE=1

FEDSCAM_KW='{"rho_max":0.05,"alpha_rho":1.0,"gamma":1.0,"beta_align":0.8,"kappa":1.0}'
FEDGMT_KW='{"alpha_ema":0.95,"gamma_kl":1.0,"tau":3.0,"beta":10.0,"n_total_clients":7}'

# 60 cells = 2 algos × 6 partitions × 5 seeds.
# Cell format: "algo:partition:alpha:seed" — alpha is "0" placeholder for IID.
ALGOS=("fedscam" "fedgmt")
# 6 partitions; alpha=0 marks IID (sentinel — partition string disambiguates).
PARTITIONS=("iid:0" "dirichlet:0.05" "dirichlet:0.10" "dirichlet:0.50" \
            "dirichlet:1.00" "dirichlet:5.00")
SEEDS=(42 0 1 2 3)

# Build the full 60-cell list (order: outer algo, then partition, then seed).
cells=()
for algo in "${ALGOS[@]}"; do
    for partspec in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cells+=("${algo}:${partspec}:${seed}")
        done
    done
done

# Round-robin assign cells across 4 GPUs (15 per GPU).
declare -a CHAIN0 CHAIN1 CHAIN2 CHAIN3
for i in "${!cells[@]}"; do
    case $((i % 4)) in
        0) CHAIN0+=("${cells[$i]}") ;;
        1) CHAIN1+=("${cells[$i]}") ;;
        2) CHAIN2+=("${cells[$i]}") ;;
        3) CHAIN3+=("${cells[$i]}") ;;
    esac
done

echo "Total cells: ${#cells[@]}"
echo "CHAIN0 (gpu0): ${#CHAIN0[@]} cells"
echo "CHAIN1 (gpu1): ${#CHAIN1[@]} cells"
echo "CHAIN2 (gpu2): ${#CHAIN2[@]} cells"
echo "CHAIN3 (gpu3): ${#CHAIN3[@]} cells"

# Canonical cell name (must match _v7_cell_metadata.cell_name()).
# IID: v7_lstm_<algo>_iid_n7_s<seed>
# Dirichlet: v7_lstm_<algo>_dirichlet_a<X>p<YY>_n7_s<seed>  e.g. a0p50
build_name() {
    local algo=$1 partition=$2 alpha=$3 seed=$4
    if [[ "$partition" == "iid" ]]; then
        echo "v7_lstm_${algo}_iid_n7_s${seed}"
    else
        local alpha_tag
        alpha_tag=$(printf "%.2f" "$alpha" | sed 's/\./p/')
        echo "v7_lstm_${algo}_dirichlet_a${alpha_tag}_n7_s${seed}"
    fi
}

run_cell() {
    local gpu=$1 algo=$2 partition=$3 alpha=$4 seed=$5
    local name
    name=$(build_name "$algo" "$partition" "$alpha" "$seed")
    local cell_dir="$OUTDIR/$name"
    if [[ -f "$cell_dir/summary.json" ]]; then
        echo "[gpu$gpu] SKIP $name (exists)"
        return 0
    fi

    local algo_kw
    case "$algo" in
        fedscam) algo_kw="$FEDSCAM_KW" ;;
        fedgmt)  algo_kw="$FEDGMT_KW"  ;;
        *) echo "[gpu$gpu] BUG: unknown algo $algo"; return 1 ;;
    esac

    local partition_args
    if [[ "$partition" == "iid" ]]; then
        partition_args=(--partition-mode iid --n-clients 7)
    else
        partition_args=(--partition-mode dirichlet --n-clients 7 --alpha "$alpha")
    fi

    echo "[gpu$gpu] START $name (algo=$algo part=$partition alpha=$alpha seed=$seed)"
    local t0
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES="$gpu" \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       python experiments/run_v7_fl_arch_sweep.py \
            --arch lstm \
            --algorithm "$algo" \
            --algo-kwargs "$algo_kw" \
            "${partition_args[@]}" \
            --num-rounds 100 \
            --clients-per-round 5 \
            --max-steps-per-round 50 \
            --batch-size 64 \
            --lr 5.0e-4 \
            --lr-warmup-rounds 3 \
            --sample-ratio 1.0 \
            --threshold 0.10 \
            --seq-len 5 \
            --seed "$seed" \
            --unified-parquet "$PARQ" \
            --output-dir "$OUTDIR" \
            --pos-weight-split train \
            --mixed-precision bf16 \
            --name "$name" \
            > "$LOGDIR/v100_sam_gpu${gpu}_${name}.log" 2>&1
    then
        local dt=$(( $(date +%s) - t0 ))
        echo "[gpu$gpu] DONE $name in ${dt}s"
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        # Don't propagate failure — chain continues. NaN cells get marked
        # via missing summary.json which is re-detectable on rerun.
        echo "[gpu$gpu] FAIL $name rc=$rc in ${dt}s (see log)"
    fi
}

run_chain() {
    local gpu=$1
    shift
    local cells=("$@")
    for spec in "${cells[@]}"; do
        local algo="${spec%%:*}"
        local rest="${spec#*:}"
        local partition="${rest%%:*}"
        rest="${rest#*:}"
        local alpha="${rest%%:*}"
        local seed="${rest##*:}"
        run_cell "$gpu" "$algo" "$partition" "$alpha" "$seed"
    done
    echo "[gpu$gpu] CHAIN COMPLETE"
}

echo "=== SAM-family launching 4 parallel chains at $(date) ==="

run_chain 0 "${CHAIN0[@]}" > "$LOGDIR/v100_sam_chain0.log" 2>&1 &
PID0=$!
run_chain 1 "${CHAIN1[@]}" > "$LOGDIR/v100_sam_chain1.log" 2>&1 &
PID1=$!
run_chain 2 "${CHAIN2[@]}" > "$LOGDIR/v100_sam_chain2.log" 2>&1 &
PID2=$!
run_chain 3 "${CHAIN3[@]}" > "$LOGDIR/v100_sam_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: $PID0 $PID1 $PID2 $PID3"

# Per-PID `wait || true` so one chain's failure doesn't make set -e exit
# the launcher early (`feedback_audit_before_launch.md` finding 8).
for p in $PID0 $PID1 $PID2 $PID3; do
    wait "$p" || echo "[wait] chain pid=$p exited non-zero"
done

echo "=== SAM-family complete at $(date) ==="
echo "Cells produced in $OUTDIR:"
ls "$OUTDIR" | wc -l
