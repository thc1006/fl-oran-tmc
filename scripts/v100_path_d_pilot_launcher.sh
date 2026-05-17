#!/usr/bin/env bash
# Path D V100 PILOT launcher — 4-cell production-conditions validation.
#
# Runs the 4 highest-risk NEW (algo × arch) combinations on V100 at
# sample_ratio=1.0 (matching Phase 5 + the full Path D sweep) BEFORE
# committing to the 52-hour full sweep. The 4 cells are chosen to cover:
#
#   GPU 0:  mamba × fedmoswa × IID × seed=42
#           (FedMoSWA × new arch, simplest partition)
#   GPU 1:  mamba × fedscam × dirichlet α=0.05 × seed=42
#           (SAM 2-backward × new arch × most heterogeneous Dirichlet)
#   GPU 2:  spiking_expand2 × fedmoswa × IID × seed=42
#           (highest-risk: spike binarization × cyclical-LR ×
#            SCAFFOLD-style variance reduction)
#   GPU 3:  spiking_expand2 × fedscam × dirichlet α=0.05 × seed=42
#           (SAM perturbation on binarized Spiking output × extreme α)
#
# Per-cell wall: Mamba ~1100s (18 min), Spiking ~2700s (45 min).
# Total pilot wall: ~45 min (4 GPUs parallel, longest cell = Spiking).
#
# Output goes to artifacts/v7_sam_family/<cell_name>/ so the FULL sweep
# launcher will automatically --skip-existing-summary on these 4 cells.
#
# Pass criteria for proceeding to full sweep:
#   - All 4 cells emit summary.json with finite test_auc
#   - No NonFiniteLossError raised
#   - test_auc > 0.55 (rough sanity floor; far below baseline IID 0.91)
#   - V100 master GPU util > 30% during training (cards not stalled)
#
# Fail mode: any NaN/Inf → fix root cause before launching full sweep.
# The --continue-on-cell-failure flag prevents one bad cell from killing
# the others.

set -eu
cd ~/fl-oran-tmc
source .venv/bin/activate

OUTDIR=artifacts/v7_sam_family
LOGDIR=logs
PARQ=/dev/shm/coloran_raw_unified.parquet

mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -f "$PARQ" ]]; then
    echo "ERROR: $PARQ missing — copy from \$HOME/data first:"
    echo "  cp ~/data/coloran_raw_unified.parquet $PARQ"
    exit 1
fi

# Required hyperparameters per paper §6.1 (FedMoSWA) and existing
# v100_sam_family_launcher.sh defaults (FedSCAM).
FEDSCAM_KW='{"rho_max":0.05,"alpha_rho":1.0,"gamma":1.0,"beta_align":0.8,"kappa":1.0}'
FEDMOSWA_KW='{"rho":0.1,"alpha_la":1.5,"gamma":0.2,"n_total_clients":7}'

# CPU threading: 4 threads per chain × 4 chains = 16 threads.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# torch.compile / dynamo disabled — V100 + FedMoSWA cyclical-LR.
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_pilot_cell() {
    # Args: gpu_idx arch algo partition alpha algo_kwargs
    local gpu=$1 arch=$2 algo=$3 partition=$4 alpha=$5 algo_kw=$6

    local name
    if [[ "$partition" == "iid" ]]; then
        name="v7_${arch}_${algo}_iid_n7_s42"
    else
        local alpha_tag
        alpha_tag=$(printf "%.2f" "$alpha" | sed 's/\./p/')
        name="v7_${arch}_${algo}_dirichlet_a${alpha_tag}_n7_s42"
    fi

    local partition_args=()
    if [[ "$partition" == "iid" ]]; then
        partition_args=(--partition-mode iid --n-clients 7)
    else
        partition_args=(--partition-mode dirichlet --n-clients 7 --alpha "$alpha")
    fi

    echo "[gpu$gpu] START $name ($algo × $arch × $partition)"
    local t0
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES="$gpu" \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       TORCHDYNAMO_DISABLE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       python experiments/run_v7_fl_arch_sweep.py \
            --arch "$arch" \
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
            --seed 42 \
            --unified-parquet "$PARQ" \
            --output-dir "$OUTDIR" \
            --pos-weight-split train \
            --mixed-precision bf16 \
            --name "$name" \
            > "$LOGDIR/v100_path_d_pilot_${name}.log" 2>&1
    then
        local dt=$(( $(date +%s) - t0 ))
        echo "[gpu$gpu] DONE $name in ${dt}s"
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        echo "[gpu$gpu] FAIL $name rc=$rc in ${dt}s (see log)"
    fi
}

echo "=== Path D pilot launching 4 parallel cells at $(date) ==="
echo "    GPU 0: mamba × fedmoswa × IID"
echo "    GPU 1: mamba × fedscam × α=0.05"
echo "    GPU 2: spiking_expand2 × fedmoswa × IID"
echo "    GPU 3: spiking_expand2 × fedscam × α=0.05"
echo "    Expected wall: ~45 min (Spiking is longest)"

run_pilot_cell 0 mamba           fedmoswa iid       0    "$FEDMOSWA_KW" \
    > "$LOGDIR/v100_path_d_pilot_chain0.log" 2>&1 &
PID0=$!
run_pilot_cell 1 mamba           fedscam  dirichlet 0.05 "$FEDSCAM_KW" \
    > "$LOGDIR/v100_path_d_pilot_chain1.log" 2>&1 &
PID1=$!
run_pilot_cell 2 spiking_expand2 fedmoswa iid       0    "$FEDMOSWA_KW" \
    > "$LOGDIR/v100_path_d_pilot_chain2.log" 2>&1 &
PID2=$!
run_pilot_cell 3 spiking_expand2 fedscam  dirichlet 0.05 "$FEDSCAM_KW" \
    > "$LOGDIR/v100_path_d_pilot_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: gpu0=$PID0 gpu1=$PID1 gpu2=$PID2 gpu3=$PID3"

for p in $PID0 $PID1 $PID2 $PID3; do
    wait "$p" || echo "[wait] chain pid=$p exited non-zero"
done

echo "=== Path D pilot complete at $(date) ==="
echo "Cells produced:"
for name in \
    v7_mamba_fedmoswa_iid_n7_s42 \
    v7_mamba_fedscam_dirichlet_a0p05_n7_s42 \
    v7_spiking_expand2_fedmoswa_iid_n7_s42 \
    v7_spiking_expand2_fedscam_dirichlet_a0p05_n7_s42
do
    if [[ -f "$OUTDIR/$name/summary.json" ]]; then
        echo "  ✓ $name"
    else
        echo "  ✗ $name (missing summary.json)"
    fi
done
echo
echo "If all 4 cells passed, launch full sweep:"
echo "  bash scripts/v100_path_d_launcher.sh"
