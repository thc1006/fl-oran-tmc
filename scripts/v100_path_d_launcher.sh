#!/usr/bin/env bash
# Path D V100 launcher — 540-cell SAM-family scale-up sweep.
#
# Spec:    experiments/specs/path_d_full.yaml
# Cells:   540 = 3 archs (lstm, mamba, spiking_expand2)
#                × 3 algos (fedscam, fedgmt, fedmoswa)
#                × 6 partitions (iid + 5 Dirichlet)
#                × 10 seeds [0..8, 42]
# Hardware: 4× Tesla V100-SXM2-32GB
# Output:   ~/fl-oran-tmc/artifacts/v7_sam_family/<cell_name>/
# Logs:     ~/fl-oran-tmc/logs/v100_path_d_chain{0..3}.log
#
# Wall budget: each chain holds 135 cells (auto-balanced to 45 LSTM +
# 45 Mamba + 45 Spiking via run_v7_phase_sweep --shard N/4).
# Per-cell walls: LSTM ~530s, Mamba ~1100s, Spiking ~2700s.
# Per-chain wall: 45·(530+1100+2700) = 194850s ≈ 54.1 hr.
# Total elapsed: ~54 hr (chains run in parallel).
# With --skip-completed: existing 60 LSTM × {fedscam,fedgmt} × seeds
# {0,1,2,3,42} cells are preserved → effective new wall ≈ 52 hr.
#
# Efficiency vs the previous per-cell-process v100_sam_family_launcher.sh:
# - This script runs ONE python process per chain via run_v7_phase_sweep
#   (spec-driven). CUDA context + cuDNN cache persist across cells in
#   each chain → ~10s saved per cell × 480 cells = ~80 min saved.
# - TORCHDYNAMO_DISABLE=1: V100 lacks bundled triton in torch 2.4+cu121.
#   FedMoSWA's per-step LR mutation would break compile graph capture
#   anyway. Eager mode is the safer reference.
# - CPU threads: 4 per chain × 4 chains = 16 threads. V100 cluster nodes
#   typically have 32–64 cores so we're not CPU-bound on data prep.
# - bf16 amp: emulated on sm_70 (~30% slower than fp16) but stable
#   without GradScaler. fp16 would require adding GradScaler to
#   _local_loop.run_local_sgd — deferred (correctness > speed).
# - NaN guard: _local_loop.NonFiniteLossError raises on divergent loss;
#   --continue-on-cell-failure marks the cell failed and the chain moves
#   to the next cell without poisoning state.
#
# Safety:
# - set -u catches unset variables (typo guard)
# - per-PID wait || true so one chain's failure doesn't kill the others
# - 4 chains × independent state → cross-chain bugs are confined

set -eu
cd ~/fl-oran-tmc
source .venv/bin/activate

SPEC=experiments/specs/path_d_full.yaml
OUTDIR=artifacts/v7_sam_family
LOGDIR=logs
PARQ=/dev/shm/coloran_raw_unified.parquet

mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -f "$PARQ" ]]; then
    echo "ERROR: $PARQ missing — copy from \$HOME/data first:"
    echo "  cp ~/data/coloran_raw_unified.parquet $PARQ"
    exit 1
fi

if [[ ! -f "$SPEC" ]]; then
    echo "ERROR: $SPEC missing — check git state"
    exit 1
fi

# CPU threading: 4 threads per chain × 4 chains = 16 threads total.
# Adjust upward if the V100 node has many cores (32+ recommended for
# data-prep parallelism). OMP/MKL/OPENBLAS pin matmul + numpy threads.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Disable torch.compile / dynamo / inductor — see file header rationale.
export TORCHDYNAMO_DISABLE=1

# CUDA backend tweaks for fixed-shape workloads.
# - cudnn benchmark: faster after warmup (seq_len=5, batch=64 fixed).
# - matmul TF32: no-op on V100 sm_70 (Ampere+ only) but defensive default.
# fl_v7 sets these via V7Config; we re-export here for clarity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pre-flight: verify the spec expands + all 540 cells construct cleanly.
# Cheap (< 5s) — catches missing kwargs / typos before any GPU time.
echo "=== pre-flight: dry-instantiate 540 cells (no training) ==="
python experiments/run_v7_phase_sweep.py \
    --spec "$SPEC" \
    --output-dir "$OUTDIR" \
    --unified-parquet "$PARQ" \
    --dry-run \
    > "$LOGDIR/v100_path_d_preflight.log" 2>&1
echo "pre-flight ok (see $LOGDIR/v100_path_d_preflight.log)"

run_chain() {
    local gpu=$1
    local shard_n=$2
    echo "[gpu$gpu] CHAIN STARTING shard=$shard_n/4 at $(date)"
    local t0
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    TORCHDYNAMO_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python experiments/run_v7_phase_sweep.py \
        --spec "$SPEC" \
        --output-dir "$OUTDIR" \
        --unified-parquet "$PARQ" \
        --shard "${shard_n}/4" \
        --skip-existing-summary \
        --continue-on-cell-failure \
        --summary-tag "path_d_chain${shard_n}"
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[gpu$gpu] CHAIN DONE shard=$shard_n/4 rc=$rc dt=${dt}s at $(date)"
}

echo "=== Path D launching 4 parallel chains at $(date) ==="
echo "    spec=$SPEC"
echo "    outdir=$OUTDIR"
echo "    expected wall: ~54 hr per chain (parallel)"

run_chain 0 1 > "$LOGDIR/v100_path_d_chain0.log" 2>&1 &
PID0=$!
run_chain 1 2 > "$LOGDIR/v100_path_d_chain1.log" 2>&1 &
PID1=$!
run_chain 2 3 > "$LOGDIR/v100_path_d_chain2.log" 2>&1 &
PID2=$!
run_chain 3 4 > "$LOGDIR/v100_path_d_chain3.log" 2>&1 &
PID3=$!

echo "PIDs: gpu0=$PID0 gpu1=$PID1 gpu2=$PID2 gpu3=$PID3"

# Per-PID `wait || true` so one chain's failure doesn't make set -e exit
# the launcher early. Each chain has --continue-on-cell-failure so the
# only way a chain exits non-zero is a launcher-level error.
for p in $PID0 $PID1 $PID2 $PID3; do
    wait "$p" || echo "[wait] chain pid=$p exited non-zero"
done

echo "=== Path D complete at $(date) ==="
echo "Cells produced in $OUTDIR:"
ls "$OUTDIR" | wc -l
echo
echo "Next steps:"
echo "  scp -P 51419 leo07010@203.145.216.194:'~/fl-oran-tmc/$OUTDIR/v7_*/summary.json' ./local_mirror/"
echo "  python scripts/aggregate_v7_results.py --sweep-dir $OUTDIR \\"
echo "      --output docs/RESULTS_V7_PATH_D.md"
