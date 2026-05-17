#!/usr/bin/env bash
# V100 extended pilot launcher — 4-cell smoke for xLSTM + Mamba-3 on V100.
#
# DRY-RUN BY DEFAULT. Set EXECUTE=1 to actually launch the pilot.
#
# Spec:    experiments/specs/path_d_extended_pilot.yaml
# Cells:   4 = 2 archs (xlstm, mamba3) × 2 algos (fedavg, fedscam)
#              × 1 partition (iid) × 1 seed (42)
# Hardware: 1× Tesla V100-SXM2-32GB (single GPU; 4 cells run sequentially)
# Output:   ~/fl-oran-tmc/artifacts/v7_path_d_extended_pilot/<cell_name>/
# Logs:     ~/fl-oran-tmc/logs/v100_extended_pilot.log
#
# Wall budget (placeholder, this run REFINES it):
#   xlstm cell  ~520s  (placeholder from 4060 smoke)
#   mamba3 cell ~854s  (placeholder from 4060 smoke, ~1.05× mamba)
#   Total wall: 2·520 + 2·854 ≈ 2748s ≈ 46 min on 1 V100
#
# Pre-flight: --dry-run on run_v7_phase_sweep.py verifies the 4 cells
# expand from spec without instantiating GPU. Always runs before the
# actual launch.
#
# Why single GPU (not 4-way shard like path D full):
#   - 4 cells is small; the sharding overhead dominates
#   - We want SEQUENTIAL completion order so wall-time numbers are
#     cleanly attributable (no GPU contention if one V100 lane stalls)
#   - Other 3 V100s likely still busy with Path D core sweep (#25) anyway
#
# Safety:
# - set -u catches unset variable typos
# - Pre-flight --dry-run catches spec errors before GPU time spent
# - --continue-on-cell-failure: one cell failing doesn't kill the rest
#   (we WANT to see all 4 outcomes for the pilot decision matrix)
# - --skip-existing-summary: re-running the script is idempotent

set -eu

DRY_RUN_MODE="${EXECUTE:-0}"

# --- header banner ---
echo "================================================================"
echo "V100 EXTENDED PILOT LAUNCHER — xLSTM + Mamba-3 × 4 cells"
echo "================================================================"
if [[ "$DRY_RUN_MODE" != "1" ]]; then
    echo ">>> DRY-RUN MODE — set EXECUTE=1 to actually launch on V100 <<<"
fi
echo ""

# --- expected V100 environment ---
EXPECTED_HOST="leo07010"
EXPECTED_REPO="$HOME/fl-oran-tmc"
SPEC=experiments/specs/path_d_extended_pilot.yaml
OUTDIR=artifacts/v7_path_d_extended_pilot
LOGDIR=logs
PARQ=/dev/shm/coloran_raw_unified.parquet
LOG_FILE="$LOGDIR/v100_extended_pilot.log"

echo "Planned config:"
echo "  spec    = $SPEC"
echo "  outdir  = $EXPECTED_REPO/$OUTDIR"
echo "  logfile = $EXPECTED_REPO/$LOG_FILE"
echo "  parquet = $PARQ"
echo "  cells   = 4 (xlstm + mamba3) × (fedavg + fedscam) × IID × s42"
echo ""

# --- dry-run path: print + exit ---
if [[ "$DRY_RUN_MODE" != "1" ]]; then
    echo "DRY-RUN: would execute the following 3 steps on V100 ($EXPECTED_HOST):"
    echo ""
    echo "  Step 1: pre-flight (cheap, no GPU):"
    echo "    cd $EXPECTED_REPO && source .venv/bin/activate"
    echo "    python experiments/run_v7_phase_sweep.py \\"
    echo "        --spec $SPEC \\"
    echo "        --output-dir $OUTDIR \\"
    echo "        --unified-parquet $PARQ \\"
    echo "        --dry-run"
    echo ""
    echo "  Step 2: launch on 1 V100 (sequential, ~46 min wall):"
    echo "    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
    echo "    export TORCHDYNAMO_DISABLE=1"
    echo "    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    echo "    CUDA_VISIBLE_DEVICES=0 \\"
    echo "      python experiments/run_v7_phase_sweep.py \\"
    echo "        --spec $SPEC \\"
    echo "        --output-dir $OUTDIR \\"
    echo "        --unified-parquet $PARQ \\"
    echo "        --skip-existing-summary \\"
    echo "        --continue-on-cell-failure \\"
    echo "        --summary-tag pilot_extended \\"
    echo "        > $LOG_FILE 2>&1 &"
    echo ""
    echo "  Step 3: monitor (poll for completion + summary):"
    echo "    tail -f $LOG_FILE"
    echo "    # When done: ls $OUTDIR | wc -l should show 4"
    echo ""
    echo "  Step 4 (post-pilot, after 4 cells complete):"
    echo "    python scripts/aggregate_v7_results.py \\"
    echo "        --sweep-dir $OUTDIR \\"
    echo "        --out-md docs/RESULTS_V7_PATHD_PILOT.md \\"
    echo "        --out-json $OUTDIR/aggregated.json"
    echo "    # Inspect test_auc + wall_time per cell → CHECKPOINT 3 (#50)"
    echo "    # Refine scripts/sweep_dashboard.py ARCH_WALL_FALLBACK_S"
    echo "    # if V100 wall != 530s/854s placeholders"
    echo ""
    echo "DRY-RUN END. Run \`EXECUTE=1 ./scripts/v100_extended_pilot_launcher.sh\`"
    echo "(on the V100 cluster) to actually fire."
    exit 0
fi

# --- live path: actually execute ---
cd "$EXPECTED_REPO"
source .venv/bin/activate

mkdir -p "$OUTDIR" "$LOGDIR"

if [[ ! -f "$PARQ" ]]; then
    echo "ERROR: $PARQ missing. Copy from \$HOME/data first:"
    echo "  cp ~/data/coloran_raw_unified.parquet $PARQ"
    exit 1
fi

if [[ ! -f "$SPEC" ]]; then
    echo "ERROR: $SPEC missing. Did you git pull?"
    exit 1
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Step 1: pre-flight dry-instantiate 4 cells (no GPU) ==="
python experiments/run_v7_phase_sweep.py \
    --spec "$SPEC" \
    --output-dir "$OUTDIR" \
    --unified-parquet "$PARQ" \
    --dry-run \
    > "$LOGDIR/v100_extended_pilot_preflight.log" 2>&1
echo "pre-flight OK (see $LOGDIR/v100_extended_pilot_preflight.log)"
echo ""

echo "=== Step 2: launch pilot at $(date) ==="
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 \
python experiments/run_v7_phase_sweep.py \
    --spec "$SPEC" \
    --output-dir "$OUTDIR" \
    --unified-parquet "$PARQ" \
    --skip-existing-summary \
    --continue-on-cell-failure \
    --summary-tag pilot_extended \
    > "$LOG_FILE" 2>&1 &
PID=$!

echo "[pid=$PID] pilot started; tailing $LOG_FILE"
echo ""

wait "$PID" || echo "[wait] pilot pid=$PID exited non-zero"
dt=$(( $(date +%s) - t0 ))

echo ""
echo "=== Step 3: pilot complete at $(date) (wall=${dt}s) ==="
echo "Cells produced in $OUTDIR:"
ls "$OUTDIR" 2>/dev/null | wc -l
echo ""
echo "Per-cell summary.json paths:"
find "$OUTDIR" -name "summary.json" -printf "  %p (mtime: %TY-%Tm-%Td %TH:%TM)\n" | head -10
echo ""
echo "Next steps (post-pilot CHECKPOINT 3, task #50):"
echo "  1. Aggregate pilot results into a Markdown report:"
echo "     python scripts/aggregate_v7_results.py \\"
echo "         --sweep-dir $OUTDIR \\"
echo "         --out-md docs/RESULTS_V7_PATHD_PILOT.md"
echo "  2. Inspect per-cell test_auc + wall_time."
echo "  3. If GO: update scripts/sweep_dashboard.py ARCH_WALL_FALLBACK_S"
echo "     with the empirical V100 wall numbers; launch #40."
echo "  4. If NO-GO: debug before re-attempting #40."
