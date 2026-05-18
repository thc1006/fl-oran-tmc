#!/usr/bin/env bash
# V100 extended sweep waiter + auto-aggregator (task #41).
#
# DRY-RUN BY DEFAULT. Set EXECUTE=1 to actually launch the polling loop
# and auto-aggregator. Designed to be backgrounded with `nohup` or `tmux`
# on the 4060-dev workstation while the V100 cluster runs the extended
# sweep.
#
# What it does (in EXECUTE mode):
#   1. Polls V100 (`leo07010@203.145.216.194:51419` via Tailscale-routable
#      public IP) every POLL_INTERVAL seconds, counting `summary.json`
#      files under V100's `~/fl-oran-tmc/artifacts/v7_sam_family/` (the
#      Path D output directory shared between core 540 cells and
#      extended 360 cells).
#   2. Detects completion via 2 alternative triggers:
#      a) Cell count >= TARGET_CELLS (default 900 = 540 core + 360 ext)
#      b) Cell count >= MIN_FOR_PARTIAL (default 800) AND no cell has
#         been updated in STALE_THRESHOLD_S seconds (default 1800 = 30 min)
#         → sweep likely stalled with most cells done; analyse partial.
#   3. On completion:
#      a) Rsync `v7_sam_family/` from V100 to local
#         `artifacts/v7_sam_family/` (idempotent; rsync --update so
#         we don't clobber locally-newer files).
#      b) Run `scripts/aggregate_v7_results.py` against the synced cells
#         → `docs/RESULTS_V7_PATH_D_EXTENDED.md`.
#      c) Run `scripts/aggregate_path_d.py` against the synced cells
#         → `docs/RESULTS_V7_PATH_D_PAPER.md`.
#   4. Logs to `logs/v100_extended_waiter.log` throughout.
#
# Idempotency:
#   - SIGNAL_FILE (`.waiter_done` in OUTDIR) is created when the waiter
#     fires its auto-aggregator; subsequent invocations short-circuit
#     and exit cleanly. Delete the file to re-trigger.
#
# Why dry-run-by-default:
#   - Polling V100 every 5 min creates SSH login records; users may want
#     to inspect the plan before authorising sustained connections.
#   - The aggregator run touches `docs/RESULTS_*.md` files; users may
#     want to dry-review before letting it commit to disk.
#
# Why a bash script (not Python):
#   - Pure shell + rsync + ssh is the minimum-dep path on the 4060-dev
#     workstation (no extra Python deps needed).
#   - The dashboard script (`scripts/sweep_dashboard.py`) is the
#     interactive variant; this is the headless complement.

set -eu

DRY_RUN_MODE="${EXECUTE:-0}"

# --- tunables (env-overridable) ---
POLL_INTERVAL="${POLL_INTERVAL:-300}"        # 5 min between polls
TARGET_CELLS="${TARGET_CELLS:-900}"          # 540 core + 360 ext
MIN_FOR_PARTIAL="${MIN_FOR_PARTIAL:-800}"    # accept partial >= 800
STALE_THRESHOLD_S="${STALE_THRESHOLD_S:-1800}"  # 30 min
MAX_POLLS="${MAX_POLLS:-2880}"               # 10 days worst-case

# --- expected V100 + local environment ---
EXPECTED_USER="leo07010"
SSH_HOST="${EXPECTED_USER}@203.145.216.194"
SSH_PORT="51419"
V100_OUTDIR="artifacts/v7_sam_family"

LOCAL_REPO="$HOME/fl-oran-tmc"
LOCAL_OUTDIR="$LOCAL_REPO/artifacts/v7_sam_family"
LOCAL_LOGDIR="$LOCAL_REPO/logs"
LOG_FILE="$LOCAL_LOGDIR/v100_extended_waiter.log"
SIGNAL_FILE="$LOCAL_OUTDIR/.waiter_done"

# --- header banner ---
echo "================================================================"
echo "V100 EXTENDED WAITER + AUTO-AGGREGATOR (task #41)"
echo "================================================================"
if [[ "$DRY_RUN_MODE" != "1" ]]; then
    echo ">>> DRY-RUN MODE — set EXECUTE=1 to actually start polling <<<"
fi
echo ""
echo "Tunables:"
echo "  poll_interval        = ${POLL_INTERVAL}s"
echo "  target_cells         = ${TARGET_CELLS}"
echo "  min_for_partial      = ${MIN_FOR_PARTIAL}"
echo "  stale_threshold_s    = ${STALE_THRESHOLD_S}s"
echo "  max_polls            = ${MAX_POLLS}"
echo ""
echo "Paths:"
echo "  V100 source          = ${SSH_HOST}:${V100_OUTDIR}/"
echo "  local mirror         = ${LOCAL_OUTDIR}/"
echo "  signal file          = ${SIGNAL_FILE}"
echo "  log                  = ${LOG_FILE}"
echo ""

# --- DRY-RUN: print plan + exit ---
if [[ "$DRY_RUN_MODE" != "1" ]]; then
    echo "DRY-RUN: would execute the following loop:"
    echo ""
    echo "  while iteration < max_polls AND signal_file does NOT exist:"
    echo "    n_cells = ssh $SSH_HOST 'cd /home/$EXPECTED_USER/fl-oran-tmc &&"
    echo "                              find $V100_OUTDIR -name summary.json | wc -l'"
    echo "    latest_mtime = ssh $SSH_HOST 'cd /home/$EXPECTED_USER/fl-oran-tmc &&"
    echo "                                  find $V100_OUTDIR -name summary.json"
    echo "                                  -printf %T@\\\\n | sort -rn | head -1'"
    echo "    if n_cells >= $TARGET_CELLS:"
    echo "      → trigger DONE"
    echo "    elif n_cells >= $MIN_FOR_PARTIAL AND"
    echo "         (now() - latest_mtime) > ${STALE_THRESHOLD_S}s:"
    echo "      → trigger DONE (partial)"
    echo "    sleep $POLL_INTERVAL"
    echo ""
    echo "  on DONE:"
    echo "    1) rsync -avz --update ${SSH_HOST}:${V100_OUTDIR}/ ${LOCAL_OUTDIR}/"
    echo "    2) cd $LOCAL_REPO && source .venv/bin/activate"
    echo "    3) python scripts/aggregate_v7_results.py \\"
    echo "         --sweep-dir $LOCAL_OUTDIR \\"
    echo "         --out-md docs/RESULTS_V7_PATH_D_EXTENDED.md"
    echo "    4) python scripts/aggregate_path_d.py \\"
    echo "         --sweep-dir $LOCAL_OUTDIR \\"
    echo "         --out-md docs/RESULTS_V7_PATH_D_PAPER.md"
    echo "    5) touch $SIGNAL_FILE     # idempotency"
    echo "    6) echo 'DONE — see docs/RESULTS_V7_PATH_D_*.md' to log"
    echo ""
    echo "  on MAX_POLLS reached without DONE:"
    echo "    write 'TIMEOUT after $((MAX_POLLS * POLL_INTERVAL))s' to log + exit non-zero"
    echo ""
    echo "DRY-RUN END. To actually launch:"
    echo "  nohup EXECUTE=1 ./scripts/v100_extended_waiter.sh \\"
    echo "    > $LOG_FILE 2>&1 &"
    echo "  disown"
    echo "  # Then monitor:  tail -f $LOG_FILE"
    exit 0
fi

# --- live path: actually run polling loop ---
mkdir -p "$LOCAL_OUTDIR" "$LOCAL_LOGDIR"

# Idempotency: if signal file already exists, don't re-run the aggregator.
if [[ -f "$SIGNAL_FILE" ]]; then
    echo "$(date -u +%FT%TZ) [SHORT-CIRCUIT] signal file exists ($SIGNAL_FILE) — aggregator already ran"
    echo "Delete the signal file to re-trigger:  rm $SIGNAL_FILE"
    exit 0
fi

# SSH options shared with sweep_dashboard.py for consistency.
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=8)

poll_remote() {
    # Print "<n_cells> <latest_mtime_epoch>" to stdout
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
        "cd /home/${EXPECTED_USER}/fl-oran-tmc && \
         echo -n \"\$(find ${V100_OUTDIR} -name summary.json 2>/dev/null | wc -l) \" && \
         find ${V100_OUTDIR} -name summary.json -printf '%T@\n' 2>/dev/null | \
             sort -rn | head -1"
}

log() {
    echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG_FILE"
}

trigger_aggregator() {
    local mode="$1"   # "full" or "partial"
    log "[AGGREGATOR] mode=$mode — starting rsync + aggregate"

    log "[AGGREGATOR] step 1: rsync v7_sam_family from V100"
    rsync -avz --update -e "ssh ${SSH_OPTS[*]}" \
        "${SSH_HOST}:/home/${EXPECTED_USER}/fl-oran-tmc/${V100_OUTDIR}/" \
        "${LOCAL_OUTDIR}/" \
        >> "$LOG_FILE" 2>&1

    cd "$LOCAL_REPO"
    # shellcheck disable=SC1091
    source .venv/bin/activate

    log "[AGGREGATOR] step 2: aggregate_v7_results.py"
    python scripts/aggregate_v7_results.py \
        --sweep-dir "$LOCAL_OUTDIR" \
        --out-md docs/RESULTS_V7_PATH_D_EXTENDED.md \
        >> "$LOG_FILE" 2>&1 || log "[AGGREGATOR] aggregate_v7_results.py failed"

    log "[AGGREGATOR] step 3: aggregate_path_d.py"
    python scripts/aggregate_path_d.py \
        --sweep-dir "$LOCAL_OUTDIR" \
        --out-md docs/RESULTS_V7_PATH_D_PAPER.md \
        >> "$LOG_FILE" 2>&1 || log "[AGGREGATOR] aggregate_path_d.py failed"

    touch "$SIGNAL_FILE"
    log "[AGGREGATOR] step 4: signal file written — $SIGNAL_FILE"
    log "[DONE] $mode aggregation complete. Outputs:"
    log "  docs/RESULTS_V7_PATH_D_EXTENDED.md"
    log "  docs/RESULTS_V7_PATH_D_PAPER.md"
}

log "[START] V100 extended waiter — polling every ${POLL_INTERVAL}s, target ${TARGET_CELLS}"

iteration=0
while [[ $iteration -lt $MAX_POLLS ]]; do
    iteration=$((iteration + 1))

    # Probe remote
    remote_state=$(poll_remote 2>>"$LOG_FILE") || {
        log "[poll #${iteration}] ssh failed; sleeping and retrying"
        sleep "$POLL_INTERVAL"
        continue
    }
    n_cells=$(echo "$remote_state" | awk '{print $1}')
    latest_mtime=$(echo "$remote_state" | awk '{print $2}')

    # Compute staleness (default 0 if mtime unparseable)
    now=$(date +%s)
    if [[ -n "${latest_mtime:-}" ]] && [[ "$latest_mtime" =~ ^[0-9.]+$ ]]; then
        staleness=$(( now - ${latest_mtime%.*} ))
    else
        staleness=0
    fi

    log "[poll #${iteration}] cells=${n_cells} staleness=${staleness}s"

    # Trigger A: full target reached
    if [[ "$n_cells" -ge "$TARGET_CELLS" ]]; then
        log "[TRIGGER] full target reached ($n_cells >= $TARGET_CELLS)"
        trigger_aggregator "full"
        exit 0
    fi

    # Trigger B: partial + stalled
    if [[ "$n_cells" -ge "$MIN_FOR_PARTIAL" ]] && \
       [[ "$staleness" -ge "$STALE_THRESHOLD_S" ]]; then
        log "[TRIGGER] partial+stalled (cells=$n_cells, staleness=${staleness}s)"
        trigger_aggregator "partial"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done

log "[TIMEOUT] reached MAX_POLLS=${MAX_POLLS} (${POLL_INTERVAL}s each) without DONE"
exit 1
