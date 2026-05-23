#!/usr/bin/env bash
# no-BLER leakage-control ablation (V100, eager fp16). Drops dl_bler/ul_bler from
# the model input (target still built from ul_bler at t+1, so labels unchanged) and
# re-runs the factorial. If run_dirichlet (intact) STILL >> dirichlet (row) without
# the BLER feature, the inverted-alpha gap is channel-state sequence-integrity, not a
# BLER-rate confound. Hypers mirror scripts/prea1/run_factorial_grid.sh exactly.
# 25 cells: {run_dirichlet, dirichlet} x {0.1, 1.0} x 5 seeds + iid x 5 seeds.
set -u
cd ~/fl-oran-tmc || exit 1
OUT=artifacts/prea1_nobler_ablation
mkdir -p "$OUT"
COMMON="--arch lstm --algorithm fedavg --n-clients 7 --num-rounds 100 --clients-per-round 5 \
--max-steps-per-round 50 --batch-size 64 --lr 5e-4 --lr-warmup-rounds 3 --grad-clip 1.0 \
--seq-len 5 --sample-ratio 1.0 --threshold 0.10 --pos-weight-split train \
--mixed-precision fp16 --drop-continuous dl_bler,ul_bler --device cuda --output-dir $OUT"

JOBS=()
for s in 0 1 2 3 4; do
  JOBS+=("run_dirichlet|0.1|$s"); JOBS+=("run_dirichlet|1.0|$s")
  JOBS+=("dirichlet|0.1|$s");     JOBS+=("dirichlet|1.0|$s")
  JOBS+=("iid|0.5|$s")            # iid ignores alpha (natural-by-BS anchor)
done

run_chain() {
  local gpu=$1
  local idx=$1
  while [ "$idx" -lt "${#JOBS[@]}" ]; do
    IFS='|' read -r mode a s <<< "${JOBS[$idx]}"
    echo "[gpu$gpu] START $mode a=$a s=$s $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" TORCHDYNAMO_DISABLE=1 \
      .venv/bin/python experiments/run_v7_fl_arch_sweep.py \
      --partition-mode "$mode" --alpha "$a" --seed "$s" $COMMON \
      >> "$OUT/gpu${gpu}.log" 2>&1
    echo "[gpu$gpu] DONE  $mode a=$a s=$s $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    idx=$((idx + 4))
  done
}

for gpu in 0 1 2 3; do run_chain "$gpu" & done
wait
echo "ALL_NOBLER_DONE $(date +%H:%M:%S)" | tee -a "$OUT/grid.log"
