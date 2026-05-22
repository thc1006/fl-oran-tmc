#!/usr/bin/env bash
# PREREG-A1 sequence-integrity factorial grid (V100, eager, fp16, matched hypers).
# 21 cells: run_dirichlet + (row) dirichlet at alpha{0.1,0.5,1.0} x seeds{0,1,2}
# + random_split x seeds{0,1,2}. Distributed round-robin over 4 GPU-chains.
# Pairs with already-done natural (E2, 0.916) + run_random (0.916).
set -u
cd ~/fl-oran-tmc || exit 1
OUT=artifacts/prea1_factorial
mkdir -p "$OUT"
COMMON="--arch lstm --algorithm fedavg --n-clients 7 --num-rounds 100 \
--clients-per-round 5 --max-steps-per-round 50 --batch-size 64 --lr 5e-4 \
--lr-warmup-rounds 3 --grad-clip 1.0 --seq-len 5 --sample-ratio 1.0 \
--threshold 0.10 --pos-weight-split train --mixed-precision fp16 --device cuda \
--output-dir $OUT"

JOBS=()
for a in 0.1 0.5 1.0; do
  for s in 0 1 2; do
    JOBS+=("run_dirichlet|$a|$s")
    JOBS+=("dirichlet|$a|$s")
  done
done
for s in 0 1 2; do
  JOBS+=("random_split|NA|$s")
done

run_chain() {
  local gpu=$1
  local idx=$1
  while [ "$idx" -lt "${#JOBS[@]}" ]; do
    IFS='|' read -r mode a s <<< "${JOBS[$idx]}"
    local alpha_arg=""
    [ "$a" != "NA" ] && alpha_arg="--alpha $a"
    echo "[gpu$gpu] START $mode a=$a s=$s $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" TORCHDYNAMO_DISABLE=1 \
      .venv/bin/python experiments/run_v7_fl_arch_sweep.py \
      --partition-mode "$mode" $alpha_arg --seed "$s" $COMMON \
      >> "$OUT/gpu${gpu}.log" 2>&1
    echo "[gpu$gpu] DONE  $mode a=$a s=$s $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    idx=$((idx + 4))
  done
}

for gpu in 0 1 2 3; do run_chain "$gpu" & done
wait
echo "ALL_FACTORIAL_DONE $(date +%H:%M:%S)" | tee -a "$OUT/grid.log"
