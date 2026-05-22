#!/usr/bin/env bash
# Multi-arch confirmation of the sequence-integrity artifact (V100, eager).
# Mamba + Spiking-SSM x {run_dirichlet (intact), dirichlet (row, corrupt)}
# x alpha{0.1, 1.0} x seeds{0,1,2} = 24 cells. Confirms the artifact (run_dir
# >> dir within each arch) is arch-general, not LSTM-specific.
# Precision: both archs use full fp32 (--mixed-precision off). Mamba's selective
# scan NaNs in fp16; Spiking's surrogate gradients need full precision; valid AMP
# tokens are {off, fp16, bf16} ("fp32" is NOT a token). V100 fp32 is fast + stable.
set -u
cd ~/fl-oran-tmc || exit 1
OUT=artifacts/prea1_factorial_multiarch
mkdir -p "$OUT"
COMMON="--algorithm fedavg --n-clients 7 --num-rounds 100 --clients-per-round 5 \
--max-steps-per-round 50 --batch-size 64 --lr 5e-4 --lr-warmup-rounds 3 \
--grad-clip 1.0 --seq-len 5 --sample-ratio 1.0 --threshold 0.10 \
--pos-weight-split train --device cuda --output-dir $OUT"

JOBS=()
for arch in mamba spiking_expand2; do
  for a in 0.1 1.0; do
    for s in 0 1 2; do
      JOBS+=("$arch|run_dirichlet|$a|$s")
      JOBS+=("$arch|dirichlet|$a|$s")
    done
  done
done

run_chain() {
  local gpu=$1
  local idx=$1
  while [ "$idx" -lt "${#JOBS[@]}" ]; do
    IFS='|' read -r arch mode a s <<< "${JOBS[$idx]}"
    echo "[gpu$gpu] START $arch $mode a=$a s=$s prec=off $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" TORCHDYNAMO_DISABLE=1 \
      .venv/bin/python experiments/run_v7_fl_arch_sweep.py \
      --arch "$arch" --partition-mode "$mode" --alpha "$a" --seed "$s" \
      --mixed-precision off $COMMON \
      >> "$OUT/gpu${gpu}.log" 2>&1
    echo "[gpu$gpu] DONE  $arch $mode a=$a s=$s $(date +%H:%M:%S)" >> "$OUT/gpu${gpu}.log"
    idx=$((idx + 4))
  done
}

for gpu in 0 1 2 3; do run_chain "$gpu" & done
wait
echo "ALL_MULTIARCH_DONE $(date +%H:%M:%S)" | tee -a "$OUT/grid.log"
