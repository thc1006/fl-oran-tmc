#!/usr/bin/env bash
# R2 C3 V100 launcher — distribute 15 cells (3 archs × 5 seeds; BS-expansion
# happens inside each cell) across 4 V100-SXM2 cards with up to 4 concurrent
# processes per card (oversubscription is OK because each cell uses ~5 GiB
# VRAM and V100 has 32 GiB).
#
# Per artifacts/audit/r2_gpu_design.md C3 section:
#   N_cells = 15 (arch × seed; BS-loop inside each cell)
#   per-cell wall ~3 min on V100 fp16
#   4 cards × 4 concurrent → ~30-40 min wall total
#
# Usage (on V100 head node, after `cd ~/fl-oran-tmc`):
#   bash scripts/v100_r2_c3_launcher.sh
#
# Outputs: artifacts/r2_post_hoc_per_bs_finetune/<arch>_s<seed>.json
#
# Cells are skipped if their output JSON already exists, so this script is
# safely re-runnable after partial failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-artifacts/r2_post_hoc_per_bs_finetune}"
PARQUET="${PARQUET:-/home/leo07010/data/coloran_raw_unified.parquet}"
FINETUNE_STEPS="${FINETUNE_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
# V100 BF16 is software-emulated; FP16 is native and ~2× faster on Volta sm_70.
PRECISION="${PRECISION:-fp16}"
N_CONCURRENT_PER_GPU="${N_CONCURRENT_PER_GPU:-4}"

mkdir -p "$OUT_DIR"

# 3 archs × 5 seeds = 15 cells. Distribute across the 4 V100s round-robin.
ARCHS=(lstm mamba spiking_expand2)
SEEDS=(0 1 2 3 42)

# Pre-build the 15 cell strings.
ALL_CELLS=()
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ALL_CELLS+=("${arch}:s${seed}")
  done
done

echo "Total cells: ${#ALL_CELLS[@]}; will distribute round-robin across 4 V100 cards."

# Group cells by GPU (round-robin: cell i → GPU (i % 4))
declare -a GPU0=() GPU1=() GPU2=() GPU3=()
for i in "${!ALL_CELLS[@]}"; do
  gpu=$(( i % 4 ))
  case "$gpu" in
    0) GPU0+=("${ALL_CELLS[$i]}") ;;
    1) GPU1+=("${ALL_CELLS[$i]}") ;;
    2) GPU2+=("${ALL_CELLS[$i]}") ;;
    3) GPU3+=("${ALL_CELLS[$i]}") ;;
  esac
done

# Launch one background process per GPU, each handling its share of cells
# sequentially.  Within-GPU N_CONCURRENT_PER_GPU is left at 1 here for
# simplicity — true oversubscription would require splitting cell lists
# further.  ~4 cells/GPU × 3 min/cell ≈ 12 min wall worst case.
launch_gpu() {
  local gpu_num="$1"; shift
  local cells_csv
  cells_csv=$(IFS=,; echo "$*")
  echo "  GPU $gpu_num: $cells_csv"
  CUDA_VISIBLE_DEVICES="$gpu_num" \
    python experiments/run_r2_post_hoc_per_bs_finetune.py \
      --parquet "$PARQUET" \
      --cells "$cells_csv" \
      --device "cuda:0" \
      --finetune-steps "$FINETUNE_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --mixed-precision "$PRECISION" \
      --out "$OUT_DIR" \
      > "$OUT_DIR/launcher_gpu${gpu_num}.log" 2>&1 &
}

echo "Launching 4 background processes..."
launch_gpu 0 "${GPU0[@]}"
launch_gpu 1 "${GPU1[@]}"
launch_gpu 2 "${GPU2[@]}"
launch_gpu 3 "${GPU3[@]}"

echo "Waiting for all 4 GPU groups to finish..."
wait
echo "All GPU groups done. Outputs in $OUT_DIR/"
echo "Aggregate via: python scripts/r2_aggregate_c3.py  (see test_r2_post_hoc_finetune.py for expected schema)"
