#!/usr/bin/env bash
# Short 3-round FL run on 5% data — validates the full pipeline quickly.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

COMMON="--sample-ratio 0.01 --samples-per-client 10000 --num-rounds 3 --local-epochs 1 --clients-per-round 3 --batch-size 256 --num-workers 4 --no-compile"

echo "== v1.0.6 smoke =="
python -m fl_oran --variant v106 --name smoke_v106 $COMMON

echo "== v1.0.7-1 smoke =="
python -m fl_oran --variant v107_1 --name smoke_v107_1 $COMMON --seq-len 5

echo "== v1.0.7-2 smoke =="
python -m fl_oran --variant v107_2 --name smoke_v107_2 $COMMON --dp --dp-clip 1.0 --dp-noise 0.1

echo "smoke tests OK."
