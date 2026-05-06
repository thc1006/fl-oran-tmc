"""P2.2 inference latency + communication bytes (reviewer MC6).

Reviewer MC6: paper claims O-RAN deployment relevance but measures only
training AUC and training energy. Essential deployment measurements:
  (a) inference latency per-prediction — compare to near-RT RIC 10ms-1s
      control loop budget
  (b) model state_dict size in bytes — sets per-round FL communication
      cost (state_dict × N_clients_per_round)

This script measures both for the 3 backbones (LSTM, Mamba, Spiking)
using a Phase 5 trained checkpoint per arch. CPU + GPU latency reported
for single-sample (1×5×21) and batched (256×5×21) inputs to bracket
edge-device vs accelerator deployment scenarios.

Output: artifacts/p2_inference/results.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

ARCH_REGISTRY = {
    "lstm": ("fl_oran.models.forecaster_v2", "ForecasterV2", {}),
    "mamba": ("fl_oran.models.mamba_forecaster", "MambaForecaster", {}),
    "spiking_expand2": (
        "fl_oran.models.spiking_forecaster", "SpikingForecaster",
        {"backbone_d_model": 56, "backbone_expand": 2},
    ),
}


def _load_model(arch: str, ckpt_path: Path):
    """Build model + load checkpoint state_dict (strip _orig_mod. prefix)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import importlib
    from fl_oran.utils.seed import seed_everything
    from fl_oran.data_v2.encoders import FeatureSchema
    from fl_oran.training.centralized_v3 import (
        V3_CATEGORICAL,
        V3_CAT_SIZES,
        V3_CONTINUOUS,
    )

    if arch not in ARCH_REGISTRY:
        raise ValueError(f"unknown arch={arch!r}")
    module_path, cls_name, extra_kwargs = ARCH_REGISTRY[arch]
    cls = getattr(importlib.import_module(module_path), cls_name)
    schema = FeatureSchema(
        categorical=V3_CATEGORICAL,
        categorical_sizes=V3_CAT_SIZES,
        continuous=V3_CONTINUOUS,
    )
    seed_everything(0, deterministic=True)
    model = cls(schema=schema, task="classification", seq_len=5, **extra_kwargs)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model, schema


def _benchmark_latency(
    model, schema, device, batch_sizes, n_warmup=20, n_iter=200
):
    """Time `model(cat, cont)` forward pass per batch size."""
    n_cat = schema.n_categorical
    n_cont = schema.n_continuous
    seq_len = 5
    results = {}
    for bs in batch_sizes:
        cat = torch.zeros(bs, seq_len, n_cat, dtype=torch.long, device=device)
        cont = torch.zeros(bs, seq_len, n_cont, dtype=torch.float32, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(cat, cont)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model(cat, cont)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / n_iter) * 1000
        per_sample_ms = per_call_ms / bs
        results[bs] = {
            "latency_per_batch_ms": per_call_ms,
            "latency_per_sample_ms": per_sample_ms,
            "throughput_samples_per_s": bs / (elapsed / n_iter),
            "n_iter": n_iter,
        }
    return results


def _state_dict_bytes(model) -> tuple[int, int]:
    """Return (total_bytes, total_params) for the model's trainable state."""
    total_bytes = 0
    total_params = 0
    for p in model.state_dict().values():
        if hasattr(p, "numel") and hasattr(p, "element_size"):
            total_params += p.numel()
            total_bytes += p.numel() * p.element_size()
    return total_bytes, total_params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-root", type=Path,
        default=REPO_ROOT / "artifacts" / "v7_stage2_full",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Which Phase 5 seed's checkpoint to load (default 42)")
    ap.add_argument("--archs", type=str, nargs="+",
                    default=["lstm", "mamba", "spiking_expand2"])
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 8, 64, 256])
    ap.add_argument("--n-clients-per-round", type=int, default=5,
                    help="Match Phase 5 clients_per_round (5 of 7) for comm-bytes calc")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "p2_inference" / "results.json",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; CUDA available: {torch.cuda.is_available()}")
    print(f"Archs: {args.archs}; batch sizes: {args.batch_sizes}")

    per_arch = {}
    for arch in args.archs:
        ckpt = args.ckpt_root / f"v7_{arch}_fedavg_iid_n7_s{args.seed}" / "best.pt"
        if not ckpt.exists():
            print(f"  SKIP {arch}: missing {ckpt}")
            continue
        print(f"\n=== {arch} (loaded {ckpt.name}) ===")
        model, schema = _load_model(arch, ckpt)
        # Communication cost
        bytes_per_round_per_client, params = _state_dict_bytes(model)
        bytes_per_round_total = bytes_per_round_per_client * args.n_clients_per_round
        print(f"  state_dict: {params:,} params, {bytes_per_round_per_client:,} bytes/client/round "
              f"({bytes_per_round_per_client/1024:.1f} KiB)")
        print(f"  total round bytes (5 clients × upload): {bytes_per_round_total:,} "
              f"({bytes_per_round_total/1024:.1f} KiB)")

        # Latency on GPU
        gpu_lat = {}
        if device.type == "cuda":
            model_gpu = model.to(device)
            gpu_lat = _benchmark_latency(model_gpu, schema, device, args.batch_sizes)
            for bs, r in gpu_lat.items():
                print(f"  GPU bs={bs:<4}  {r['latency_per_batch_ms']:>8.3f} ms/batch  "
                      f"{r['latency_per_sample_ms']:>8.4f} ms/sample  "
                      f"{r['throughput_samples_per_s']:>10,.0f} samples/s")
            model = model.to("cpu")  # move back for CPU benchmark

        # Latency on CPU (deployment-edge realistic for some scenarios)
        cpu_dev = torch.device("cpu")
        cpu_lat = _benchmark_latency(model, schema, cpu_dev,
                                      args.batch_sizes, n_warmup=5, n_iter=50)
        for bs, r in cpu_lat.items():
            print(f"  CPU bs={bs:<4}  {r['latency_per_batch_ms']:>8.3f} ms/batch  "
                  f"{r['latency_per_sample_ms']:>8.4f} ms/sample  "
                  f"{r['throughput_samples_per_s']:>10,.0f} samples/s")

        per_arch[arch] = {
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
            "n_params": params,
            "bytes_per_client_per_round": bytes_per_round_per_client,
            "bytes_per_round_total_5clients": bytes_per_round_total,
            "kib_per_client_per_round": bytes_per_round_per_client / 1024,
            "gpu_latency": gpu_lat,
            "cpu_latency": cpu_lat,
        }

    if not per_arch:
        print("ERROR: no checkpoints loaded", file=sys.stderr)
        return 1

    # Near-RT RIC budget headroom analysis
    print("\n=== Near-RT RIC 10ms-1s control-loop budget headroom (single-sample) ===")
    near_rt_budget_ms = 10  # tightest near-RT RIC budget
    for arch, m in per_arch.items():
        gpu_1ms = m["gpu_latency"].get(1, {}).get("latency_per_sample_ms", float("nan"))
        cpu_1ms = m["cpu_latency"].get(1, {}).get("latency_per_sample_ms", float("nan"))
        print(f"  {arch:18s}  GPU {gpu_1ms:.3f} ms ({near_rt_budget_ms/gpu_1ms if gpu_1ms else 0:.1f}× headroom)  "
              f"CPU {cpu_1ms:.3f} ms ({near_rt_budget_ms/cpu_1ms if cpu_1ms else 0:.1f}× headroom)")

    payload = {
        "description": f"P2.2 inference latency + comm bytes (arch checkpoints from seed={args.seed})",
        "device_gpu_avail": torch.cuda.is_available(),
        "device_gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_clients_per_round": args.n_clients_per_round,
        "batch_sizes_tested": args.batch_sizes,
        "near_rt_ric_budget_ms": near_rt_budget_ms,
        "per_arch": per_arch,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
