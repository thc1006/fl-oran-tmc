"""Recompute energy.json for every v6 cell using the corrected energy_metrics.

Earlier S1 cells were saved with energy.json values from the original
fvcore-only flops counter. The post-spike out_proj fix in
energy_metrics.py changes the dense MAC count for SpikingForecaster
cells; LSTM and Mamba cells are unaffected by the fix but are
re-recorded here for consistency.

**Design notes (Round 4 fixes):**

* Uses ``scripts/_v6_cell_metadata.py`` for arch registry, cell parsing
  and kwargs reconstruction. This is the single source of truth shared
  with ``measure_v6_gpu_energy.py``; keeping the logic in one place
  prevents the two scripts from drifting apart (they had drifted —
  the previous version of this script silently corrupted t5sum /
  ``_lif_*`` / ``_expand2`` cells by passing wrong kwargs).
* Per-cell try/except so one misconfigured cell does not abort the
  whole sweep.
* Atomic ``energy.json`` writes via ``atomic_write_text`` so a crash
  cannot leave a half-written file that breaks the aggregator.
* ``--force`` flag controls overwriting cells whose ``energy.json``
  already contains the post-fix three-accounting fields. Without
  ``--force`` we skip cells whose energy is already current.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from fl_oran.data_v2.encoders import apply_continuous_scaler, fit_continuous_scaler
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.split import ood_split_by_tr
from fl_oran.evaluation.energy_metrics import estimate_energy_pJ_per_inference
from fl_oran.logging_utils import get_logger
from fl_oran.models.forecaster_v2 import ForecasterV2
from fl_oran.models.mamba_forecaster import MambaForecaster
from fl_oran.models.spiking_forecaster import SpikingForecaster
from fl_oran.training.centralized_v3 import V3Config, _load_and_prepare

# Local helper module (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v6_cell_metadata import (  # noqa: E402
    atomic_write_text,
    build_kwargs_from_suffix,
    parse_cell_dir,
)

log = get_logger(__name__)

# Map arch_base → constructor. Mirrors the runner's ARCH_REGISTRY but
# does not have to carry kwargs (those come from the runner via
# build_kwargs_from_suffix).
ARCH_CTOR = {
    "lstm": ForecasterV2,
    "mamba": MambaForecaster,
    "mamba_expand2": MambaForecaster,
    "spiking": SpikingForecaster,
    "spiking_expand2": SpikingForecaster,
}


# Fields that ``estimate_energy_pJ_per_inference`` writes once the
# three-accounting fix is in place. If all of these exist in an
# existing ``energy.json``, the cell is already at the post-fix schema
# and there is no reason to overwrite (saves time and avoids any small
# numerical jitter from re-running fvcore on a re-loaded model).
_POST_FIX_KEYS = (
    "total_energy_pJ_gpu_dense",
    "total_energy_pJ_sparsity_aware",
    "total_energy_pJ_neuromorphic",
)


def _is_already_current(energy_json_path: Path) -> bool:
    if not energy_json_path.exists():
        return False
    try:
        d = json.loads(energy_json_path.read_text())
    except json.JSONDecodeError:
        return False
    return all(k in d for k in _POST_FIX_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="artifacts/v6_arch_sweep")
    parser.add_argument("--energy-batch", type=int, default=64)
    parser.add_argument("--unified-parquet", type=str,
                        default="data/coloran_raw_unified.parquet")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if energy.json already has the "
                             "post-fix three-accounting schema")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cells = sorted(d for d in sweep_dir.glob("*_s*") if d.is_dir())
    log.info("recomputing energy for up to %d cells under %s "
             "(force=%s)", len(cells), sweep_dir, args.force)

    cfg = V3Config(
        unified_parquet=Path(args.unified_parquet),
        sample_ratio=1.0,
        seq_len=5,
        threshold=0.10,
    )
    df, schema = _load_and_prepare(cfg)
    feat_cols = schema.categorical + schema.continuous
    split = ood_split_by_tr(df, cfg.train_tr, cfg.val_tr, cfg.test_tr)
    X_tr, _ = build_run_sequences(split.train, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    X_te, _ = build_run_sequences(split.test, feat_cols, ["y_sla_next"], seq_len=cfg.seq_len)
    scaler = fit_continuous_scaler({0: X_tr}, schema)
    cat, cont = apply_continuous_scaler(X_te, schema, scaler)
    te_cat = torch.from_numpy(cat[: args.energy_batch])
    te_cont = torch.from_numpy(cont[: args.energy_batch])
    log.info("energy slice: %d rows × seq_len=%d", te_cat.shape[0], te_cat.shape[1])

    n_skipped_unknown = 0
    n_skipped_no_state = 0
    n_skipped_already_current = 0
    n_failed = 0
    n_recomputed = 0

    for cell_dir in cells:
        try:
            arch_base, seed, suffix = parse_cell_dir(cell_dir.name)
        except ValueError as exc:
            log.warning("skip un-parseable cell %s (%s)", cell_dir.name, exc)
            n_skipped_unknown += 1
            continue
        if arch_base not in ARCH_CTOR:
            log.warning("skip unknown arch %s in %s (extend ARCH_CTOR if "
                        "this is a new ablation)", arch_base, cell_dir)
            n_skipped_unknown += 1
            continue

        energy_json_path = cell_dir / "energy.json"
        if not args.force and _is_already_current(energy_json_path):
            log.info("[%s s=%d %s] energy.json already at post-fix schema; "
                     "skipping (use --force to override)",
                     arch_base, seed, suffix or "-")
            n_skipped_already_current += 1
            continue

        # Refuse to compute energy on a randomly-initialised model: the
        # spike rate (and therefore SOPs) depends on input × weights, so
        # a random model gives a meaningless energy.json that would
        # silently corrupt the aggregated table.
        best_state_path = cell_dir / "best_state.pt"
        if not best_state_path.exists():
            log.warning("[%s s=%d %s] no best_state.pt — SKIPPING (refusing "
                        "to write energy.json from random init)",
                        arch_base, seed, suffix or "-")
            n_skipped_no_state += 1
            continue

        try:
            kwargs = build_kwargs_from_suffix(arch_base, suffix)
            ctor = ARCH_CTOR[arch_base]
            model = ctor(schema=schema, task="classification",
                         seq_len=cfg.seq_len, **kwargs)
            model.load_state_dict(
                torch.load(best_state_path, map_location="cpu", weights_only=True)
            )
            energy = estimate_energy_pJ_per_inference(model, te_cat, te_cont)
            out = {k: float(v) for k, v in energy.items()}
            atomic_write_text(energy_json_path, json.dumps(out, indent=2))
            log.info("[%s s=%d %s] flops=%.0f sops=%.0f energy_pJ=%.2e",
                     arch_base, seed, suffix or "-",
                     out["flops"], out["sops"], out["total_energy_pJ"])
            n_recomputed += 1
        except Exception as exc:
            log.error("[%s s=%d %s] recompute failed: %s\n%s",
                      arch_base, seed, suffix or "-", exc,
                      traceback.format_exc())
            n_failed += 1
            continue

    log.info(
        "recompute summary: recomputed=%d skipped_already_current=%d "
        "skipped_no_state=%d skipped_unknown_arch=%d failed=%d",
        n_recomputed, n_skipped_already_current, n_skipped_no_state,
        n_skipped_unknown, n_failed,
    )
    if n_failed:
        # Don't return success if any cell failed — the orchestrator's
        # set -e relies on us signalling problems explicitly.
        sys.exit(1)


if __name__ == "__main__":
    main()
