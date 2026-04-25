"""Recompute energy.json for every v6 cell using the corrected energy_metrics.

Earlier S1 cells were saved with energy.json values from the original
fvcore-only flops counter. The post-spike out_proj fix in
energy_metrics.py changes the dense MAC count for SpikingForecaster
cells; LSTM and Mamba cells are unaffected by the fix but are
re-recorded here for consistency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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

log = get_logger(__name__)

ARCH_CTOR = {
    "lstm": ForecasterV2,
    "mamba": MambaForecaster,
    "spiking": SpikingForecaster,
}


def parse_cell_dir(name: str) -> tuple[str, int, str]:
    """Returns (arch_base, seed, suffix) where suffix may be empty."""
    arch, _, rest = name.partition("_s")
    seed_part, _, suffix = rest.partition("_")
    return arch, int(seed_part), suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=str, default="artifacts/v6_arch_sweep")
    parser.add_argument("--energy-batch", type=int, default=64)
    parser.add_argument("--unified-parquet", type=str, default="data/coloran_raw_unified.parquet")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    cells = sorted(sweep_dir.glob("*_s*"))
    log.info("recomputing energy for %d cells under %s", len(cells), sweep_dir)

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

    for cell_dir in cells:
        arch_base, seed, suffix = parse_cell_dir(cell_dir.name)
        if arch_base not in ARCH_CTOR:
            log.warning("skip unknown arch %s in %s", arch_base, cell_dir)
            continue
        ctor = ARCH_CTOR[arch_base]
        kwargs: dict = {}
        # Recover spiking T_inner from the cell suffix label.
        if arch_base == "spiking" and "t5" in suffix:
            kwargs["t_inner"] = 5
        model = ctor(schema=schema, task="classification", seq_len=cfg.seq_len, **kwargs)

        best_state_path = cell_dir / "best_state.pt"
        if best_state_path.exists():
            model.load_state_dict(torch.load(best_state_path, map_location="cpu", weights_only=True))
        else:
            log.warning("no best_state.pt in %s — using random init", cell_dir)

        energy = estimate_energy_pJ_per_inference(model, te_cat, te_cont)
        out = {k: float(v) for k, v in energy.items()}
        (cell_dir / "energy.json").write_text(json.dumps(out, indent=2))
        log.info("[%s s=%d %s] flops=%.0f sops=%.0f energy_pJ=%.2e",
                 arch_base, seed, suffix or '-', out['flops'], out['sops'], out['total_energy_pJ'])


if __name__ == "__main__":
    main()
