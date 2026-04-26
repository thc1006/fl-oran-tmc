"""Spec-driven launcher for v7 FL sweeps (Phase 2 / 3a / 3b).

Reads ``experiments/specs/<spec>.yaml`` via :mod:`scripts._v7_spec_loader`,
applies ``arch_overrides`` per ADR D-20 (e.g. ``lr=1e-4`` and
``lr_warmup_rounds=5`` for ``spiking_expand2``), then runs each expanded
cell sequentially through :func:`fl_oran.training.fl_v7.run_v7_sweep`.

Why this exists vs. ``run_v7_fl_arch_sweep_matrix.py``: the matrix driver
takes a single ``--lr`` and ``--lr-warmup-rounds`` flag for the whole
matrix and therefore cannot honor per-arch overrides — running
``spiking_expand2`` at the dense-arch lr=5e-4 violates ADR D-20 and the
surrogate gradient typically diverges. This launcher is the canonical
entrypoint for spec'd sweeps; the matrix driver remains useful for
single-arch ad-hoc cells.

Example::

    python experiments/run_v7_phase_sweep.py \\
        --spec experiments/specs/phase2_min.yaml \\
        --output-dir artifacts/v7_phase2_min \\
        --summary-tag phase2_min \\
        --continue-on-cell-failure

A combined ``_phase_summary_<tag>_<ts>.csv`` is written incrementally
(after each cell) under ``--output-dir`` so a crash mid-sweep still
leaves usable rows.
"""
from __future__ import annotations

# Performance env vars (must be set BEFORE torch import). Mirrors the
# matrix driver's setup per ADR D-22 perf checklist.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import csv
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import torch

from fl_oran.logging_utils import get_logger
from fl_oran.training.fl_v7 import V7Config, run_v7_sweep

log = get_logger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC_LOADER_PATH = _REPO_ROOT / "scripts" / "_v7_spec_loader.py"


def _load_spec_loader():
    """Load ``scripts/_v7_spec_loader`` without requiring it on sys.path."""
    spec = importlib.util.spec_from_file_location(
        "_v7_spec_loader", _SPEC_LOADER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {_SPEC_LOADER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_v7_spec_loader", mod)
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Spec-driven launcher for v7 FL sweeps. Honors per-arch "
            "hyperparameter overrides (lr, warmup) which the matrix "
            "driver cannot."
        ),
    )
    p.add_argument(
        "--spec", required=True, type=Path,
        help="Path to YAML sweep spec, e.g. experiments/specs/phase2_min.yaml.",
    )
    p.add_argument(
        "--output-dir", default="artifacts/v7_phase_sweep",
        help="Per-cell artifact root (<output_dir>/<cell_name>/).",
    )
    p.add_argument(
        "--unified-parquet", default="data/coloran_raw_unified.parquet",
        help="Override input parquet path.",
    )
    p.add_argument(
        "--continue-on-cell-failure", action="store_true",
        help="Log + continue if a cell crashes; default aborts.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List expanded cells without running training and exit.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Run only the first N expanded cells (0 = all).",
    )
    p.add_argument(
        "--summary-tag", default="",
        help="Optional tag appended to summary CSV filename.",
    )
    return p.parse_args()


def _build_cfg(cell: dict, output_dir: Path, unified_parquet: Path) -> V7Config:
    """Translate one expanded cell dict into a V7Config.

    ``expand_spec`` sets ``alpha=None`` for IID cells; V7Config's dataclass
    annotates ``alpha: float = 0.5`` and downstream code does
    ``f"{cfg.alpha:.2f}"`` in non-IID branches. We pop None and let the
    default (0.5) flow through — IID's name format and partition logic
    don't reference alpha, so the value is inert.
    """
    kw = dict(cell)
    if kw.get("alpha") is None:
        kw.pop("alpha", None)
    return V7Config(
        output_dir=output_dir,
        unified_parquet=unified_parquet,
        **kw,
    )


def main() -> None:
    args = _parse_args()
    loader_mod = _load_spec_loader()
    spec = loader_mod.load_spec(args.spec)
    cells = loader_mod.expand_spec(spec)

    # Cell-name uniqueness is a load-bearing invariant — duplicates would
    # overwrite each other's artifacts/ directory silently. _v7_spec_loader
    # validates partition uniqueness but not the post-expand cross-product
    # explicitly, so cross-check here.
    names = [c["name"] for c in cells]
    if len(set(names)) != len(names):
        from collections import Counter
        dups = [n for n, k in Counter(names).items() if k > 1]
        raise SystemExit(f"expanded cells have duplicate names: {dups}")

    if args.limit and args.limit > 0:
        cells = cells[: args.limit]

    log.info(
        "spec=%s description=%r n_cells=%d (limit=%d)",
        args.spec, spec.get("description", ""), len(cells), args.limit,
    )

    if args.dry_run:
        for i, c in enumerate(cells):
            log.info(
                "cell %3d/%d: name=%s arch=%s algo=%s partition=%s "
                "alpha=%s n_clients=%s seed=%s lr=%s warmup_rounds=%s",
                i + 1, len(cells), c["name"], c["arch"], c["algorithm"],
                c["partition_mode"], c.get("alpha"), c["n_clients"], c["seed"],
                c.get("lr"), c.get("lr_warmup_rounds"),
            )
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parq = Path(args.unified_parquet)
    if not parq.is_file():
        raise SystemExit(f"unified parquet not found: {parq}")

    if torch.cuda.is_available():
        log.info(
            "cuda device: %s (%.1f GiB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        )
    else:
        log.warning("cuda NOT available; cells with device=cuda will fail")

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.summary_tag}" if args.summary_tag else ""
    summary_path = out_dir / f"_phase_summary{tag}_{ts}.csv"
    latest_link = out_dir / f"_phase_summary{tag}_latest.csv"

    summary_rows: list[dict] = []
    n_failed = 0
    t0 = time.time()

    for i, cell in enumerate(cells):
        cfg = _build_cfg(cell, out_dir, parq)
        log.info(
            "--- cell %d/%d: %s (lr=%g warmup=%d) ---",
            i + 1, len(cells), cfg.name, cfg.lr, cfg.lr_warmup_rounds,
        )
        t_cell = time.time()
        row: dict = {
            "name": cfg.name,
            "arch": cfg.arch,
            "algorithm": cfg.algorithm,
            "partition_mode": cfg.partition_mode,
            "alpha": cfg.alpha if cfg.partition_mode == "dirichlet" else "",
            "n_clients": cfg.n_clients,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "lr_warmup_rounds": cfg.lr_warmup_rounds,
        }
        try:
            result = run_v7_sweep(cfg)
            dt = time.time() - t_cell
            test_m = result.get("test", {})
            row.update({
                "test_auc": test_m.get("auc", float("nan")),
                "test_acc": test_m.get("accuracy", float("nan")),
                "test_f1": test_m.get("f1", float("nan")),
                "best_val_auc": result.get("best_val_auc", float("nan")),
                "duration_s": round(dt, 2),
                "status": "ok",
            })
            summary_rows.append(row)
        except Exception as exc:
            n_failed += 1
            dt = time.time() - t_cell
            log.error(
                "cell FAILED %s: %s\n%s", cfg.name, exc, traceback.format_exc(),
            )
            row.update({
                "test_auc": float("nan"),
                "test_acc": float("nan"),
                "test_f1": float("nan"),
                "best_val_auc": float("nan"),
                "duration_s": round(dt, 2),
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
            summary_rows.append(row)
            if not args.continue_on_cell_failure:
                log.error(
                    "aborting sweep — use --continue-on-cell-failure to skip "
                    "and continue to remaining cells.",
                )
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental write so a crash halfway leaves usable data.
        with summary_path.open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    if summary_rows:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(summary_path.name)
        except OSError:
            latest_link.write_text(summary_path.read_text())

    total = time.time() - t0
    log.info(
        "done: %d/%d cells succeeded in %.1fs (%.1f min); summary=%s",
        len(cells) - n_failed, len(cells), total, total / 60, summary_path,
    )
    if n_failed > 0 and not args.continue_on_cell_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
