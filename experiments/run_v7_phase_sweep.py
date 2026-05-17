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
from fl_oran.training.fl_v7 import V7Config, _select_algorithm, run_v7_sweep

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
    p.add_argument(
        "--skip-completed", action="store_true",
        help=(
            "Read the latest _phase_summary_*.csv in --output-dir and "
            "skip any cell whose status was 'ok'. Cells with status "
            "starting with 'failed' are RE-RUN. Useful for retrying a "
            "partially-failed sweep without redoing the successful cells."
        ),
    )
    p.add_argument(
        "--no-preflight", action="store_true",
        help=(
            "Skip the algorithm-instantiation pre-flight check. Default is "
            "to dry-instantiate every cell's V7Config + algorithm class "
            "before any training, to catch missing kwargs / config bugs "
            "in seconds rather than after each cell's CPU prep."
        ),
    )
    p.add_argument(
        "--shard", default="",
        help=(
            "Shard filter N/M (1-based): run only cells whose post-expand "
            "index satisfies (i mod M) == N - 1. Used by V100 multi-chain "
            "launchers to split the master spec across GPUs without writing "
            "per-chain spec files. Stable across runs since expand_spec is "
            "deterministic. Applied AFTER --limit and BEFORE --skip-completed."
        ),
    )
    p.add_argument(
        "--skip-existing-summary", action="store_true",
        help=(
            "Filesystem-based skip: drop any cell whose output dir already "
            "contains a non-empty summary.json. Stronger than --skip-completed "
            "for sweeps that were partially run via run_v7_fl_arch_sweep.py "
            "(no _phase_summary CSV). Path D launcher requires this to "
            "preserve the existing 60 LSTM cells from the original 60-cell "
            "sweep that predate the spec-driven launcher."
        ),
    )
    return p.parse_args()


def _load_skip_set(out_dir: Path, summary_tag: str) -> set[str]:
    """Return the set of cell names whose previous run status was 'ok'.

    Reads the most recent ``_phase_summary[_<tag>]_*.csv`` in
    ``out_dir`` (matched by mtime, not symlink, since the symlink can
    point to a stale file). If no summary exists or it has no 'ok'
    rows, returns the empty set.
    """
    pattern = (
        f"_phase_summary_{summary_tag}_*.csv" if summary_tag
        else "_phase_summary_*.csv"
    )
    candidates = sorted(
        (p for p in out_dir.glob(pattern) if not p.is_symlink()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return set()
    src = candidates[0]
    skip: set[str] = set()
    with src.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("status") == "ok":
                skip.add(row["name"])
    if skip:
        log.info(
            "--skip-completed: found %d ok cells in %s",
            len(skip), src.name,
        )
    return skip


def _preflight(cells: list[dict], out_dir: Path, parq: Path) -> None:
    """Construct V7Config + algorithm class for every cell, no training.

    Catches the bug class that wasted 22 GPU-min in Phase 2: the spec
    expanded successfully and 18 cells worth of CPU prep ran before
    every fedprox cell crashed at training time on
    ``FedProx(missing 'mu')``. With this check, identical bugs surface
    in ~1 second of zero-GPU work.

    The check intentionally calls ``_select_algorithm`` AND constructs
    the algorithm instance with the fl_v7-auto-filled kwargs +
    ``cfg.algo_kwargs`` overlay — same code path run_v7_sweep uses, so
    any signature mismatch surfaces here.
    """
    log.info("pre-flight: dry-instantiate %d cells (no training)", len(cells))
    for i, cell in enumerate(cells):
        try:
            cfg = _build_cfg(cell, out_dir, parq)
            algo_cls = _select_algorithm(cfg)
            # Mirror fl_v7._run_training_v7's auto-fill set.
            test_kwargs = {
                "max_steps": cfg.max_steps_per_round,
                "batch_size": cfg.batch_size,
                "grad_clip": cfg.grad_clip,
                "amp_enabled": False,
                "amp_dtype": None,
            }
            test_kwargs.update(cfg.algo_kwargs)
            algo_cls(**test_kwargs)
        except Exception as exc:
            raise SystemExit(
                f"pre-flight FAILED at cell {i+1}/{len(cells)} "
                f"(name={cell.get('name')!r}, arch={cell.get('arch')!r}, "
                f"algo={cell.get('algorithm')!r}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    log.info("pre-flight ok — all %d cells construct cleanly", len(cells))


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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parq = Path(args.unified_parquet)

    # Order matters: limit → shard → skip-completed → dry-run. Limit narrows
    # the universe first; shard splits across chains by stable post-expand
    # index; skip-completed prunes already-finished work; only then does
    # dry-run print what WILL actually execute.
    if args.limit and args.limit > 0:
        cells = cells[: args.limit]

    if args.shard:
        try:
            n_str, m_str = args.shard.split("/")
            n_shard = int(n_str)
            m_shard = int(m_str)
        except ValueError as e:
            raise SystemExit(
                f"--shard must be of form N/M (e.g. 1/4); got {args.shard!r}"
            ) from e
        if not (1 <= n_shard <= m_shard):
            raise SystemExit(
                f"--shard requires 1 <= N <= M; got N={n_shard} M={m_shard}"
            )
        if m_shard < 1:
            raise SystemExit(f"--shard M must be >= 1; got {m_shard}")
        before = len(cells)
        cells = [c for i, c in enumerate(cells) if i % m_shard == n_shard - 1]
        log.info(
            "--shard %d/%d: %d → %d cells",
            n_shard, m_shard, before, len(cells),
        )

    if args.skip_completed:
        skip_names = _load_skip_set(out_dir, args.summary_tag)
        if skip_names:
            before = len(cells)
            cells = [c for c in cells if c["name"] not in skip_names]
            log.info(
                "--skip-completed: %d → %d cells after filter",
                before, len(cells),
            )

    if args.skip_existing_summary:
        before = len(cells)
        cells = [
            c for c in cells
            if not (
                (out_dir / c["name"] / "summary.json").is_file()
                and (out_dir / c["name"] / "summary.json").stat().st_size > 0
            )
        ]
        log.info(
            "--skip-existing-summary: %d → %d cells after filesystem check",
            before, len(cells),
        )

    log.info(
        "spec=%s description=%r n_cells=%d (limit=%d)",
        args.spec, spec.get("description", ""), len(cells), args.limit,
    )

    if not cells:
        log.info("nothing to run after filtering; exiting.")
        return

    if args.dry_run:
        for i, c in enumerate(cells):
            log.info(
                "cell %3d/%d: name=%s arch=%s algo=%s algo_kwargs=%s "
                "partition=%s alpha=%s n_clients=%s seed=%s lr=%s warmup_rounds=%s",
                i + 1, len(cells), c["name"], c["arch"], c["algorithm"],
                c.get("algo_kwargs", {}),
                c["partition_mode"], c.get("alpha"), c["n_clients"], c["seed"],
                c.get("lr"), c.get("lr_warmup_rounds"),
            )
        return

    if not parq.is_file():
        raise SystemExit(f"unified parquet not found: {parq}")

    if not args.no_preflight:
        _preflight(cells, out_dir, parq)

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

        # Phase 1.5n cleanup (2026-04-28 Phase 5 perf fix, defense-in-
        # depth): reset torch.compile / dynamo state and force GC even
        # if the cell crashed (run_v7_sweep's own cleanup at function
        # end is bypassed on exception → next cell would inherit stale
        # state → linear slowdown over 900 cells). Belt + suspenders
        # with the in-function cleanup.
        try:
            import torch._dynamo as _dynamo
            _dynamo.reset()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
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
