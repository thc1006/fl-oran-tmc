"""Tests for ``experiments/run_v7_phase_sweep.py`` pre-flight + resume
(Phase 1.5g-3).

Two helpers cover the two new safety layers:

A. ``_preflight(cells, out_dir, parq)`` constructs every cell's
   V7Config and dry-instantiates the algorithm class with the same
   auto-filled-kwargs overlay used by ``fl_v7._run_training_v7``. Any
   missing/unknown kwarg surfaces in ~1 second of zero-GPU work
   instead of after each cell's CPU prep wastes 30+ seconds.

B. ``_load_skip_set(out_dir, tag)`` reads the most recent matching
   ``_phase_summary_*.csv`` and returns the set of cell names whose
   status was ``ok``. Cells with status starting with ``failed`` are
   intentionally NOT in the set — those should be re-run. Together
   with ``--skip-completed`` this is the resume mechanism that makes
   retrying a partially-failed sweep idempotent.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LAUNCHER_PATH = _REPO_ROOT / "experiments" / "run_v7_phase_sweep.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location(
        "run_v7_phase_sweep_test_import", _LAUNCHER_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_v7_phase_sweep_test_import", mod)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _ok_lstm_fedavg_cell() -> dict:
    """One cell that should pass pre-flight cleanly."""
    return {
        "name": "v7_lstm_fedavg_iid_n7_s42",
        "arch": "lstm",
        "algorithm": "fedavg",
        "algo_kwargs": {},
        "partition_mode": "iid",
        "alpha": None,
        "n_clients": 7,
        "seed": 42,
        "num_rounds": 1,
        "clients_per_round": 1,
        "max_steps_per_round": 1,
        "batch_size": 16,
    }


class TestPreflight:
    def test_passes_for_valid_lstm_fedavg(self, launcher, tmp_path):
        # parquet doesn't need to exist for pre-flight (we don't read data).
        # Use a placeholder that satisfies _build_cfg's signature.
        launcher._preflight(
            [_ok_lstm_fedavg_cell()],
            tmp_path / "out",
            tmp_path / "fake.parquet",
        )

    def test_fails_for_fedprox_without_mu(self, launcher, tmp_path):
        bad = _ok_lstm_fedavg_cell()
        bad["algorithm"] = "fedprox"
        bad["algo_kwargs"] = {}  # missing 'mu' — exact Phase 2 bug
        bad["name"] = "v7_lstm_fedprox_iid_n7_s42"
        with pytest.raises(SystemExit, match=r"pre-flight FAILED.*fedprox"):
            launcher._preflight([bad], tmp_path / "out", tmp_path / "fake")

    def test_fails_for_unknown_algorithm(self, launcher, tmp_path):
        bad = _ok_lstm_fedavg_cell()
        bad["algorithm"] = "fedfoo"
        bad["name"] = "v7_lstm_fedfoo_iid_n7_s42"
        with pytest.raises(SystemExit, match=r"pre-flight FAILED"):
            launcher._preflight([bad], tmp_path / "out", tmp_path / "fake")

    def test_first_failure_aborts(self, launcher, tmp_path):
        """Pre-flight should fail FAST on the first bad cell, not enumerate
        all 36 to give one mega-error. The location in the error message
        helps users find the offending cell.
        """
        cells = [
            _ok_lstm_fedavg_cell(),
            {**_ok_lstm_fedavg_cell(),
             "algorithm": "fedprox", "algo_kwargs": {},
             "name": "v7_lstm_fedprox_iid_n7_s42"},
            _ok_lstm_fedavg_cell(),
        ]
        with pytest.raises(SystemExit, match=r"cell 2/3"):
            launcher._preflight(cells, tmp_path / "out", tmp_path / "fake")


# ---------------------------------------------------------------------------
# --skip-completed
# ---------------------------------------------------------------------------


def _write_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["name", "status"]
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class TestLoadSkipSet:
    def test_no_summary_returns_empty_set(self, launcher, tmp_path):
        assert launcher._load_skip_set(tmp_path, "any_tag") == set()

    def test_extracts_only_ok_status(self, launcher, tmp_path):
        rows = [
            {"name": "cell_a", "status": "ok"},
            {"name": "cell_b", "status": "failed: TypeError: ..."},
            {"name": "cell_c", "status": "ok"},
        ]
        _write_summary(tmp_path / "_phase_summary_x_20260427.csv", rows)
        assert launcher._load_skip_set(tmp_path, "x") == {"cell_a", "cell_c"}

    def test_picks_most_recent_summary_when_multiple(
        self, launcher, tmp_path,
    ):
        """If a sweep was retried multiple times, multiple summary CSVs
        accumulate. The newest is authoritative — older ones might
        reflect outdated failures.
        """
        import time
        old = tmp_path / "_phase_summary_x_20260101_000000.csv"
        new = tmp_path / "_phase_summary_x_20260427_010000.csv"
        _write_summary(old, [{"name": "cell_old", "status": "ok"}])
        # Force mtime ordering even if writes are too fast for filesystem
        # timestamps to differentiate.
        import os
        os.utime(old, (0, 1_700_000_000))
        _write_summary(new, [{"name": "cell_new", "status": "ok"}])
        os.utime(new, (0, 1_750_000_000))
        assert launcher._load_skip_set(tmp_path, "x") == {"cell_new"}

    def test_summary_tag_filters_correctly(self, launcher, tmp_path):
        """Summaries from a different phase shouldn't leak into this
        phase's skip set.
        """
        _write_summary(
            tmp_path / "_phase_summary_phase2_min_20260427.csv",
            [{"name": "p2_cell", "status": "ok"}],
        )
        _write_summary(
            tmp_path / "_phase_summary_phase3a_20260427.csv",
            [{"name": "p3_cell", "status": "ok"}],
        )
        assert launcher._load_skip_set(tmp_path, "phase2_min") == {"p2_cell"}
        assert launcher._load_skip_set(tmp_path, "phase3a") == {"p3_cell"}

    def test_empty_tag_matches_any_phase_summary(self, launcher, tmp_path):
        _write_summary(
            tmp_path / "_phase_summary_20260427_010000.csv",
            [{"name": "untagged_cell", "status": "ok"}],
        )
        assert launcher._load_skip_set(tmp_path, "") == {"untagged_cell"}

    def test_symlink_files_ignored(self, launcher, tmp_path):
        """The latest-symlink (``_phase_summary_<tag>_latest.csv``)
        points at the same file already discovered by glob — counting
        it would double-process. Only real files matter.
        """
        real = tmp_path / "_phase_summary_x_20260427.csv"
        _write_summary(real, [{"name": "cell_real", "status": "ok"}])
        link = tmp_path / "_phase_summary_x_latest.csv"
        link.symlink_to(real.name)
        # If symlinks were counted, both would resolve to the same file —
        # set semantics already dedup, but the test asserts the helper
        # explicitly skips symlinks to avoid stat() races.
        assert launcher._load_skip_set(tmp_path, "x") == {"cell_real"}
