"""P1.2-RED methodology tests for tr embedding bug-quantification experiment.

These verify the experiment scaffolding works correctly. The actual
hypothesis (whether fixing the bug shrinks natural-by-BS dominance) is
preregistered separately in experiments/preregistered/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_PATH = REPO_ROOT / "experiments" / "run_p1_tr_embedding_check.py"


def _import_from(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_p1_2_experiment_script_exists() -> None:
    """GREEN (xfail removed 2026-05-06): experiments/run_p1_tr_embedding_check.py
    exists per P1.2-GREEN; ran 18 cells (9 IID + 9 Dirichlet seeds) and
    showed bug shrinks gap by 9.2% (residual 0.0491 AUC; H1.2.B passes,
    H1.2.C borderline FAIL by 0.0009 << seed-σ but substantive
    conclusion C1 survives)."""
    assert EXPERIMENT_PATH.exists(), (
        f"{EXPERIMENT_PATH} not found. Implement P1.2-GREEN: 12-cell sweep "
        f"(LSTM × {{natural-by-BS, Dirichlet α=0.05}} × {{normal, frozen test_tr}} "
        f"× 3 seeds). Compares natural-by-BS vs Dirichlet AUC gap with and "
        f"without the tr embedding fix."
    )


def test_freeze_test_tr_helper_returns_modified_embedding() -> None:
    """GREEN: freeze_test_tr_rows(embedding, train_tr_indices, mode='mean')
    replaces untrained rows with the same vector (mean of trained rows),
    eliminating per-row random-init variation."""
    if not EXPERIMENT_PATH.exists():
        pytest.skip("script not yet implemented (P1.2-GREEN)")
    mod = _import_from(EXPERIMENT_PATH, "run_p1_tr_embedding_check")
    assert hasattr(mod, "freeze_test_tr_rows"), (
        "must export freeze_test_tr_rows(embedding_weight, train_tr_indices)"
    )
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((30, 8)).astype(np.float32)
    train_tr = list(range(22))  # {0..21}

    # mode='mean' (default): test rows all equal mean of trained rows
    fixed_mean = mod.freeze_test_tr_rows(weight.copy(), train_tr, mode="mean")
    # Train rows preserved bit-exactly
    import torch
    fm = fixed_mean.numpy() if hasattr(fixed_mean, "numpy") else fixed_mean
    np.testing.assert_array_equal(fm[0:22], weight[0:22])
    # All untrained rows must be identical to row 22 (i.e., share one vector)
    for r in range(22, 30):
        np.testing.assert_allclose(
            fm[r], fm[22], rtol=1e-6,
            err_msg=f"row {r} should equal row 22 in mode='mean'",
        )
    # That vector must equal the mean of the trained rows
    np.testing.assert_allclose(
        fm[22], weight[0:22].mean(axis=0), rtol=1e-6,
        err_msg="mode='mean' must replace with mean of trained rows",
    )

    # mode='zero': test rows are exactly zero
    fixed_zero = mod.freeze_test_tr_rows(weight.copy(), train_tr, mode="zero")
    fz = fixed_zero.numpy() if hasattr(fixed_zero, "numpy") else fixed_zero
    np.testing.assert_array_equal(fz[0:22], weight[0:22])
    np.testing.assert_array_equal(fz[22:30], np.zeros_like(fz[22:30]))


def test_p1_2_results_json_schema() -> None:
    """RED: artifacts/p1_tr_embedding/results.json must contain the
    canonical comparison keys."""
    results_path = REPO_ROOT / "artifacts" / "p1_tr_embedding" / "results.json"
    if not results_path.exists():
        pytest.skip("results not yet computed (P1.2-GREEN)")
    import json
    data = json.loads(results_path.read_text())
    expected_keys = {
        "natural_by_bs_normal_auc_mean", "natural_by_bs_frozen_auc_mean",
        "dirichlet_a005_normal_auc_mean", "dirichlet_a005_frozen_auc_mean",
        "n_seeds", "computed_at",
    }
    missing = expected_keys - data.keys()
    assert not missing, f"results.json missing keys: {missing}"
