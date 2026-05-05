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


@pytest.mark.xfail(strict=True, reason="P1.2-GREEN: implement experiments/run_p1_tr_embedding_check.py")
def test_p1_2_experiment_script_exists() -> None:
    """RED: experiments/run_p1_tr_embedding_check.py must exist.
    xfail strict=True → when GREEN lands, remove this marker."""
    assert EXPERIMENT_PATH.exists(), (
        f"{EXPERIMENT_PATH} not found. Implement P1.2-GREEN: 12-cell sweep "
        f"(LSTM × {{natural-by-BS, Dirichlet α=0.05}} × {{normal, frozen test_tr}} "
        f"× 3 seeds). Compares natural-by-BS vs Dirichlet AUC gap with and "
        f"without the tr embedding fix."
    )


def test_freeze_test_tr_helper_returns_modified_embedding() -> None:
    """RED: freeze_test_tr_rows(embedding, train_tr_indices) must zero or
    re-init the rows for unseen test tr values."""
    if not EXPERIMENT_PATH.exists():
        pytest.skip("script not yet implemented (P1.2-GREEN)")
    mod = _import_from(EXPERIMENT_PATH, "run_p1_tr_embedding_check")
    assert hasattr(mod, "freeze_test_tr_rows"), (
        "must export freeze_test_tr_rows(embedding_weight, train_tr_indices)"
    )
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((30, 8)).astype(np.float32)
    train_tr = list(range(22))  # {0..21}
    fixed = mod.freeze_test_tr_rows(weight.copy(), train_tr)
    # Train rows preserved
    np.testing.assert_array_equal(fixed[0:22], weight[0:22])
    # Test/val/unused rows zeroed (or otherwise normalized)
    assert (fixed[22:30] == 0).all() or np.allclose(fixed[22:30], fixed[22:30].mean()), (
        "freeze_test_tr_rows must remove the per-row variation in untrained rows"
    )


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
