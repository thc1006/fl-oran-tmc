"""P1.1-RED methodology tests for naive baselines.

These are CI-gated unit tests of the baseline implementations themselves
(scripts/baseline_last_bler.py + scripts/baseline_logreg.py). They verify
the baseline functions return correctly-shaped, numerically-sane outputs;
they do NOT preregister hypotheses about the baseline AUC values relative
to FL methods (those go into experiments/preregistered/).

Tests fail until P1.1-GREEN implements the baseline scripts.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAST_BLER_PATH = REPO_ROOT / "scripts" / "baseline_last_bler.py"
LOGREG_PATH = REPO_ROOT / "scripts" / "baseline_logreg.py"


def _import_from(path: Path, module_name: str):
    """Import a script as a module from its file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- last-BLER persistence baseline ----------

@pytest.mark.xfail(strict=True, reason="P1.1-GREEN: implement scripts/baseline_last_bler.py")
def test_baseline_last_bler_script_exists() -> None:
    """RED: scripts/baseline_last_bler.py must exist (P1.1-GREEN creates it).
    xfail strict=True → when GREEN lands, this test must pass and xfail
    marker must be removed."""
    assert LAST_BLER_PATH.exists(), (
        f"{LAST_BLER_PATH} not found. Implement P1.1-GREEN: a script that "
        f"loads the test parquet, extracts ul_bler at offset seq_len-1 "
        f"(last seen time step), and predicts y_t+1 = 1[ul_bler_t > 0.10]."
    )


def test_baseline_last_bler_predict_returns_correct_shape() -> None:
    """RED: predict_last_bler(ul_bler_array, threshold) must return
    np.ndarray of shape (N,) and dtype matching the input length."""
    if not LAST_BLER_PATH.exists():
        pytest.skip("script not yet implemented (P1.1-GREEN)")
    mod = _import_from(LAST_BLER_PATH, "baseline_last_bler")
    assert hasattr(mod, "predict_last_bler"), (
        "baseline_last_bler.py must export predict_last_bler(ul_bler, threshold)"
    )
    ul_bler = np.array([0.05, 0.10, 0.15, 0.20, 0.08], dtype=np.float32)
    pred = mod.predict_last_bler(ul_bler, threshold=0.10)
    assert pred.shape == (5,), f"shape {pred.shape}, expected (5,)"
    # threshold-aware: > 0.10 → 1, otherwise 0; 0.10 itself is NOT > 0.10
    expected = np.array([0, 0, 1, 1, 0])
    np.testing.assert_array_equal(pred, expected)


# ---------- logistic regression baseline ----------

@pytest.mark.xfail(strict=True, reason="P1.1-GREEN: implement scripts/baseline_logreg.py")
def test_baseline_logreg_script_exists() -> None:
    """RED: scripts/baseline_logreg.py must exist (P1.1-GREEN creates it).
    xfail strict=True → when GREEN lands, remove this marker."""
    assert LOGREG_PATH.exists(), (
        f"{LOGREG_PATH} not found. Implement P1.1-GREEN: a script that "
        f"trains sklearn LogisticRegression on V3_CONTINUOUS features "
        f"from train rows, evaluates AUC on test rows."
    )


def test_baseline_logreg_fit_returns_sklearn_estimator() -> None:
    """RED: fit_logreg(X, y) must return an sklearn estimator with
    predict_proba()."""
    if not LOGREG_PATH.exists():
        pytest.skip("script not yet implemented (P1.1-GREEN)")
    mod = _import_from(LOGREG_PATH, "baseline_logreg")
    assert hasattr(mod, "fit_logreg"), (
        "baseline_logreg.py must export fit_logreg(X_train, y_train)"
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 17)).astype(np.float32)  # 17 = V3_CONTINUOUS len
    y = (X[:, 0] > 0).astype(np.int32)  # learnable signal in feature 0
    est = mod.fit_logreg(X, y)
    assert hasattr(est, "predict_proba"), "fit_logreg must return sklearn-compatible estimator"
    proba = est.predict_proba(X[:5])
    assert proba.shape == (5, 2), f"predict_proba shape {proba.shape}, expected (5, 2)"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilities out of [0, 1]"


# ---------- result-aggregation contract ----------

def test_baseline_results_json_schema() -> None:
    """RED: artifacts/baselines/naive_results.json must contain the
    canonical keys downstream P1.5 + paper §6 will reference."""
    results_path = REPO_ROOT / "artifacts" / "baselines" / "naive_results.json"
    if not results_path.exists():
        pytest.skip("results not yet computed (P1.1-GREEN)")
    import json
    data = json.loads(results_path.read_text())
    expected_keys = {"last_bler_test_auc", "logreg_test_auc",
                     "n_test_rows", "positive_rate", "computed_at"}
    missing = expected_keys - data.keys()
    assert not missing, f"naive_results.json missing keys: {missing}"
    assert 0.0 <= data["last_bler_test_auc"] <= 1.0
    assert 0.0 <= data["logreg_test_auc"] <= 1.0
    assert 0.2 <= data["positive_rate"] <= 0.4, (
        f"positive_rate = {data['positive_rate']}, expected ~0.309 per paper §3.2"
    )
