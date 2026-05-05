"""P1.3-RED methodology tests for FedBN algorithm class.

Verifies the FedBN FLAlgorithm correctly excludes BatchNorm (or, in our
no-BN models, LayerNorm/no-norm parameters appropriately) from server-side
aggregation. The actual paper claim (FedBN within FedAdam ±0.01) is
preregistered separately.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FEDBN_PATH = REPO_ROOT / "src" / "fl_oran" / "federated" / "algorithms" / "fedbn.py"


@pytest.mark.xfail(strict=True, reason="P1.3-GREEN: implement src/fl_oran/federated/algorithms/fedbn.py")
def test_fedbn_module_exists() -> None:
    """RED: src/fl_oran/federated/algorithms/fedbn.py must exist.
    xfail strict=True → when GREEN lands, remove this marker."""
    assert FEDBN_PATH.exists(), (
        f"{FEDBN_PATH} not found. Implement P1.3-GREEN: FedBN FLAlgorithm "
        f"that mirrors FedAvg's server_aggregate but skips BatchNorm "
        f"params (Li et al. 2021, ICLR; arXiv:2102.07623)."
    )


def test_fedbn_class_implements_flalgorithm_protocol() -> None:
    """RED: FedBN must expose init_state / local_train / server_aggregate."""
    if not FEDBN_PATH.exists():
        pytest.skip("FedBN not yet implemented (P1.3-GREEN)")
    from fl_oran.federated.algorithms.fedbn import FedBN
    for method in ("init_state", "local_train", "server_aggregate"):
        assert hasattr(FedBN, method), f"FedBN missing required method {method!r}"


def test_fedbn_skips_bn_params_in_aggregation() -> None:
    """RED: server_aggregate must skip parameters whose name matches the
    BatchNorm / personalisation pattern (typically `*.bn.*`, `*.batchnorm.*`,
    `*.norm.*`, or any explicitly listed in the FedBN spec).
    For our 3 archs (no BatchNorm), this means LayerNorm-equivalent params
    (or a configurable name-pattern list) should pass through unchanged."""
    if not FEDBN_PATH.exists():
        pytest.skip("FedBN not yet implemented (P1.3-GREEN)")
    from fl_oran.federated.algorithms.fedbn import FedBN, _is_personalised_param

    # Names that SHOULD be personalised (skipped from aggregation)
    assert _is_personalised_param("encoder.bn1.weight")
    assert _is_personalised_param("encoder.bn1.bias")
    assert _is_personalised_param("layer.batchnorm.running_mean")
    # Names that should NOT be personalised (regular aggregation)
    assert not _is_personalised_param("lstm1.weight_ih_l0")
    assert not _is_personalised_param("classifier.weight")
    assert not _is_personalised_param("embeddings.tr.weight")
