"""Tests for ``fl_v7._ALGO_REQUIRED_KWARGS`` (Phase 1.5g-1).

The static algorithm→required-kwargs map is the foundation that lets
the spec loader validate ``algo_kwargs`` *before* a cell ever reaches
GPU. Phase 2 minimum sweep wasted 22 GPU-min because all 18 fedprox
cells crashed at training time on
``FedProx.__init__() missing required keyword-only argument: 'mu'`` —
this map (+ spec-loader hookup in 1.5g-2) makes that bug class
undeployable: spec validation refuses to load before any cell launches.

Tests fall into three layers:

1. Shape + content of the static table (it exists, types are right,
   each known algorithm appears).
2. Consistency with ``_select_algorithm``'s registry — no orphan keys,
   no algorithms missing.
3. Truth check via ``inspect.signature``: each declared required kwarg
   really is required by the underlying class (no default, kw-only,
   not auto-filled by ``run_v7_sweep``). This guards against the
   table going stale when an algorithm's signature changes.
"""
from __future__ import annotations

import inspect

import pytest

from fl_oran.training.fl_v7 import _ALGO_REQUIRED_KWARGS
from fl_oran.federated.algorithms import get_algorithm


# fl_v7.run_v7_sweep injects these into every algorithm's __init__ before
# overlaying cfg.algo_kwargs (see _run_training_v7's algo_kwargs dict).
# The required-kwargs table must NOT list these — they aren't user-facing.
_AUTO_FILLED_BY_FL_V7 = {
    "max_steps", "batch_size", "grad_clip", "amp_enabled", "amp_dtype",
}


class TestStaticTable:
    def test_table_exists_and_is_dict(self):
        assert isinstance(_ALGO_REQUIRED_KWARGS, dict)

    def test_keys_are_strings_values_are_sets(self):
        for k, v in _ALGO_REQUIRED_KWARGS.items():
            assert isinstance(k, str), f"key {k!r} not str"
            assert isinstance(v, (set, frozenset)), (
                f"_ALGO_REQUIRED_KWARGS[{k!r}] is {type(v).__name__}, "
                f"expected set/frozenset"
            )

    def test_fedavg_has_no_required_kwargs(self):
        assert _ALGO_REQUIRED_KWARGS["fedavg"] == set()

    def test_fedprox_requires_mu(self):
        assert _ALGO_REQUIRED_KWARGS["fedprox"] == {"mu"}

    def test_fedadam_requires_server_lr(self):
        assert _ALGO_REQUIRED_KWARGS["fedadam"] == {"server_lr"}

    def test_scaffold_has_no_required_kwargs(self):
        assert _ALGO_REQUIRED_KWARGS["scaffold"] == set()

    def test_feddyn_requires_alpha(self):
        # FedDyn's regularization 'alpha' is an algorithm-internal field
        # — semantically distinct from V7Config.alpha (Dirichlet). The
        # spec passes them via different paths (cfg.alpha vs algo_kwargs).
        assert _ALGO_REQUIRED_KWARGS["feddyn"] == {"alpha"}


class TestRegistryConsistency:
    """Every algorithm reachable via get_algorithm() must appear in the
    table (no missing entries) and every table key must resolve via
    get_algorithm() (no orphans). MOON is excluded — it's deferred per
    ADR D-22 / D-16 and _select_algorithm raises NotImplementedError.
    """

    def test_all_table_keys_resolve_to_real_algorithms(self):
        for name in _ALGO_REQUIRED_KWARGS:
            cls = get_algorithm(name)
            assert cls is not None, f"{name!r} in table but not in registry"

    def test_no_unexpected_extra_algorithms(self):
        # Future-proofing: if someone adds a new algorithm class to the
        # registry but forgets to update the table, this test catches it.
        # The expected set is what Phase 1.5/2 supports per ADR D-22,
        # extended 2026-05 with FedSCAM + FedGMT for the SAM-family ablation,
        # then extended 2026-05-17 with FedMoSWA (Liu et al. arXiv:2507.20016)
        # for the Path D 480-cell scale-up sweep.
        expected = {
            "fedavg", "fedprox", "fedadam", "scaffold", "feddyn",
            "fedbn", "fedswa", "fedscam", "fedgmt", "fedmoswa",
        }
        assert set(_ALGO_REQUIRED_KWARGS.keys()) == expected, (
            "table drift — update _ALGO_REQUIRED_KWARGS and this test together"
        )


class TestSignatureTruth:
    """Each declared required kwarg must really be required by the
    underlying algorithm class — kw-only AND no default AND not in
    fl_v7's auto-fill set. If someone adds a default value to (e.g.)
    FedProx.mu, this test will (correctly) fail and force a table
    update.
    """

    @pytest.mark.parametrize("algo_name", list(_ALGO_REQUIRED_KWARGS))
    def test_declared_required_kwargs_are_actually_required(self, algo_name):
        cls = get_algorithm(algo_name)
        sig = inspect.signature(cls.__init__)
        for kw in _ALGO_REQUIRED_KWARGS[algo_name]:
            assert kw in sig.parameters, (
                f"{algo_name}.{kw!r} not in signature {list(sig.parameters)}"
            )
            param = sig.parameters[kw]
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{algo_name}.{kw} kind={param.kind}, expected KEYWORD_ONLY"
            )
            assert param.default is inspect.Parameter.empty, (
                f"{algo_name}.{kw} has default {param.default!r}; "
                f"either remove the default or remove from required set"
            )

    @pytest.mark.parametrize("algo_name", list(_ALGO_REQUIRED_KWARGS))
    def test_all_truly_required_kwargs_are_declared(self, algo_name):
        """Inverse of the above: every kw-only required parameter on the
        class (excluding fl_v7's auto-filled set) must appear in the
        table. Otherwise spec validation would silently let a buggy
        spec through.
        """
        cls = get_algorithm(algo_name)
        sig = inspect.signature(cls.__init__)
        truly_required = {
            name for name, p in sig.parameters.items()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is inspect.Parameter.empty
            and name not in _AUTO_FILLED_BY_FL_V7
        }
        # Special case: MOON's encode_fn is required but the class is
        # itself deferred (D-22) — MOON isn't in the table on purpose.
        # No other algorithm has hidden required kwargs; assert the
        # table covers everything truly required.
        declared = _ALGO_REQUIRED_KWARGS[algo_name]
        assert truly_required == declared, (
            f"{algo_name}: truly_required={truly_required} "
            f"vs declared={declared} — update the table"
        )
