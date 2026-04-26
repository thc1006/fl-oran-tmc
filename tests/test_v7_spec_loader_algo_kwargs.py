"""Tests for ``_v7_spec_loader`` algorithm-kwargs support (Phase 1.5g-2).

The original loader only accepted ``algorithms: [name1, name2, ...]``
and never propagated per-algorithm kwargs. Phase 2 minimum sweep
exposed the gap: 18 fedprox cells crashed at training time because the
spec couldn't say ``mu: 0.01``.

This file asserts the new loader contract:

A. ``algorithms`` accepts list-of-strings (back-compat) — kwargs default
   to empty dict; validation REJECTS the spec if the algorithm has any
   required kwargs (``_ALGO_REQUIRED_KWARGS[algo]``) — fail-fast at
   load time, not at training time.
B. ``algorithms`` accepts list-of-dicts ``{name: str, kwargs: dict}``;
   validation checks kwargs cover ``_ALGO_REQUIRED_KWARGS[name]`` and
   propagates the dict into each expanded cell as ``algo_kwargs``.
C. Mixed list (some strings, some dicts) is allowed — each element
   independently validated.
D. Cells emitted by ``expand_spec`` include an ``algo_kwargs`` key
   even for algorithms with no required kwargs (empty dict, not
   missing — keeps downstream V7Config(**cell) construction uniform).

The spec loader lives in ``scripts/`` (not ``src/``) so it's loaded by
absolute path, mirroring the launcher's bootstrap.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOADER_PATH = _REPO_ROOT / "scripts" / "_v7_spec_loader.py"


@pytest.fixture(scope="module")
def loader():
    spec = importlib.util.spec_from_file_location(
        "_v7_spec_loader_test_import", _LOADER_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_v7_spec_loader_test_import", mod)
    spec.loader.exec_module(mod)
    return mod


def _base_spec() -> dict:
    """Minimum valid spec — caller mutates ``algorithms`` per test."""
    return {
        "archs": ["lstm"],
        "algorithms": ["fedavg"],  # caller overrides
        "partitions": [{"mode": "iid", "n_clients": 7}],
        "seeds": [42],
        "shared": {
            "num_rounds": 1, "clients_per_round": 1,
            "max_steps_per_round": 1, "batch_size": 16,
        },
    }


# ---------------------------------------------------------------------------
# Contract A — list of strings (back-compat path)
# ---------------------------------------------------------------------------


class TestListOfStrings:
    def test_fedavg_string_form_validates_and_expands(self, loader):
        spec = _base_spec()
        spec["algorithms"] = ["fedavg"]
        loader.validate_spec(spec)  # no raise
        cells = loader.expand_spec(spec)
        assert len(cells) == 1
        # Empty algo_kwargs must still be present (uniform cell shape).
        assert cells[0]["algo_kwargs"] == {}

    def test_fedprox_string_form_REJECTED(self, loader):
        """fedprox requires 'mu'; bare string form lacks any kwargs and
        must fail at validate, not at training time. This is the entire
        Phase 2 motivation.
        """
        spec = _base_spec()
        spec["algorithms"] = ["fedprox"]
        with pytest.raises(ValueError, match=r"fedprox.*missing.*mu"):
            loader.validate_spec(spec)

    def test_feddyn_string_form_REJECTED(self, loader):
        spec = _base_spec()
        spec["algorithms"] = ["feddyn"]
        with pytest.raises(ValueError, match=r"feddyn.*missing.*alpha"):
            loader.validate_spec(spec)


# ---------------------------------------------------------------------------
# Contract B — list of dicts (new full form)
# ---------------------------------------------------------------------------


class TestListOfDicts:
    def test_fedprox_with_mu_validates_and_propagates(self, loader):
        spec = _base_spec()
        spec["algorithms"] = [{"name": "fedprox", "kwargs": {"mu": 0.01}}]
        loader.validate_spec(spec)
        cells = loader.expand_spec(spec)
        assert len(cells) == 1
        assert cells[0]["algorithm"] == "fedprox"
        assert cells[0]["algo_kwargs"] == {"mu": 0.01}

    def test_fedprox_dict_with_empty_kwargs_REJECTED(self, loader):
        """Explicit {kwargs: {}} should fail the same as bare string —
        otherwise users can hide a missing kwarg by spelling it out as
        empty. The cure is to be explicit about WHAT is missing.
        """
        spec = _base_spec()
        spec["algorithms"] = [{"name": "fedprox", "kwargs": {}}]
        with pytest.raises(ValueError, match=r"fedprox.*missing.*mu"):
            loader.validate_spec(spec)

    def test_dict_without_kwargs_key_means_empty(self, loader):
        """{name: fedavg} (no kwargs key) is equivalent to {name: fedavg,
        kwargs: {}}. Allowed for algorithms with no required kwargs.
        """
        spec = _base_spec()
        spec["algorithms"] = [{"name": "fedavg"}]
        loader.validate_spec(spec)
        cells = loader.expand_spec(spec)
        assert cells[0]["algo_kwargs"] == {}

    def test_dict_with_extra_optional_kwargs_passes_through(self, loader):
        """Optional kwargs (e.g. fedadam beta1) pass through unchanged
        — the table only enforces REQUIRED kwargs.
        """
        spec = _base_spec()
        spec["algorithms"] = [
            {"name": "fedadam", "kwargs": {"server_lr": 0.01, "beta1": 0.95}},
        ]
        loader.validate_spec(spec)
        cells = loader.expand_spec(spec)
        assert cells[0]["algo_kwargs"] == {"server_lr": 0.01, "beta1": 0.95}

    def test_unknown_kwarg_NOT_silently_dropped(self, loader):
        """Typos like ``mue`` instead of ``mu`` would silently degrade
        to default behavior at instantiation time. Validation should
        catch unknown kwargs against the algorithm's signature.
        """
        spec = _base_spec()
        spec["algorithms"] = [
            {"name": "fedprox", "kwargs": {"mu": 0.01, "muu": 0.02}},
        ]
        with pytest.raises(ValueError, match=r"fedprox.*unknown.*muu"):
            loader.validate_spec(spec)


# ---------------------------------------------------------------------------
# Contract C — mixed list
# ---------------------------------------------------------------------------


class TestMixedList:
    def test_mixed_string_and_dict(self, loader):
        spec = _base_spec()
        spec["algorithms"] = [
            "fedavg",
            {"name": "fedprox", "kwargs": {"mu": 0.01}},
        ]
        loader.validate_spec(spec)
        cells = loader.expand_spec(spec)
        # 1 arch × 2 algos × 1 partition × 1 seed = 2 cells
        assert len(cells) == 2
        algo_to_kwargs = {c["algorithm"]: c["algo_kwargs"] for c in cells}
        assert algo_to_kwargs == {"fedavg": {}, "fedprox": {"mu": 0.01}}


# ---------------------------------------------------------------------------
# Contract D — duplicate detection still works
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_duplicate_dict_form_detected(self, loader):
        spec = _base_spec()
        spec["algorithms"] = [
            {"name": "fedprox", "kwargs": {"mu": 0.01}},
            {"name": "fedprox", "kwargs": {"mu": 0.05}},
        ]
        with pytest.raises(ValueError, match=r"duplicate.*fedprox"):
            loader.validate_spec(spec)

    def test_duplicate_string_form_detected(self, loader):
        # Existing behavior — re-assert it survives the kwargs upgrade.
        spec = _base_spec()
        spec["algorithms"] = ["fedavg", "fedavg"]
        with pytest.raises(ValueError, match=r"duplicate.*fedavg"):
            loader.validate_spec(spec)

    def test_duplicate_mixed_form_detected(self, loader):
        spec = _base_spec()
        spec["algorithms"] = [
            "fedavg",
            {"name": "fedavg"},
        ]
        with pytest.raises(ValueError, match=r"duplicate.*fedavg"):
            loader.validate_spec(spec)


# ---------------------------------------------------------------------------
# Contract — MOON still rejected
# ---------------------------------------------------------------------------


class TestMoonStillRejected:
    def test_moon_string_rejected(self, loader):
        spec = _base_spec()
        spec["algorithms"] = ["moon"]
        with pytest.raises(ValueError, match=r"MOON"):
            loader.validate_spec(spec)

    def test_moon_dict_form_also_rejected(self, loader):
        spec = _base_spec()
        spec["algorithms"] = [
            {"name": "moon", "kwargs": {"mu": 0.01, "tau": 0.5}},
        ]
        with pytest.raises(ValueError, match=r"MOON"):
            loader.validate_spec(spec)
