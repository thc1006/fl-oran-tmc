"""Tests for scripts/_v6_cell_metadata.py — the shared cell-metadata helper.

This module is the single source of truth for going from a sweep cell
directory name (e.g. ``spiking_s42_t5sum_50k``) back to the
``(arch_ctor, kwargs)`` the cell was trained under. Both
``measure_v6_gpu_energy.py`` and ``recompute_v6_energy.py`` depend on it.

Round 4 root-cause: the prior version of ``recompute_v6_energy.py``
carried its own (incomplete) copy of this logic, and the two drifted
apart — the recompute script silently corrupted ``energy.json`` for
``spiking_t5sum``, ``spiking_lif_*`` and any ``_expand2`` ablation cell
because it never reconstructed those kwargs. These tests pin the
exact (cell_name → kwargs) mapping so a future drift fails CI loudly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "_v6_cell_metadata.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("_v6_cell_metadata", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v6_cell_metadata"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def helper():
    return _load_helper()


@pytest.mark.parametrize(
    "cell_name, expected_arch, expected_seed, expected_suffix",
    [
        ("lstm_s42", "lstm", 42, ""),
        ("lstm_s0_25k", "lstm", 0, "25k"),
        ("lstm_s0_50k", "lstm", 0, "50k"),
        ("lstm_s42_100k", "lstm", 42, "100k"),
        ("mamba_s11_25k", "mamba", 11, "25k"),
        ("mamba_s23_50k", "mamba", 23, "50k"),
        ("mamba_expand2_s1", "mamba_expand2", 1, ""),
        ("spiking_s7", "spiking", 7, ""),
        ("spiking_s17_lr5e4_25k", "spiking", 17, "lr5e4_25k"),
        ("spiking_s42_t5sum_50k", "spiking", 42, "t5sum_50k"),
        ("spiking_s2_t5", "spiking", 2, "t5"),
        ("spiking_s13_t5sum", "spiking", 13, "t5sum"),
        ("spiking_s42_lif_t05_b09", "spiking", 42, "lif_t05_b09"),
        ("spiking_expand2_s3", "spiking_expand2", 3, ""),
    ],
)
def test_parse_cell_dir(helper, cell_name, expected_arch, expected_seed, expected_suffix):
    arch, seed, suffix = helper.parse_cell_dir(cell_name)
    assert arch == expected_arch
    assert seed == expected_seed
    assert suffix == expected_suffix


def test_parse_cell_dir_rejects_malformed(helper):
    """Names without `_s<digit>` must raise rather than silently producing
    nonsense (`int("")` ValueError or `int("abc")` ValueError)."""
    with pytest.raises(ValueError):
        helper.parse_cell_dir("garbage_no_seed")
    with pytest.raises(ValueError):
        helper.parse_cell_dir("lstm_s")  # rest is empty
    with pytest.raises(ValueError):
        helper.parse_cell_dir("")


def test_t5sum_routes_to_t_inner_5_AND_decode_mode_sum(helper):
    """RR40-2 regression. The prior recompute script set t_inner=5 for
    any suffix containing 't5' but never set decode_mode='sum', so cells
    like spiking_t5sum_50k were rebuilt with decode_mode='majority' and
    their energy.json was corrupted. Lock the correct mapping here."""
    kw = helper.build_kwargs_from_suffix("spiking", "t5sum_50k")
    assert kw["t_inner"] == 5
    assert kw["decode_mode"] == "sum"


def test_t5_without_sum_routes_to_majority(helper):
    """Plain `_t5` (without `sum`) keeps decode_mode at the registry
    default (majority for spiking)."""
    kw = helper.build_kwargs_from_suffix("spiking", "t5")
    assert kw["t_inner"] == 5
    assert kw.get("decode_mode") != "sum"


def test_lif_kwargs_two_digit_beta(helper):
    """`lif_t05_b09` → threshold=0.5, beta=0.9."""
    kw = helper.build_kwargs_from_suffix("spiking", "lif_t05_b09")
    assert kw["lif_threshold"] == pytest.approx(0.5)
    assert kw["lif_beta"] == pytest.approx(0.9)


def test_lif_kwargs_three_digit_beta(helper):
    """`lif_t10_b095` → threshold=1.0, beta=0.95 (three-digit branch)."""
    kw = helper.build_kwargs_from_suffix("spiking", "lif_t10_b095")
    assert kw["lif_threshold"] == pytest.approx(1.0)
    assert kw["lif_beta"] == pytest.approx(0.95)


def test_expand2_pulls_from_registry(helper):
    """`spiking_expand2` and `mamba_expand2` must inherit registry defaults
    (backbone_d_model + backbone_expand). RR40-1 regression: the prior
    recompute script had no ARCH_CTOR entry for these and silently skipped."""
    spk = helper.build_kwargs_from_suffix("spiking_expand2", "")
    assert spk["backbone_d_model"] == 56
    assert spk["backbone_expand"] == 2
    mam = helper.build_kwargs_from_suffix("mamba_expand2", "")
    assert mam["backbone_d_model"] == 48
    assert mam["backbone_expand"] == 2


def test_unknown_arch_returns_empty_kwargs(helper):
    """A future arch not yet in the registry must return {} so the caller
    falls back to constructor defaults — never a KeyError."""
    assert helper.build_kwargs_from_suffix("future_unknown", "anything") == {}


def test_atomic_write_text_replaces_atomically(helper, tmp_path):
    """Writing must go through a tempfile + os.replace, not a plain
    truncating write — otherwise a crash mid-write leaves a half-written
    JSON that breaks the aggregator."""
    target = tmp_path / "energy.json"
    helper.atomic_write_text(target, json.dumps({"a": 1}))
    assert target.exists()
    assert json.loads(target.read_text()) == {"a": 1}
    # Overwrite must also work.
    helper.atomic_write_text(target, json.dumps({"a": 2, "b": [1, 2]}))
    assert json.loads(target.read_text()) == {"a": 2, "b": [1, 2]}
    # No leftover .tmp files in the directory.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_atomic_write_creates_parent_dir(helper, tmp_path):
    target = tmp_path / "newdir" / "energy.json"
    helper.atomic_write_text(target, '{"x": 1}')
    assert target.exists()


def test_runner_arch_registry_caches(helper):
    """Second call must return the same dict object (cached) so a
    150-cell loop doesn't re-import the runner module 150 times."""
    a = helper.runner_arch_registry()
    b = helper.runner_arch_registry()
    assert a is b


def test_known_arches_matches_registry(helper):
    """`known_arches()` must reflect the runner's ARCH_REGISTRY exactly."""
    expected = set(helper.runner_arch_registry().keys())
    assert helper.known_arches() == expected
    # Defensive: every arch must be free of '_s' so parse_cell_dir works.
    for a in expected:
        assert "_s" not in a, a
