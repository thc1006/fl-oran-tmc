"""Tests for Phase 2 minimum-viable sweep specification.

The sweep matrix lives at ``experiments/specs/phase2_min.yaml``. Per
ADR-001 D-22 + project state line 117-122, Phase 2 minimum is::

    3 archs (lstm, mamba, spiking_expand2)
    × 2 algorithms (fedavg, fedprox; MOON deferred per D-22)
    × 2 partitions (iid + dirichlet α=0.5)
    × 3 seeds
    = 36 cells × ~10 min/cell ≈ 6 hr GPU

A loader at ``scripts/_v7_spec_loader.py`` reads the YAML, validates,
and expands into per-cell V7Config kwarg dicts. Tests pin both:

* YAML structural contract (required keys, value types, registry
  membership)
* Expansion correctness (36 cells, no duplicates, MOON not present,
  per-arch lr overrides applied)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "experiments" / "specs" / "phase2_min.yaml"
LOADER_PATH = REPO_ROOT / "scripts" / "_v7_spec_loader.py"


def _load_loader():
    spec_obj = importlib.util.spec_from_file_location(
        "_v7_spec_loader", LOADER_PATH,
    )
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules["_v7_spec_loader"] = mod
    spec_obj.loader.exec_module(mod)
    return mod


def _load_metadata_helper():
    """Reuse Stage B helper to access registries for cross-validation."""
    p = REPO_ROOT / "scripts" / "_v7_cell_metadata.py"
    spec_obj = importlib.util.spec_from_file_location("_v7_cell_metadata", p)
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules.setdefault("_v7_cell_metadata", mod)
    spec_obj.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loader():
    return _load_loader()


@pytest.fixture(scope="module")
def spec_dict(loader):
    return loader.load_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def expanded(loader, spec_dict):
    return loader.expand_spec(spec_dict)


@pytest.fixture(scope="module")
def helper():
    return _load_metadata_helper()


# ---------------------------------------------------------------------------
# 1. Spec file presence + structural contract
# ---------------------------------------------------------------------------

def test_spec_file_exists():
    assert SPEC_PATH.is_file(), (
        f"Phase 2 minimum spec missing at {SPEC_PATH}"
    )


def test_spec_loads_as_dict(spec_dict):
    """``load_spec`` returns a plain dict (no class-instance YAML hooks)."""
    assert isinstance(spec_dict, dict)


@pytest.mark.parametrize(
    "required_key", ["archs", "algorithms", "partitions", "seeds", "shared"],
)
def test_spec_has_required_top_level_keys(spec_dict, required_key):
    """Each top-level key is mandatory; missing one would silently shrink
    the cross-product and undercount cells."""
    assert required_key in spec_dict, (
        f"required top-level key {required_key!r} missing"
    )


def test_spec_archs_is_list_of_known_archs(spec_dict, helper):
    """Every arch named in the spec must exist in the v6 ARCH_REGISTRY,
    otherwise the matrix driver crashes at the first cell."""
    archs = spec_dict["archs"]
    assert isinstance(archs, list) and len(archs) > 0
    known = helper.known_archs()
    for arch in archs:
        assert arch in known, (
            f"spec arch {arch!r} not in registry; known: {sorted(known)}"
        )


def test_spec_algorithms_is_list_of_known_algorithms(spec_dict, helper):
    """Every algorithm in the spec must exist in
    fl_oran.federated.algorithms.REGISTRY."""
    algos = spec_dict["algorithms"]
    assert isinstance(algos, list) and len(algos) > 0
    known = helper.known_algorithms()
    for algo in algos:
        assert algo in known, (
            f"spec algo {algo!r} not in registry; known: {sorted(known)}"
        )


def test_spec_does_not_include_moon(spec_dict):
    """MOON is deferred in Phase 1.5 / 2 minimum per ADR D-22.
    Including it would cause the matrix driver to raise
    NotImplementedError for every (MOON, non-LSTM) cell — silent
    sweep failure on 2/3 of the grid."""
    assert "moon" not in spec_dict["algorithms"]


def test_spec_partitions_well_formed(spec_dict):
    """Partitions must be a list of dicts each with 'mode' and (for
    dirichlet) 'alpha' + 'n_clients'."""
    parts = spec_dict["partitions"]
    assert isinstance(parts, list) and len(parts) >= 1
    for p in parts:
        assert isinstance(p, dict)
        assert p.get("mode") in ("iid", "dirichlet")
        if p["mode"] == "dirichlet":
            assert "alpha" in p
            assert "n_clients" in p
        elif p["mode"] == "iid":
            assert "n_clients" in p  # documented even though IID ignores it


def test_spec_seeds_are_unique_ints(spec_dict):
    seeds = spec_dict["seeds"]
    assert isinstance(seeds, list) and len(seeds) >= 1
    assert all(isinstance(s, int) and s >= 0 for s in seeds)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds: {seeds}"


# ---------------------------------------------------------------------------
# 2. Phase 2 minimum cardinality
# ---------------------------------------------------------------------------

def test_phase2_min_cell_count_is_36(expanded):
    """3 archs × 2 algos × 2 partitions × 3 seeds = 36 cells."""
    assert len(expanded) == 36


def test_phase2_min_archs_count(spec_dict):
    assert len(spec_dict["archs"]) == 3


def test_phase2_min_algos_count(spec_dict):
    assert len(spec_dict["algorithms"]) == 2


def test_phase2_min_partitions_count(spec_dict):
    assert len(spec_dict["partitions"]) == 2


def test_phase2_min_seeds_count(spec_dict):
    assert len(spec_dict["seeds"]) == 3


# ---------------------------------------------------------------------------
# 3. Expanded cells satisfy V7Config-compatible kwargs
# ---------------------------------------------------------------------------

def test_expanded_cells_have_all_v7config_kwargs(expanded):
    """Each expanded cell-dict must carry the V7Config keys actually used
    in the smoke tests (per tests/test_v7_fl_arch_agnostic.py)."""
    required = {
        "arch", "algorithm", "partition_mode", "n_clients",
        "num_rounds", "clients_per_round", "max_steps_per_round",
        "batch_size", "lr", "lr_warmup_rounds", "seq_len",
        "sample_ratio", "threshold", "seed",
    }
    for cell in expanded:
        missing = required - cell.keys()
        assert not missing, (
            f"expanded cell missing keys {missing}: {cell}"
        )


def test_expanded_cells_carry_canonical_name(expanded, helper):
    """Each cell carries a ``name`` field equal to the canonical
    ``cell_name(...)`` produced by ``_v7_cell_metadata`` so that the
    aggregator's directory enumeration agrees with the matrix driver."""
    for cell in expanded:
        expected = helper.cell_name(
            cell["arch"], cell["algorithm"],
            cell["partition_mode"], cell["seed"],
            alpha=cell.get("alpha"),
        )
        assert cell["name"] == expected, (
            f"name {cell['name']!r} != canonical {expected!r}"
        )


def test_expanded_cells_have_no_duplicate_names(expanded):
    """All 36 cell names must be distinct so the matrix driver does
    not overwrite outputs."""
    names = [c["name"] for c in expanded]
    assert len(set(names)) == 36, (
        f"duplicate cell names; {len(names) - len(set(names))} collisions"
    )


def test_expanded_cells_iid_alpha_is_none(expanded):
    for cell in expanded:
        if cell["partition_mode"] == "iid":
            assert cell.get("alpha") is None
        else:
            assert isinstance(cell["alpha"], (int, float))
            assert cell["alpha"] > 0


# ---------------------------------------------------------------------------
# 4. Per-arch lr override applied (ADR D-20)
# ---------------------------------------------------------------------------

def test_lr_overrides_applied_per_arch(expanded):
    """Per ADR D-20, Spiking variants need lr=1e-4; LSTM/Mamba use 5e-4.
    The expander must honour ``arch_overrides`` from the spec."""
    for cell in expanded:
        if cell["arch"] in ("lstm", "mamba"):
            assert cell["lr"] == pytest.approx(5e-4)
        elif cell["arch"] == "spiking_expand2":
            assert cell["lr"] == pytest.approx(1e-4)
        else:
            pytest.fail(f"unexpected arch in expanded spec: {cell['arch']}")


def test_shared_fields_propagate_to_every_cell(expanded):
    """Fields under ``shared:`` must appear on every cell with the
    same value (unless overridden per-arch)."""
    for cell in expanded:
        assert cell["num_rounds"] == 20
        assert cell["clients_per_round"] == 5
        assert cell["max_steps_per_round"] == 50
        assert cell["batch_size"] == 64
        assert cell["seq_len"] == 5
        assert cell["sample_ratio"] == pytest.approx(1.0)
        assert cell["threshold"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 5. validate_spec catches malformed input
# ---------------------------------------------------------------------------

def test_validate_spec_rejects_unknown_arch(loader, spec_dict):
    bad = dict(spec_dict)
    bad["archs"] = list(spec_dict["archs"]) + ["nonexistent_arch"]
    with pytest.raises(ValueError, match=r"arch"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_moon(loader, spec_dict):
    bad = dict(spec_dict)
    bad["algorithms"] = list(spec_dict["algorithms"]) + ["moon"]
    with pytest.raises(ValueError, match=r"[Mm]oon|D-22"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_missing_top_level(loader, spec_dict):
    bad = {k: v for k, v in spec_dict.items() if k != "seeds"}
    with pytest.raises(ValueError, match=r"seeds"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_duplicate_seeds(loader, spec_dict):
    bad = dict(spec_dict)
    bad["seeds"] = [42, 42, 0]
    with pytest.raises(ValueError, match=r"[Dd]uplicate|seed"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_duplicate_archs(loader, spec_dict):
    bad = dict(spec_dict)
    bad["archs"] = list(spec_dict["archs"]) + [spec_dict["archs"][0]]
    with pytest.raises(ValueError, match=r"[Dd]uplicate|arch"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_duplicate_algorithms(loader, spec_dict):
    bad = dict(spec_dict)
    bad["algorithms"] = list(spec_dict["algorithms"]) + [spec_dict["algorithms"][0]]
    with pytest.raises(ValueError, match=r"[Dd]uplicate|algorithm"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_arch_override_of_reserved_dimension_key(loader, spec_dict):
    """arch_overrides may set hyperparameters but NOT dimension keys
    (arch / algorithm / partition_mode / alpha / n_clients / seed / name).
    Overriding a dimension key would silently shift the cross-product
    and break aggregator grouping."""
    bad = dict(spec_dict)
    bad["arch_overrides"] = dict(spec_dict.get("arch_overrides") or {})
    bad["arch_overrides"]["lstm"] = {"lr": 5e-4, "seed": 9999}
    with pytest.raises(ValueError, match=r"reserved|seed"):
        loader.validate_spec(bad)


def test_validate_spec_rejects_duplicate_partitions(loader, spec_dict):
    """Two partition entries with the same (mode, alpha) would yield
    duplicate cell keys when expanded — silent overcounting."""
    bad = dict(spec_dict)
    bad["partitions"] = list(spec_dict["partitions"]) + [{"mode": "iid", "n_clients": 7}]
    with pytest.raises(ValueError, match=r"[Dd]uplicate|partition"):
        loader.validate_spec(bad)
