"""Tests for scripts/_v7_cell_metadata.py — Stage 2 sweep cell metadata.

The v7 layer extends v6's ``(arch_base, seed, suffix)`` cell-name model
to a 5-tuple ``(arch, algorithm, partition_mode, alpha, seed)``: every
Phase 2 sweep cell varies an FL algorithm and a partition mode in
addition to the architecture, so the v6 parser is insufficient.

Two design rules guide the API surface:

* ``cell_name`` produces a canonical name; ``parse_cell_name`` is its
  exact left-inverse. Round-trip equivalence is pinned by tests.
* The aggregator should NOT depend on parsing — fl_v7's ``summary.json``
  carries explicit fields. ``parse_cell_name`` is a defensive helper for
  tools that enumerate cell directories before opening the JSON.

Naming convention pinned by tests:

* IID:        ``v7_<arch>_<algorithm>_iid_s<seed>`` (no alpha)
* Dirichlet:  ``v7_<arch>_<algorithm>_dir_a<alpha_tag>_s<seed>`` where
              ``alpha_tag = f"{alpha:.2f}".replace(".", "p")``

The ``v7_`` prefix mirrors v5's ``v5_`` convention so cells from
different phases coexist in the same artifact root without colliding.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "_v7_cell_metadata.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("_v7_cell_metadata", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v7_cell_metadata"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def helper():
    return _load_helper()


# ---------------------------------------------------------------------------
# 1. Canonical cell_name() construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arch, algo, partition_mode, alpha, n_clients, seed, expected",
    [
        ("lstm", "fedavg", "iid", None, 7, 42, "v7_lstm_fedavg_iid_n7_s42"),
        ("mamba", "fedprox", "iid", None, 5, 0, "v7_mamba_fedprox_iid_n5_s0"),
        ("spiking_expand2", "fedavg", "iid", None, 7, 17,
         "v7_spiking_expand2_fedavg_iid_n7_s17"),
        ("lstm", "fedavg", "dirichlet", 0.5, 7, 42,
         "v7_lstm_fedavg_dirichlet_a0p50_n7_s42"),
        ("spiking_expand2", "fedprox", "dirichlet", 0.05, 7, 0,
         "v7_spiking_expand2_fedprox_dirichlet_a0p05_n7_s0"),
        ("mamba_expand2", "fedavg", "dirichlet", 10.0, 5, 23,
         "v7_mamba_expand2_fedavg_dirichlet_a10p00_n5_s23"),
        ("lstm", "fedavg", "dirichlet", 1.0, 7, 42,
         "v7_lstm_fedavg_dirichlet_a1p00_n7_s42"),
    ],
)
def test_cell_name_is_canonical(helper, arch, algo, partition_mode, alpha, n_clients, seed, expected):
    """``cell_name`` produces the documented canonical string for both
    IID (no alpha tag) and Dirichlet (with ``a<tag>``) cases. Format
    must match V7Config.__post_init__ — see
    ``test_cell_name_matches_v7config_post_init``."""
    assert helper.cell_name(
        arch, algo, partition_mode, seed, n_clients, alpha=alpha,
    ) == expected


def test_cell_name_iid_rejects_alpha(helper):
    """IID partitions ignore alpha by definition. Passing alpha for IID
    is an upstream bug (caller confused); raise so the bug surfaces at
    construction rather than during parse where it would silently drop
    information."""
    with pytest.raises(ValueError, match=r"alpha.*iid"):
        helper.cell_name("lstm", "fedavg", "iid", 42, 7, alpha=0.5)


def test_cell_name_dirichlet_requires_alpha(helper):
    """Dirichlet without alpha is meaningless. Construction must fail."""
    with pytest.raises(ValueError, match=r"alpha.*dirichlet"):
        helper.cell_name("lstm", "fedavg", "dirichlet", 42, 7, alpha=None)


def test_cell_name_unknown_partition_mode(helper):
    """Only iid and dirichlet are supported per partition.py."""
    with pytest.raises(ValueError, match=r"partition_mode"):
        helper.cell_name("lstm", "fedavg", "noniid_slice", 42, 7, alpha=0.5)


@pytest.mark.parametrize("bad_n", [0, -1, 1.5, "7", True])
def test_cell_name_rejects_invalid_n_clients(helper, bad_n):
    """n_clients must be a positive int (per V7Config.n_clients=7
    default)."""
    with pytest.raises(ValueError, match=r"n_clients"):
        helper.cell_name("lstm", "fedavg", "iid", 42, bad_n)


@pytest.mark.parametrize(
    "bad_alpha",
    [
        0.005,           # 3rd-decimal — would silently round to 0.01
        0.123,           # not in canonical grid; would round to 0.12
        -0.5,            # negative — meaningless for Dirichlet
        float("inf"),    # non-finite
        float("nan"),    # non-finite
    ],
)
def test_cell_name_dirichlet_rejects_unencodable_alpha(helper, bad_alpha):
    """Alpha values that don't round-trip cleanly at 2-decimal precision
    must raise — silent rounding (0.005 → 0p01) would let the same
    intent be encoded as different cells, breaking aggregator grouping."""
    with pytest.raises(ValueError):
        helper.cell_name("lstm", "fedavg", "dirichlet", 42, 7, alpha=bad_alpha)


@pytest.mark.parametrize("bad_arch", ["", None, 42])
def test_cell_name_rejects_invalid_arch(helper, bad_arch):
    """Empty / non-str arch must raise — silently producing
    ``v7__fedavg_iid_s42`` would be unparseable later."""
    with pytest.raises(ValueError):
        helper.cell_name(bad_arch, "fedavg", "iid", 42, 7)


def test_cell_name_rejects_arch_with_empty_segment(helper):
    """``foo__bar`` would parse-trip incorrectly (empty middle segment).
    Catch at construction."""
    with pytest.raises(ValueError):
        helper.cell_name("foo__bar", "fedavg", "iid", 42, 7)


@pytest.mark.parametrize("bad_algo", ["", None, 42])
def test_cell_name_rejects_invalid_algorithm(helper, bad_algo):
    """Same defensive contract as arch."""
    with pytest.raises(ValueError):
        helper.cell_name("lstm", bad_algo, "iid", 42, 7)


@pytest.mark.parametrize("bad_seed", [-1, "42", 1.5, True])
def test_cell_name_rejects_invalid_seed(helper, bad_seed):
    """Seed must be a non-negative int. ``"42"`` would format to a valid
    name but downstream lookups by integer seed would silently miss; a
    bool sneaks past int-checks and prints as ``True``."""
    with pytest.raises(ValueError):
        helper.cell_name("lstm", "fedavg", "iid", bad_seed, 7)


# ---------------------------------------------------------------------------
# 2. parse_cell_name() — exact left-inverse of cell_name()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arch, algo, partition_mode, alpha, n_clients, seed",
    [
        ("lstm", "fedavg", "iid", None, 7, 42),
        ("mamba", "fedprox", "iid", None, 5, 0),
        ("spiking_expand2", "fedavg", "iid", None, 7, 17),
        ("lstm", "fedavg", "dirichlet", 0.5, 7, 42),
        ("spiking_expand2", "fedprox", "dirichlet", 0.05, 7, 0),
        ("mamba_expand2", "fedavg", "dirichlet", 10.0, 5, 23),
        ("lstm", "fedavg", "dirichlet", 1.0, 7, 42),
        ("lstm", "fedavg", "dirichlet", 0.1, 7, 7),
    ],
)
def test_parse_round_trips_with_cell_name(helper, arch, algo, partition_mode, alpha, n_clients, seed):
    """For every tuple, ``parse_cell_name(cell_name(*tuple)) == tuple``."""
    name = helper.cell_name(arch, algo, partition_mode, seed, n_clients, alpha=alpha)
    parsed = helper.parse_cell_name(name)
    assert parsed["arch"] == arch
    assert parsed["algorithm"] == algo
    assert parsed["partition_mode"] == partition_mode
    if alpha is None:
        assert parsed["alpha"] is None
    else:
        assert parsed["alpha"] == pytest.approx(alpha, abs=1e-6)
    assert parsed["n_clients"] == n_clients
    assert parsed["seed"] == seed


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "lstm_fedavg_iid_n7_s42",                  # missing v7_ prefix
        "v7_lstm_fedavg",                          # missing partition + n + seed
        "v7_lstm_fedavg_iid_s42",                  # missing n<N>
        "v7_lstm_fedavg_iid_n7",                   # missing seed
        "v7_lstm_fedavg_iid_n7_sX",                # non-integer seed
        "v7_lstm_fedavg_iid_nX_s42",               # non-integer n_clients
        "v7_lstm_fedavg_iid_n0_s42",               # non-positive n_clients
        "v7_lstm_fedavg_dirichlet_n7_s42",         # dirichlet without _a<tag>
        "v7_lstm_fedavg_dirichlet_a0p5_n7_s42",    # alpha tag must be 2-decimal
        "v7_lstm_unknownalgo_iid_n7_s42",          # algorithm not in REGISTRY
        "v7__fedavg_iid_n7_s42",                   # empty arch
    ],
)
def test_parse_cell_name_rejects_malformed(helper, bad_name):
    """Defensive parsing: malformed names raise ValueError rather than
    silently returning bogus fields. Each case targets one specific
    failure mode the parser must catch."""
    with pytest.raises(ValueError):
        helper.parse_cell_name(bad_name)


def test_parse_handles_arch_with_underscore_unambiguously(helper):
    """``mamba_expand2`` contains an underscore. The parser must NOT
    greedily strip ``expand2_fedavg`` thinking ``expand2`` is the arch
    and ``fedavg`` is part of the partition tail. Must split arch from
    algorithm by suffix-matching against the algorithm registry."""
    name = "v7_mamba_expand2_fedavg_iid_n7_s42"
    parsed = helper.parse_cell_name(name)
    assert parsed["arch"] == "mamba_expand2"
    assert parsed["algorithm"] == "fedavg"


# ---------------------------------------------------------------------------
# Cross-validation: cell_name MUST match V7Config.__post_init__ byte-exact
# ---------------------------------------------------------------------------

def test_cell_name_matches_v7config_post_init(helper):
    """V7Config.__post_init__ is the on-disk source of truth for cell
    naming. ``_v7_cell_metadata.cell_name`` exists for tools that need
    the name without instantiating V7Config (heavier import). The two
    MUST agree byte-for-byte; this test pins the equivalence so any
    future drift in either side fails CI loudly.

    Skips if fl_v7 not yet importable — only relevant when the v7
    trainer module is on the Python path.
    """
    pytest.importorskip("fl_oran.training.fl_v7")
    from fl_oran.training.fl_v7 import V7Config

    cases = [
        # (kwargs, expected via cell_name)
        (dict(arch="lstm", algorithm="fedavg", partition_mode="iid",
              alpha=0.5, n_clients=7, seed=42),
         dict(partition_mode="iid", alpha=None, n_clients=7)),
        (dict(arch="mamba", algorithm="fedprox", partition_mode="iid",
              alpha=0.5, n_clients=5, seed=0),
         dict(partition_mode="iid", alpha=None, n_clients=5)),
        (dict(arch="spiking_expand2", algorithm="fedavg",
              partition_mode="dirichlet", alpha=0.05, n_clients=7, seed=17),
         dict(partition_mode="dirichlet", alpha=0.05, n_clients=7)),
        (dict(arch="mamba_expand2", algorithm="fedavg",
              partition_mode="dirichlet", alpha=10.0, n_clients=5, seed=23),
         dict(partition_mode="dirichlet", alpha=10.0, n_clients=5)),
    ]
    for v7_kwargs, helper_kwargs in cases:
        cfg = V7Config(**v7_kwargs)
        expected = helper.cell_name(
            v7_kwargs["arch"], v7_kwargs["algorithm"],
            helper_kwargs["partition_mode"], v7_kwargs["seed"],
            helper_kwargs["n_clients"], alpha=helper_kwargs["alpha"],
        )
        assert cfg.name == expected, (
            f"V7Config.__post_init__ produced {cfg.name!r} but "
            f"_v7_cell_metadata.cell_name produced {expected!r} — "
            "naming drift; fix one side to match the other"
        )


# ---------------------------------------------------------------------------
# 3. Registry loaders
# ---------------------------------------------------------------------------

def test_arch_registry_returns_v6_runner_registry(helper):
    """``arch_registry()`` loads ARCH_REGISTRY from
    ``experiments/run_v6_arch_sweep.py`` (single source of truth, same
    pattern as ``_v6_cell_metadata.runner_arch_registry``)."""
    reg = helper.arch_registry()
    assert isinstance(reg, dict)
    # The five archs that Stage 1 / Stage 2 reference must all be present.
    for required in ("lstm", "mamba", "mamba_expand2", "spiking", "spiking_expand2"):
        assert required in reg, (
            f"arch {required!r} missing from registry; known: {sorted(reg)}"
        )


def test_arch_registry_caches(helper):
    """Second call returns the same object — a 36-cell sweep loop must
    not re-exec the runner module each iteration."""
    assert helper.arch_registry() is helper.arch_registry()


def test_algorithm_registry_returns_fl_oran_REGISTRY(helper):
    """``algorithm_registry()`` returns the dict from
    ``fl_oran.federated.algorithms.REGISTRY``, which is populated by
    submodule import side-effects at package load."""
    reg = helper.algorithm_registry()
    assert isinstance(reg, dict)
    for required in ("fedavg", "fedprox", "fedadam", "scaffold", "feddyn", "moon"):
        assert required in reg, (
            f"algorithm {required!r} missing; known: {sorted(reg)}"
        )


def test_known_archs_and_algorithms_match_registries(helper):
    """``known_archs()`` / ``known_algorithms()`` accessors are exactly
    the keys of their respective registries."""
    assert helper.known_archs() == set(helper.arch_registry().keys())
    assert helper.known_algorithms() == set(helper.algorithm_registry().keys())


def test_no_arch_name_collides_with_algorithm_suffix(helper):
    """Defensive precondition for ``parse_cell_name``: no arch may end
    with ``_<algorithm>`` for any registered algorithm. If this ever
    fires, the parser's suffix-stripping logic would misattribute an
    arch's tail as the algorithm. Catch at module load time."""
    archs = helper.known_archs()
    algos = helper.known_algorithms()
    for arch in archs:
        for algo in algos:
            assert not arch.endswith(f"_{algo}"), (
                f"arch {arch!r} ends with _{algo}; would confuse parse_cell_name"
            )


# ---------------------------------------------------------------------------
# 4. atomic_write_text — re-exported from v6 (D-3 single source of truth)
# ---------------------------------------------------------------------------

def test_atomic_write_text_writes_and_overwrites(helper, tmp_path):
    """Re-exported helper must support both initial write and overwrite,
    leaving no .tmp files behind."""
    target = tmp_path / "summary.json"
    helper.atomic_write_text(target, json.dumps({"test_auc": 0.85}))
    assert json.loads(target.read_text()) == {"test_auc": 0.85}
    helper.atomic_write_text(target, json.dumps({"test_auc": 0.90, "extra": True}))
    assert json.loads(target.read_text()) == {"test_auc": 0.90, "extra": True}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_atomic_write_text_creates_parent_dirs(helper, tmp_path):
    """Crash-safe write must ``mkdir -p`` the parent so callers do not
    have to pre-create every cell directory."""
    target = tmp_path / "deep" / "nested" / "summary.json"
    helper.atomic_write_text(target, '{"x": 1}')
    assert target.exists()
    assert json.loads(target.read_text()) == {"x": 1}
