"""Shared metadata helpers for v7 (Phase 2 FL × architecture) cells.

The v7 sweep crosses three dimensions that v6 did not vary:

* ``algorithm`` — one of {fedavg, fedprox, fedadam, scaffold, feddyn, moon}
  from ``fl_oran.federated.algorithms.REGISTRY``. (MOON is only valid
  for the LSTM arch per ADR D-22; the registry still lists it because
  fl_v5 uses it; fl_v7 raises NotImplementedError on MOON × non-LSTM.)
* ``partition_mode`` — ``iid`` or ``dirichlet`` per
  ``data_v2/partition.py``. IID partitions by ``bs_id`` (=7 ColO-RAN
  gNBs) and ignores ``n_clients``; Dirichlet respects ``n_clients`` and
  draws per-slice proportions from ``Dir(alpha)``.
* ``alpha`` — only meaningful for Dirichlet; ``None`` for IID.

The cell-name convention is therefore (must match
``fl_oran.training.fl_v7.V7Config.__post_init__`` byte-for-byte —
cross-validated by ``test_cell_name_matches_v7config_post_init``):

* IID:        ``v7_<arch>_<algorithm>_iid_n<N>_s<seed>``
* Dirichlet:  ``v7_<arch>_<algorithm>_dirichlet_a<alpha_tag>_n<N>_s<seed>``

where ``alpha_tag = f"{alpha:.2f}".replace(".", "p")`` and ``N`` is
``n_clients``. Two-decimal alpha formatting guarantees
``0.05 / 0.10 / 0.50 / 1.00 / 10.00`` map to distinct, unambiguous
tags. ``n_clients`` is included in every cell name (per V7Config fix
``a294d28``) so future ``n_clients`` ablation sweeps cannot silently
collide with the canonical ``n=7`` cells.

This module is the canonical place for that convention. The aggregator
should NOT depend on parsing — fl_v7's ``summary.json`` carries
explicit ``arch``, ``algorithm``, ``partition_mode``, ``alpha``,
``seed`` fields. ``parse_cell_name`` is provided for tools that
enumerate cell directories before opening the JSON, and as a defensive
double-check that the runner's name agrees with summary contents.

Public API:

* :func:`arch_registry` — cached load of v6 runner's ARCH_REGISTRY
  (single source of truth for arch ctor + per-arch hyperparameters).
* :func:`algorithm_registry` — direct re-export of
  ``fl_oran.federated.algorithms.REGISTRY``.
* :func:`known_archs` / :func:`known_algorithms` — set accessors.
* :func:`cell_name(arch, algorithm, partition_mode, seed, *, alpha)` —
  canonical name builder.
* :func:`parse_cell_name(name)` — exact left-inverse, returns dict.
* :func:`atomic_write_text(path, content)` — re-exported from v6 to
  satisfy D-3 single-source-of-truth without promoting the helper out
  of ``scripts/`` (it has v6-private siblings that should not leak).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RUNNER_PATH = _SCRIPTS_DIR.parent / "experiments" / "run_v6_arch_sweep.py"
_V6_HELPER_PATH = _SCRIPTS_DIR / "_v6_cell_metadata.py"

_RUNNER_CACHE: dict | None = None
_V6_HELPER_CACHE = None


def _load_v6_helper():
    """Lazy-load _v6_cell_metadata so we can re-export atomic_write_text
    without duplicating its 30 lines (D-3). Cached at module scope."""
    global _V6_HELPER_CACHE
    if _V6_HELPER_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "_v6_cell_metadata", _V6_HELPER_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load v6 helper from {_V6_HELPER_PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_v6_cell_metadata", mod)
        spec.loader.exec_module(mod)
        _V6_HELPER_CACHE = mod
    return _V6_HELPER_CACHE


def arch_registry() -> dict:
    """Load and cache ARCH_REGISTRY from experiments/run_v6_arch_sweep.py.

    Uses ``importlib.util.spec_from_file_location`` so we never mutate
    ``sys.path`` and never collide with other modules. Cached at module
    scope so a 36-cell Phase 2 sweep loop pays the import cost once.

    Identical pattern to ``_v6_cell_metadata.runner_arch_registry`` —
    the v6 runner remains the single source of truth for arch ctor +
    per-arch hyperparameters; v7 reuses it verbatim.
    """
    global _RUNNER_CACHE
    if _RUNNER_CACHE is None:
        spec = importlib.util.spec_from_file_location("_v7_runner", _RUNNER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load runner module from {_RUNNER_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RUNNER_CACHE = mod.ARCH_REGISTRY
    return _RUNNER_CACHE


def algorithm_registry() -> dict:
    """Return ``fl_oran.federated.algorithms.REGISTRY`` directly.

    Unlike ARCH_REGISTRY (which lives in an experiments/ script), the FL
    algorithm registry is a proper package surface, so a standard import
    is appropriate. Submodule import side-effects in the package's
    ``__init__.py`` populate the dict before we read it.
    """
    from fl_oran.federated.algorithms import REGISTRY
    return REGISTRY


def known_archs() -> set[str]:
    """All arch keys understood by the v6 runner (computed lazily)."""
    return set(arch_registry().keys())


def known_algorithms() -> set[str]:
    """All algorithm keys understood by the FL algorithm registry."""
    return set(algorithm_registry().keys())


# ---------------------------------------------------------------------------
# Canonical cell name <-> structured fields
# ---------------------------------------------------------------------------

_VALID_PARTITION_MODES = ("iid", "dirichlet")
_ALPHA_TAG_RE = re.compile(r"^(\d+)p(\d{2})$")  # e.g. "0p50", "10p00"
_SEED_RE = re.compile(r"^s(\d+)$")
_NCLIENTS_RE = re.compile(r"^n(\d+)$")
_PREFIX = "v7_"


def _alpha_to_tag(alpha: float) -> str:
    """Format alpha as a two-decimal tag with ``.`` → ``p``.

    ``0.05 → "0p05"``, ``0.50 → "0p50"``, ``1.00 → "1p00"``,
    ``10.00 → "10p00"``. Two decimals are required so the round-trip is
    lossless for the alpha grid {0.05, 0.10, 0.50, 1.0, 10.0}.

    Rejects alpha values that don't round-trip cleanly at 2 decimals
    (e.g. ``0.005`` would format as ``"0p01"`` and silently parse back
    as ``0.01``). Also rejects negative / NaN / Inf — the canonical
    alpha grid is strictly positive and finite.
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"alpha must be int or float, got {type(alpha).__name__}")
    alpha_f = float(alpha)
    if alpha_f != alpha_f or alpha_f in (float("inf"), float("-inf")):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if alpha_f < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}")
    rounded = round(alpha_f, 2)
    if abs(alpha_f - rounded) > 1e-9:
        raise ValueError(
            f"alpha={alpha!r} cannot be encoded losslessly at 2-decimal "
            f"precision (rounds to {rounded}). Use the canonical grid "
            "{0.05, 0.10, 0.50, 1.00, 10.00} or extend the encoding."
        )
    return f"{alpha_f:.2f}".replace(".", "p")


def _tag_to_alpha(tag: str) -> float:
    """Reverse of ``_alpha_to_tag``. Raises ValueError on malformed tag.

    Enforces the two-decimal contract; ``"0p5"`` raises rather than
    silently parsing as 0.5 — that would let the same alpha be encoded
    two different ways and break aggregator grouping.
    """
    m = _ALPHA_TAG_RE.match(tag)
    if not m:
        raise ValueError(
            f"alpha tag {tag!r} must match <int>p<2-digit-int> "
            "(e.g. '0p50', '10p00')"
        )
    int_part, dec_part = m.groups()
    return float(f"{int_part}.{dec_part}")


def cell_name(arch: str, algorithm: str, partition_mode: str, seed: int,
              n_clients: int, *, alpha: float | None = None) -> str:
    """Build the canonical v7 cell directory name.

    Format MUST match ``V7Config.__post_init__`` byte-for-byte (the
    ``test_cell_name_matches_v7config_post_init`` test cross-validates
    on every change). V7Config is the on-disk source of truth; this
    helper exists for tools that need to predict the name without
    instantiating a V7Config (heavier import).

    Examples::

        cell_name("lstm", "fedavg", "iid", 42, 7)
            → "v7_lstm_fedavg_iid_n7_s42"
        cell_name("spiking_expand2", "fedprox", "dirichlet", 0, 7,
                  alpha=0.05)
            → "v7_spiking_expand2_fedprox_dirichlet_a0p05_n7_s0"

    Validation:

    * ``partition_mode`` must be one of ``iid`` / ``dirichlet``.
    * ``alpha`` must be ``None`` for IID and a finite float for
      Dirichlet — the inverse of the ambiguous "alpha is silently
      ignored" pattern, which would let bugs in the caller (alpha
      meant for a different cell) leak in undetected.
    * ``n_clients`` must be a positive int — included in the name per
      V7Config fix ``a294d28`` to prevent silent collision between
      n=5 and n=7 cells in future ablation sweeps.

    The function does NOT validate ``arch`` / ``algorithm`` against
    their registries. That allows synthetic testing with arbitrary
    arch/algo strings; production callers should validate upstream.
    """
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"arch must be non-empty str, got {arch!r}")
    if "_" in arch and any(not seg for seg in arch.split("_")):
        raise ValueError(
            f"arch {arch!r} contains empty underscore segment "
            "(e.g. 'foo__bar'); would not round-trip through parse_cell_name"
        )
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError(f"algorithm must be non-empty str, got {algorithm!r}")
    if "_" in algorithm and any(not seg for seg in algorithm.split("_")):
        raise ValueError(
            f"algorithm {algorithm!r} contains empty underscore segment"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be int, got {type(seed).__name__}: {seed!r}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    if not isinstance(n_clients, int) or isinstance(n_clients, bool):
        raise ValueError(
            f"n_clients must be int, got {type(n_clients).__name__}: {n_clients!r}"
        )
    if n_clients <= 0:
        raise ValueError(f"n_clients must be > 0, got {n_clients!r}")
    if partition_mode not in _VALID_PARTITION_MODES:
        raise ValueError(
            f"partition_mode {partition_mode!r} not in {_VALID_PARTITION_MODES}"
        )
    if partition_mode == "iid" and alpha is not None:
        raise ValueError(
            "alpha must be None for partition_mode='iid' (IID partitions "
            "by bs_id and ignores alpha; passing alpha indicates upstream "
            "confusion)"
        )
    if partition_mode == "dirichlet" and alpha is None:
        raise ValueError(
            "alpha is required for partition_mode='dirichlet'"
        )
    if partition_mode == "iid":
        return f"{_PREFIX}{arch}_{algorithm}_iid_n{n_clients}_s{seed}"
    return (
        f"{_PREFIX}{arch}_{algorithm}_dirichlet_"
        f"a{_alpha_to_tag(alpha)}_n{n_clients}_s{seed}"
    )


def parse_cell_name(name: str) -> dict:
    """Parse a canonical v7 cell name back into its 6-tuple of fields.

    Returns
    ``{"arch", "algorithm", "partition_mode", "alpha", "n_clients", "seed"}``.
    ``alpha`` is ``None`` for IID, ``float`` for Dirichlet.

    Right-to-left strip order:
      1. ``s<seed>``
      2. ``n<n_clients>``
      3. partition tail (``iid`` or ``dirichlet_a<tag>``)
      4. remainder = ``<arch>_<algorithm>``; algorithm matched
         longest-first against the registry

    The right-to-left strip is robust to arch names that themselves
    contain ``_iid`` / ``_dirichlet`` / ``_n<digit>`` / ``_s<digit>``
    substrings — only the trailing segment is inspected at each step.

    Raises ValueError for any malformed input — never returns silently
    bogus fields.
    """
    if not isinstance(name, str) or not name.startswith(_PREFIX):
        raise ValueError(f"v7 cell name must start with {_PREFIX!r}: got {name!r}")
    body = name[len(_PREFIX):]
    if not body:
        raise ValueError(f"empty body after {_PREFIX!r}: {name!r}")

    parts = body.split("_")
    # Min: <arch>_<algo>_iid_n<N>_s<seed> = 5 segments.
    if len(parts) < 5:
        raise ValueError(
            f"v7 cell name needs at least <arch>_<algo>_<partition>_n<N>_<seed>; "
            f"got only {len(parts)} segments in {name!r}"
        )

    # Trailing seed.
    seed_match = _SEED_RE.match(parts[-1])
    if not seed_match:
        raise ValueError(f"trailing segment {parts[-1]!r} is not 's<int>': {name!r}")
    seed = int(seed_match.group(1))
    parts = parts[:-1]

    # n_clients.
    nclients_match = _NCLIENTS_RE.match(parts[-1])
    if not nclients_match:
        raise ValueError(
            f"second-to-last segment {parts[-1]!r} is not 'n<int>': {name!r}"
        )
    n_clients = int(nclients_match.group(1))
    if n_clients <= 0:
        raise ValueError(f"n_clients must be > 0, got {n_clients} in {name!r}")
    parts = parts[:-1]

    # Partition tail: either ``..._iid`` or ``..._dirichlet_a<tag>``.
    if parts[-1] == "iid":
        partition_mode = "iid"
        alpha: float | None = None
        parts = parts[:-1]
    elif (len(parts) >= 2 and parts[-2] == "dirichlet"
          and parts[-1].startswith("a")):
        partition_mode = "dirichlet"
        alpha_tag = parts[-1][1:]  # strip leading 'a'
        alpha = _tag_to_alpha(alpha_tag)
        parts = parts[:-2]
    else:
        raise ValueError(
            f"partition tail must be '_iid' or '_dirichlet_a<tag>': {name!r}"
        )

    if not parts:
        raise ValueError(f"missing arch and algorithm segments: {name!r}")

    # Suffix-match algorithm (longest-first) against the registry. The
    # remainder before the matched algorithm is the arch.
    algos = sorted(known_algorithms(), key=len, reverse=True)
    matched_algo: str | None = None
    arch_parts: list[str] = []
    for algo in algos:
        algo_segments = algo.split("_")
        if len(algo_segments) > len(parts):
            continue
        tail = parts[-len(algo_segments):]
        if tail == algo_segments:
            matched_algo = algo
            arch_parts = parts[: -len(algo_segments)]
            break
    if matched_algo is None:
        raise ValueError(
            f"could not match algorithm in {name!r}; "
            f"known: {sorted(known_algorithms())}"
        )
    arch = "_".join(arch_parts)
    if not arch_parts or not arch or any(not p for p in arch_parts):
        raise ValueError(f"empty or malformed arch segment in {name!r}")

    return {
        "arch": arch,
        "algorithm": matched_algo,
        "partition_mode": partition_mode,
        "alpha": alpha,
        "n_clients": n_clients,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Re-export atomic_write_text from v6 (D-3 single source of truth)
# ---------------------------------------------------------------------------


def atomic_write_text(path, content) -> None:
    """Crash-safe text write; thin wrapper that delegates to v6's
    canonical implementation. v7 deliberately does not redefine the
    body to keep the on-disk crash semantics provably identical across
    the v6 (Stage 1) and v7 (Stage 2) sweep families."""
    return _load_v6_helper().atomic_write_text(path, content)
