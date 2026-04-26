"""Phase 2 sweep specification loader + expander.

Reads ``experiments/specs/*.yaml`` and produces a flat list of per-cell
V7Config-compatible kwarg dicts that the matrix driver can pass
directly to ``fl_v7.run_v7_sweep``.

Why a separate loader (vs inline dict in the matrix driver)?

* The 36-cell sweep is data, not code. Editing YAML to change seed list
  / alpha grid / arch set should not require touching Python.
* Reviewers can audit the sweep matrix without reading code.
* Multiple sweep specs (phase2_min, phase2_full, phase3_dp) coexist as
  sibling YAMLs without code duplication.

Validation contract:

* ``archs`` must be a non-empty list of keys present in
  ``_v7_cell_metadata.arch_registry()``.
* ``algorithms`` must be a non-empty list present in
  ``_v7_cell_metadata.algorithm_registry()``. ``moon`` is rejected
  per ADR D-22 (deferred in Phase 1.5 / 2 minimum).
* ``partitions`` is a list of dicts; each must have ``mode`` ∈
  {iid, dirichlet}. Dirichlet partitions must include ``alpha`` and
  ``n_clients``.
* ``seeds`` is a list of unique non-negative ints.
* ``shared`` is a dict of fields applied to every cell.
* ``arch_overrides`` (optional) is ``{arch: {field: value}}`` —
  per-arch overrides applied last (winning over ``shared``).

The expander returns a list whose length is exactly
``len(archs) × len(algorithms) × len(partitions) × len(seeds)``.
Each cell carries a ``name`` set to the canonical
``cell_name(...)`` so the aggregator's directory enumeration agrees
with the matrix driver's outputs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Strict YAML loader — rejects duplicate keys
# ---------------------------------------------------------------------------
#
# yaml.safe_load silently overwrites duplicate keys with the last value.
# That masks copy-paste bugs in long sweep specs (e.g. accidentally
# repeating ``seeds:`` after editing). We override the mapping
# constructor on a SafeLoader subclass to raise instead.

class _NoDupSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping_no_dup(loader, node, deep: bool = False):
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"duplicate key {key!r} in YAML mapping "
                f"(line {key_node.start_mark.line + 1})"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDupSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_dup,
)


_SCRIPTS_DIR = Path(__file__).resolve().parent
_METADATA_PATH = _SCRIPTS_DIR / "_v7_cell_metadata.py"

_METADATA_CACHE = None


def _metadata():
    """Lazy-load _v7_cell_metadata once per process."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "_v7_cell_metadata", _METADATA_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"could not load _v7_cell_metadata from {_METADATA_PATH}"
            )
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_v7_cell_metadata", mod)
        spec.loader.exec_module(mod)
        _METADATA_CACHE = mod
    return _METADATA_CACHE


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_spec(path) -> dict:
    """Load a sweep specification YAML into a plain dict.

    Uses a SafeLoader subclass that (a) refuses class-instance
    constructors and (b) raises on duplicate keys — vanilla
    ``yaml.safe_load`` silently keeps only the last duplicate, which
    masks copy-paste bugs in long sweep specs.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sweep spec not found: {p}")
    with p.open() as fh:
        data = yaml.load(fh, Loader=_NoDupSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"sweep spec must be a YAML mapping; got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = ("archs", "algorithms", "partitions", "seeds", "shared")
_VALID_PARTITION_MODES = ("iid", "dirichlet")


def validate_spec(spec: dict) -> None:
    """Raise ValueError on any structural / semantic problem.

    Checks each required top-level key, then drills into archs (against
    the v6 registry), algorithms (against the FL registry; moon
    rejected), partitions (mode + required sub-keys), and seeds (non-
    negative, distinct).
    """
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")

    for key in _REQUIRED_TOP_KEYS:
        if key not in spec:
            raise ValueError(f"spec missing required top-level key: {key!r}")

    helper = _metadata()

    archs = spec["archs"]
    if not isinstance(archs, list) or not archs:
        raise ValueError("spec['archs'] must be a non-empty list")
    if len(set(archs)) != len(archs):
        raise ValueError(f"duplicate archs in spec: {archs!r}")
    known_archs = helper.known_archs()
    for a in archs:
        if a not in known_archs:
            raise ValueError(
                f"unknown arch {a!r}; known: {sorted(known_archs)}"
            )

    algos = spec["algorithms"]
    if not isinstance(algos, list) or not algos:
        raise ValueError("spec['algorithms'] must be a non-empty list")
    if len(set(algos)) != len(algos):
        raise ValueError(f"duplicate algorithms in spec: {algos!r}")
    known_algos = helper.known_algorithms()
    for a in algos:
        if a not in known_algos:
            raise ValueError(
                f"unknown algorithm {a!r}; known: {sorted(known_algos)}"
            )
        if a == "moon":
            raise ValueError(
                "MOON is deferred in Phase 1.5 / Phase 2 minimum per "
                "ADR D-22 — fl_v7._select_algorithm raises "
                "NotImplementedError. Remove 'moon' from spec or "
                "revisit D-22 first."
            )

    partitions = spec["partitions"]
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("spec['partitions'] must be a non-empty list")
    # Two partition entries with same (mode, alpha) would produce
    # duplicate cell keys downstream. Detect at validation time.
    seen_partition_keys: set = set()
    for p in partitions:
        if isinstance(p, dict):
            key = (p.get("mode"), p.get("alpha") if p.get("mode") == "dirichlet" else None)
            if key in seen_partition_keys:
                raise ValueError(
                    f"duplicate partition entry (mode={key[0]!r}, alpha={key[1]!r}) "
                    f"in spec — would generate duplicate cells"
                )
            seen_partition_keys.add(key)
    for p in partitions:
        if not isinstance(p, dict):
            raise ValueError(f"partition entry must be a dict, got {p!r}")
        mode = p.get("mode")
        if mode not in _VALID_PARTITION_MODES:
            raise ValueError(
                f"partition mode {mode!r} must be in {_VALID_PARTITION_MODES}"
            )
        if mode == "dirichlet":
            if "alpha" not in p:
                raise ValueError(
                    f"dirichlet partition missing 'alpha': {p!r}"
                )
            if "n_clients" not in p:
                raise ValueError(
                    f"dirichlet partition missing 'n_clients': {p!r}"
                )
            try:
                alpha = float(p["alpha"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"dirichlet alpha {p['alpha']!r} not a float: {exc}"
                ) from exc
            if alpha <= 0:
                raise ValueError(
                    f"dirichlet alpha must be > 0; got {alpha}"
                )
        elif mode == "iid":
            if "n_clients" not in p:
                raise ValueError(
                    f"iid partition entry must declare 'n_clients' "
                    f"(documented even though IID partitions by bs_id "
                    f"and ignores it): {p!r}"
                )
            if "alpha" in p:
                raise ValueError(
                    f"iid partition must not include 'alpha' (alpha is "
                    f"meaningless for IID; expand_spec would silently "
                    f"drop it, masking the upstream confusion): {p!r}"
                )

    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("spec['seeds'] must be a non-empty list")
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError(f"seeds must be non-negative ints; got {seeds!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in spec: {seeds!r}")

    shared = spec["shared"]
    if not isinstance(shared, dict):
        raise ValueError(f"spec['shared'] must be a dict, got {type(shared).__name__}")

    overrides = spec.get("arch_overrides", {})
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(
            f"arch_overrides must be a dict or absent, got {type(overrides).__name__}"
        )
    if overrides:
        # Reserved keys that arch_overrides MUST NOT touch — overriding
        # these would silently change the cross-product dimensions and
        # break aggregator grouping / cell-name canonicality.
        reserved = {
            "arch", "algorithm", "partition_mode", "alpha",
            "n_clients", "seed", "name",
        }
        for arch, fields in overrides.items():
            if arch not in archs:
                raise ValueError(
                    f"arch_overrides[{arch!r}] but {arch!r} not in spec['archs']"
                )
            if not isinstance(fields, dict):
                raise ValueError(
                    f"arch_overrides[{arch!r}] must be a dict, got "
                    f"{type(fields).__name__}"
                )
            forbidden = reserved & fields.keys()
            if forbidden:
                raise ValueError(
                    f"arch_overrides[{arch!r}] tries to override reserved "
                    f"dimension keys {sorted(forbidden)}; only hyperparameter "
                    "fields are allowed"
                )


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def expand_spec(spec: dict) -> list[dict]:
    """Cross-product the spec into one V7Config kwarg dict per cell.

    Calls :func:`validate_spec` first so callers cannot expand a
    malformed spec. Each cell's ``name`` is set to the canonical
    ``cell_name(...)`` (see ``_v7_cell_metadata``) so cell directory
    names agree across the matrix driver, the aggregator, and any
    ad-hoc tool that enumerates cells.

    Layered field merge per cell:
      1. ``shared``  (every cell)
      2. partition fields (``partition_mode``, ``alpha`` if Dirichlet,
         ``n_clients``)
      3. ``algorithm`` and ``arch``
      4. ``arch_overrides[arch]`` (wins over shared — typically just
         ``lr`` per ADR D-20)
      5. ``seed`` and ``name`` (canonical)
    """
    validate_spec(spec)
    helper = _metadata()
    overrides = spec.get("arch_overrides", {}) or {}
    cells: list[dict] = []
    for arch in spec["archs"]:
        for algo in spec["algorithms"]:
            for partition in spec["partitions"]:
                for seed in spec["seeds"]:
                    cell: dict[str, Any] = dict(spec["shared"])
                    cell["partition_mode"] = partition["mode"]
                    if partition["mode"] == "dirichlet":
                        cell["alpha"] = float(partition["alpha"])
                    else:
                        cell["alpha"] = None
                    cell["n_clients"] = partition["n_clients"]
                    cell["arch"] = arch
                    cell["algorithm"] = algo
                    arch_over = overrides.get(arch, {})
                    if isinstance(arch_over, dict):
                        cell.update(arch_over)
                    cell["seed"] = seed
                    cell["name"] = helper.cell_name(
                        arch, algo, partition["mode"], seed,
                        alpha=cell["alpha"],
                    )
                    cells.append(cell)
    return cells
