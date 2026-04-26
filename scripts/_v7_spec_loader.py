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
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml


# Kwargs that ``fl_v7.run_v7_sweep`` injects automatically before
# overlaying the user's ``cfg.algo_kwargs`` on every algorithm class
# (see fl_v7.py — _run_training_v7 around line 498). The spec must NOT
# specify these; if it does, fl_v7's auto-fill will silently win and
# the spec value is dead code.
_AUTO_FILLED_BY_FL_V7 = frozenset({
    "max_steps", "batch_size", "grad_clip", "amp_enabled", "amp_dtype",
})


def _import_algo_metadata():
    """Pull the static required-kwargs table + algorithm registry.

    Lazy-imported because :mod:`fl_oran.training.fl_v7` triggers the
    full torch import chain. We pay the cost once per
    :func:`validate_spec` call rather than at module load — this keeps
    ``import _v7_spec_loader`` cheap for callers that only need
    :func:`load_spec` (e.g. linters or schema browsers).
    """
    from fl_oran.federated.algorithms import get_algorithm
    from fl_oran.training.fl_v7 import _ALGO_REQUIRED_KWARGS
    return _ALGO_REQUIRED_KWARGS, get_algorithm


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


def _normalize_algorithms(raw: list) -> list[tuple[str, dict]]:
    """Coerce each algorithms list entry into a ``(name, kwargs)`` tuple.

    Two surface forms are accepted::

        algorithms: [fedavg, fedprox]                       # back-compat
        algorithms:
          - name: fedavg
          - name: fedprox
            kwargs: {mu: 0.01}

    A bare string ``"fedavg"`` becomes ``("fedavg", {})``. A dict
    ``{"name": "fedprox", "kwargs": {"mu": 0.01}}`` becomes
    ``("fedprox", {"mu": 0.01})``. Missing ``kwargs`` is treated as
    empty dict.

    Structural errors (non-str/non-dict element, dict missing ``name``,
    ``kwargs`` not a dict) raise ValueError immediately. Semantic
    validation (required-kwargs satisfied, no unknowns, no duplicates)
    happens in :func:`validate_spec`.
    """
    out: list[tuple[str, dict]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append((entry, {}))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"algorithm entry dict must have non-empty 'name' "
                    f"string; got {entry!r}"
                )
            kwargs = entry.get("kwargs", {})
            if kwargs is None:
                kwargs = {}
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"algorithm[{name!r}].kwargs must be a dict, got "
                    f"{type(kwargs).__name__}: {kwargs!r}"
                )
            # Reject unexpected top-level keys to catch typos like
            # ``args:`` instead of ``kwargs:`` early.
            extra = set(entry.keys()) - {"name", "kwargs"}
            if extra:
                raise ValueError(
                    f"algorithm[{name!r}] has unknown keys {sorted(extra)}; "
                    f"only 'name' and 'kwargs' are allowed"
                )
            out.append((name, dict(kwargs)))
        else:
            raise ValueError(
                f"algorithm entry must be str or dict, got "
                f"{type(entry).__name__}: {entry!r}"
            )
    return out


def _validate_algo_kwargs(name: str, kwargs: dict, required_table, get_algo_fn) -> None:
    """Check required kwargs are present and no unknowns slip through.

    The ``required`` set comes from the static
    ``fl_v7._ALGO_REQUIRED_KWARGS`` table — the table only lists kwargs
    a *user* must supply (kwargs that fl_v7 auto-fills like ``max_steps``
    are excluded). The "unknown" check uses ``inspect.signature`` on
    the algorithm class so that typos like ``mue`` instead of ``mu``
    fail at validation time rather than silently degrading to default
    behavior at instantiation.
    """
    if name not in required_table:
        # Algorithm is in the registry but missing from the required-
        # kwargs table — caller should not have reached here, but
        # guard anyway. fl_v7's regression test prevents this drift.
        raise ValueError(
            f"algorithm {name!r} is in the registry but missing from "
            f"_ALGO_REQUIRED_KWARGS — update the table"
        )
    required = required_table[name]
    missing = required - set(kwargs)
    if missing:
        raise ValueError(
            f"algorithm {name!r} missing required kwargs: {sorted(missing)}; "
            f"got kwargs={sorted(kwargs)}"
        )
    cls = get_algo_fn(name)
    sig_params = set(inspect.signature(cls.__init__).parameters)
    allowed = sig_params - {"self"} - _AUTO_FILLED_BY_FL_V7
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(
            f"algorithm {name!r} has unknown kwargs: {sorted(unknown)}; "
            f"allowed (excluding fl_v7-auto-filled "
            f"{sorted(_AUTO_FILLED_BY_FL_V7)}): {sorted(allowed)}"
        )


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

    algos_raw = spec["algorithms"]
    if not isinstance(algos_raw, list) or not algos_raw:
        raise ValueError("spec['algorithms'] must be a non-empty list")
    # Normalize first (catches structural shape errors), then validate
    # against algorithm registry + required-kwargs table.
    normalized = _normalize_algorithms(algos_raw)
    names = [n for n, _ in normalized]
    if len(set(names)) != len(names):
        from collections import Counter
        dups = sorted({n for n, k in Counter(names).items() if k > 1})
        raise ValueError(f"duplicate algorithms in spec: {dups}")
    known_algos = helper.known_algorithms()
    required_table, get_algo_fn = _import_algo_metadata()
    for name, kwargs in normalized:
        if name not in known_algos:
            raise ValueError(
                f"unknown algorithm {name!r}; known: {sorted(known_algos)}"
            )
        if name == "moon":
            raise ValueError(
                "MOON is deferred in Phase 1.5 / Phase 2 minimum per "
                "ADR D-22 — fl_v7._select_algorithm raises "
                "NotImplementedError. Remove 'moon' from spec or "
                "revisit D-22 first."
            )
        _validate_algo_kwargs(name, kwargs, required_table, get_algo_fn)

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
      3. ``algorithm`` (name) and ``algo_kwargs`` (per-algorithm dict
         from the normalized form — empty for fedavg/scaffold,
         ``{"mu": 0.01}`` for fedprox, etc.)
      4. ``arch``
      5. ``arch_overrides[arch]`` (wins over shared — typically just
         ``lr`` per ADR D-20)
      6. ``seed`` and ``name`` (canonical)

    Every cell gets an ``algo_kwargs`` key (possibly empty dict) so
    downstream ``V7Config(**cell)`` construction is uniform.
    """
    validate_spec(spec)
    helper = _metadata()
    overrides = spec.get("arch_overrides", {}) or {}
    normalized_algos = _normalize_algorithms(spec["algorithms"])
    cells: list[dict] = []
    for arch in spec["archs"]:
        for algo_name, algo_kwargs in normalized_algos:
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
                    cell["algorithm"] = algo_name
                    # Copy so cells don't share a mutable algo_kwargs dict
                    # (callers occasionally mutate a returned cell).
                    cell["algo_kwargs"] = dict(algo_kwargs)
                    arch_over = overrides.get(arch, {})
                    if isinstance(arch_over, dict):
                        cell.update(arch_over)
                    cell["seed"] = seed
                    cell["name"] = helper.cell_name(
                        arch, algo_name, partition["mode"], seed,
                        partition["n_clients"], alpha=cell["alpha"],
                    )
                    cells.append(cell)
    return cells
