"""Shared metadata helpers for v6 cell directories.

Both ``measure_v6_gpu_energy.py`` and ``recompute_v6_energy.py`` need
to reconstruct the exact ``(arch_ctor, kwargs)`` a cell was trained
under, given only the cell directory name (e.g.
``spiking_s42_t5sum_50k``). Historically each script carried its own
copy of this logic and the two drifted apart — recompute lacked
support for ``decode_mode=sum`` and the ``_expand2`` ablations, which
silently corrupted ``energy.json`` for affected cells.

This module is the **single source of truth**. It is intentionally
script-local (under ``scripts/``) rather than promoted to ``src/`` so
its private helpers (``_t5sum``, ``_lif_t05_b09`` recipe parsing) do
not leak into the public ``fl_oran`` package surface.

Public API:

* :data:`KNOWN_ARCHES` — set of arch keys understood by the runner.
* :func:`runner_arch_registry` — returns ``ARCH_REGISTRY`` from the
  runner module, cached after first load.
* :func:`parse_cell_dir(name)` — ``(arch_base, seed, suffix)`` from
  e.g. ``spiking_expand2_s42_t5sum_50k``.
* :func:`build_kwargs_from_suffix(arch_base, suffix)` — kwargs needed
  to ``ctor(schema=..., task="classification", seq_len=..., **kw)``
  matching what the runner used.
* :func:`atomic_write_text(path, content)` — write-through-tempfile +
  ``os.replace`` for crash-safe ``energy.json`` / ``energy_measured.json``
  writes.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_v6_arch_sweep.py"
_RUNNER_CACHE: dict | None = None


def runner_arch_registry() -> dict:
    """Load and cache ARCH_REGISTRY from experiments/run_v6_arch_sweep.py.

    Uses ``importlib.util.spec_from_file_location`` so we never mutate
    ``sys.path`` and never collide with other modules. Cached at module
    scope so a 150-cell measurement loop only pays the import cost once.
    """
    global _RUNNER_CACHE
    if _RUNNER_CACHE is None:
        spec = importlib.util.spec_from_file_location("_v6_runner", _RUNNER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load runner module from {_RUNNER_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RUNNER_CACHE = mod.ARCH_REGISTRY
    return _RUNNER_CACHE


def known_arches() -> set[str]:
    """All arch keys understood by the runner (computed lazily)."""
    return set(runner_arch_registry().keys())


def parse_cell_dir(name: str) -> tuple[str, int, str]:
    """Returns ``(arch_base, seed, suffix)``. ``suffix`` may be empty.

    The cell-name convention is ``<arch>_s<seed>[_<suffix>]``. The
    parser splits on the **first** ``_s`` substring, so any future arch
    name that itself contains ``_s`` (e.g. ``spiking_skip``) would
    misparse. We validate the registry against this constraint as a
    defensive check that fires loudly on misuse rather than silently
    corrupting the seed field.
    """
    for known_arch in known_arches():
        if "_s" in known_arch:
            raise ValueError(
                f"runner ARCH_REGISTRY contains arch {known_arch!r} with '_s' "
                f"substring; parse_cell_dir would misparse cell names with "
                f"that arch. Rename the arch or refactor this parser."
            )
    arch, sep, rest = name.partition("_s")
    if sep == "" or not rest:
        raise ValueError(f"unexpected cell dir name: {name!r}")
    seed_part, _, suffix = rest.partition("_")
    return arch, int(seed_part), suffix


def build_kwargs_from_suffix(arch_base: str, suffix: str) -> dict:
    """Reconstruct the constructor kwargs the cell was trained with.

    Order of precedence:

    1. Registry defaults for ``arch_base`` — e.g. for ``spiking_expand2``
       this returns ``{backbone_d_model: 56, backbone_expand: 2,
       t_inner: 1}``.
    2. Suffix-encoded recovery overrides for spiking variants:
       * ``_t5sum`` → ``t_inner=5, decode_mode="sum"``
       * ``_t5`` (without sum) → ``t_inner=5``
       * ``_lif_tNN_bMM`` → ``lif_threshold=NN/10, lif_beta=MM/10``
       * ``_lif_tNN_bMMM`` → ``lif_threshold=NN/10, lif_beta=MMM/100``

    The training-budget suffixes (``_25k``, ``_50k``, ``_100k``,
    ``_lr5e4``) are intentionally **not** parsed — they affect optimiser
    state, not model architecture, so they leave kwargs unchanged.
    """
    registry = runner_arch_registry()
    if arch_base not in registry:
        return {}
    kwargs = dict(registry[arch_base].get("kwargs", {}))

    if arch_base in ("spiking", "spiking_expand2"):
        # Order matters: check t5sum BEFORE t5 because "t5sum" contains "t5"
        # as a substring.
        if "t5sum" in suffix:
            kwargs["t_inner"] = 5
            kwargs["decode_mode"] = "sum"
        elif "t5" in suffix:
            kwargs["t_inner"] = 5

        if "lif_t" in suffix:
            try:
                t_str = suffix.split("lif_t")[1].split("_")[0]
                kwargs["lif_threshold"] = int(t_str) / 10.0
            except (IndexError, ValueError):
                pass
            try:
                b_str = suffix.split("_b")[-1].split("_")[0]
                if len(b_str) == 2:
                    kwargs["lif_beta"] = int(b_str) / 10.0
                elif len(b_str) == 3:
                    kwargs["lif_beta"] = int(b_str) / 100.0
            except (IndexError, ValueError):
                pass
    return kwargs


def atomic_write_text(path: Path, content: str) -> None:
    """Crash-safe text write: tempfile + ``os.replace``.

    ``Path.write_text`` opens the destination, writes through it, and
    closes — there is a window during which the file is partially
    written and any reader (the aggregator on a subsequent run) will
    see a truncated JSON and crash. Atomic write avoids that by writing
    to a sibling tempfile then renaming, which on POSIX is atomic w.r.t.
    readers of the destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
