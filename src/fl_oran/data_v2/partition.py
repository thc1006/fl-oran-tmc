"""Client data partitioning: IID (by bs_id) vs Non-IID (slice-restricted).

The Non-IID mode is what makes FL scientifically meaningful on this dataset.
In IID mode, all 7 BS see the same marginal distribution → FL ≈ centralized
with extra overhead. In Non-IID mode, each client is restricted to certain
slices, forcing the FL aggregator to learn a model that generalises across
clients each of which has a biased view.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


def partition_clients(
    df: pd.DataFrame,
    *,
    mode: Literal["iid", "noniid_slice", "dirichlet", "random_split",
                  "per_bs_dirichlet", "run_random", "run_dirichlet"],
    client_slice_map: dict[int, list[int]] | None = None,
    alpha: float | None = None,
    n_clients: int | None = None,
    seed: int | None = None,
    sub_per_bs: int | None = None,
) -> dict[int, pd.DataFrame]:
    """Return {client_id: DataFrame shard}.

    ``iid``: each client = one bs_id, sees all slices from its cell.
    ``noniid_slice``: each client = one bs_id, sees only the slices in
    ``client_slice_map[cid]`` from its cell.
    ``dirichlet``: for each slice_id, sample a ``Dirichlet(alpha)`` distribution
    over ``n_clients`` clients and distribute that slice's rows accordingly.
    Small alpha → highly concentrated (few clients get most of a slice); large
    alpha → near-uniform across clients. Requires ``alpha`` and ``n_clients``.
    ``random_split``: shuffle all rows and split into ``n_clients`` ~equal-size
    shards, ignoring every column. Each client sees the global marginal
    (true IID at first order). Used for the §7 mechanism ablation that asks
    "does natural-by-BS dominance come from bs_id structure?" — by stripping
    away both bs_id and slice_id grouping while keeping per-client sample-size
    balanced, this isolates the structural contribution. Requires ``n_clients``.
    ``run_random``: assign each WHOLE ``(run_id, slice_id)`` group to one client
    (groups never split across clients), using greedy least-loaded-by-rows
    assignment over a seed-shuffled group order. This is the sequence-integrity
    control (PREREG-A1 follow-up): unlike ``random_split`` (row-level shuffle,
    which scatters a run's timesteps across clients so ``build_run_sequences``
    builds temporally-broken windows), ``run_random`` preserves each run's
    contiguity → valid windows — while still breaking bs_id coherence (each
    client gets a random mix of BS). It isolates "intact sequences" from
    "BS-coherent partition." Requires ``n_clients``.
    ``run_dirichlet``: the **run-level analog of** ``dirichlet`` — per slice,
    distribute that slice's WHOLE ``(run_id)`` groups across clients by
    ``Dir(alpha)`` (each ``(run_id, slice_id)`` group stays in one client, so
    runs are intact → valid windows), with the SAME per-slice Dirichlet skew as
    the row-level version. Comparing ``run_dirichlet(alpha)`` vs
    ``dirichlet(alpha)`` at matched ``alpha`` isolates the sequence-corruption
    axis from the heterogeneity axis (the decisive test of whether the
    inverted-heterogeneity finding is a row-partitioning artifact). Requires
    ``alpha`` and ``n_clients``.

    Assumes ``df`` has a unique index (the default integer RangeIndex or any
    ``reset_index(drop=True)`` output). Callers whose pipeline may produce
    duplicate indices (e.g. after ``pd.concat``) should ``reset_index`` first,
    otherwise label-based row retrieval in the dirichlet branch could return
    extra rows.
    """
    if mode == "iid":
        shards = {int(cid): g.reset_index(drop=True)
                  for cid, g in df.groupby("bs_id", observed=True)}
        log.info("IID partition: %d clients, rows=%s",
                 len(shards), {c: len(s) for c, s in shards.items()})
        return shards

    if mode == "noniid_slice":
        if not client_slice_map:
            raise ValueError("noniid_slice requires client_slice_map")
        shards: dict[int, pd.DataFrame] = {}
        for cid, allowed_slices in client_slice_map.items():
            mask = (df["bs_id"] == cid) & (df["slice_id"].isin(allowed_slices))
            sub = df[mask].reset_index(drop=True)
            if len(sub) > 0:
                shards[int(cid)] = sub
        log.info("Non-IID slice partition: %d clients assigned; mapping=%s",
                 len(shards), {c: list(set(s["slice_id"].unique().tolist()))
                               for c, s in shards.items()})
        return shards

    if mode == "dirichlet":
        if alpha is None:
            raise ValueError("dirichlet mode requires alpha (Dirichlet concentration)")
        if n_clients is None:
            raise ValueError("dirichlet mode requires n_clients")
        rng = np.random.default_rng(seed)
        # Client row-index buckets; each will become a DataFrame shard.
        client_indices: dict[int, list[int]] = {cid: [] for cid in range(1, n_clients + 1)}
        # For each slice, draw proportions and partition that slice's rows.
        for slice_id, g in df.groupby("slice_id", observed=True):
            proportions = rng.dirichlet([alpha] * n_clients)
            slice_idx = g.index.to_numpy(copy=True)
            rng.shuffle(slice_idx)
            # Cumulative split points over the shuffled indices.
            splits = (np.cumsum(proportions) * len(slice_idx)).astype(int)
            # np.split needs interior split points only.
            parts = np.split(slice_idx, splits[:-1])
            for cid, part in zip(range(1, n_clients + 1), parts):
                if len(part) > 0:
                    client_indices[cid].extend(part.tolist())
        shards: dict[int, pd.DataFrame] = {}
        for cid, idx_list in client_indices.items():
            if not idx_list:
                continue
            shards[cid] = df.loc[idx_list].reset_index(drop=True)
        log.info(
            "Dirichlet partition (alpha=%.3f, n_clients=%d, seed=%s): %d non-empty "
            "shards; rows=%s",
            alpha, n_clients, seed, len(shards),
            {c: len(s) for c, s in shards.items()},
        )
        return shards

    if mode == "random_split":
        if n_clients is None:
            raise ValueError("random_split mode requires n_clients")
        rng = np.random.default_rng(seed)
        # Shuffle row positions (not labels — works regardless of index).
        n_rows = len(df)
        positions = rng.permutation(n_rows)
        # np.array_split balances within ±1 row when n_rows is not divisible.
        chunks = np.array_split(positions, n_clients)
        shards: dict[int, pd.DataFrame] = {}
        for cid, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            shards[cid] = df.iloc[chunk].reset_index(drop=True)
        log.info(
            "random_split partition (n_clients=%d, seed=%s): %d shards; rows=%s",
            n_clients, seed, len(shards),
            {c: len(s) for c, s in shards.items()},
        )
        return shards

    if mode == "run_random":
        if n_clients is None:
            raise ValueError("run_random mode requires n_clients")
        rng = np.random.default_rng(seed)
        # Whole-(run_id, slice_id)-group assignment: a group is the unit, never
        # split across clients, so build_run_sequences sees contiguous runs.
        # ``.indices`` gives POSITIONAL row arrays (index-label agnostic, like
        # the random_split branch's df.iloc usage).
        grp_pos = df.groupby(["run_id", "slice_id"], observed=True, sort=False).indices
        keys = list(grp_pos.keys())
        perm = rng.permutation(len(keys))
        # Greedy least-loaded-by-rows so shard sizes stay balanced despite
        # variable group sizes.
        client_rows = [0] * n_clients
        client_pos: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for ki in perm:
            pos = grp_pos[keys[int(ki)]]
            cid = min(range(n_clients), key=lambda c: (client_rows[c], c))
            client_pos[cid].append(pos)
            client_rows[cid] += len(pos)
        shards: dict[int, pd.DataFrame] = {}
        for cid in range(n_clients):
            if client_pos[cid]:
                allpos = np.concatenate(client_pos[cid])
                shards[cid] = df.iloc[allpos].reset_index(drop=True)
        log.info(
            "run_random partition (n_clients=%d, seed=%s): %d shards from %d "
            "(run_id,slice_id) groups; rows=%s",
            n_clients, seed, len(shards), len(keys),
            {c: len(s) for c, s in shards.items()},
        )
        return shards

    if mode == "run_dirichlet":
        if alpha is None:
            raise ValueError("run_dirichlet mode requires alpha")
        if n_clients is None:
            raise ValueError("run_dirichlet mode requires n_clients")
        rng = np.random.default_rng(seed)
        # Run-level analog of mode="dirichlet": per slice, distribute that
        # slice's WHOLE (run_id) groups across clients by Dir(alpha). Each
        # (run_id, slice_id) group stays in ONE client (intact runs → valid
        # windows), but the allocation is Dirichlet-skewed exactly like the
        # row-level version. ``.indices`` = POSITIONAL row arrays (iloc-safe).
        grp = df.groupby(["slice_id", "run_id"], observed=True, sort=False).indices
        by_slice: dict = {}
        for (sl, _run), pos in grp.items():
            by_slice.setdefault(sl, []).append(pos)
        client_pos: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for sl in by_slice:
            groups = by_slice[sl]
            order = rng.permutation(len(groups))
            proportions = rng.dirichlet([alpha] * n_clients)
            splits = (np.cumsum(proportions) * len(groups)).astype(int)
            parts = np.split(order, splits[:-1])
            for cid, part in enumerate(parts):
                for gi in part:
                    client_pos[cid].append(groups[int(gi)])
        shards: dict[int, pd.DataFrame] = {}
        for cid in range(n_clients):
            if client_pos[cid]:
                shards[cid] = df.iloc[np.concatenate(client_pos[cid])].reset_index(drop=True)
        log.info(
            "run_dirichlet partition (alpha=%.3f, n_clients=%d, seed=%s): %d "
            "non-empty shards; rows=%s",
            alpha, n_clients, seed, len(shards),
            {c: len(s) for c, s in shards.items()},
        )
        return shards

    if mode == "per_bs_dirichlet":
        # Phase 6 Rank 3 mechanism-disambiguation control:
        # for each bs_id, partition that BS's rows by Dirichlet([alpha])
        # over slice_id into sub_per_bs sub-clients. Total clients =
        # n_bs * sub_per_bs. bs grouping is preserved per client (unlike
        # mode="dirichlet" which scatters bs's across clients).
        if alpha is None:
            raise ValueError("per_bs_dirichlet mode requires alpha")
        if sub_per_bs is None:
            raise ValueError("per_bs_dirichlet mode requires sub_per_bs")
        if sub_per_bs < 1:
            raise ValueError(f"per_bs_dirichlet sub_per_bs must be >= 1, got {sub_per_bs}")
        if sub_per_bs == 1:
            log.warning(
                "per_bs_dirichlet with sub_per_bs=1 degenerates to natural-by-BS "
                "(one shard per BS, alpha is ignored). Did you mean sub_per_bs>=2?"
            )
        rng = np.random.default_rng(seed)
        shards: dict[int, pd.DataFrame] = {}
        next_cid = 0
        # Stable bs_id ordering — sorted ints — so cell IDs are deterministic.
        bs_ids_sorted = sorted(int(b) for b in df["bs_id"].unique())
        for bs in bs_ids_sorted:
            bs_df = df[df["bs_id"] == bs]
            sub_buckets: list[list[int]] = [[] for _ in range(sub_per_bs)]
            for slice_id, sg in bs_df.groupby("slice_id", observed=True):
                proportions = rng.dirichlet([alpha] * sub_per_bs)
                slice_idx = sg.index.to_numpy(copy=True)
                rng.shuffle(slice_idx)
                splits = (np.cumsum(proportions) * len(slice_idx)).astype(int)
                parts = np.split(slice_idx, splits[:-1])
                for sub_i, part in enumerate(parts):
                    if len(part) > 0:
                        sub_buckets[sub_i].extend(part.tolist())
            for sub_i, idx_list in enumerate(sub_buckets):
                if not idx_list:
                    continue
                shards[next_cid] = df.loc[idx_list].reset_index(drop=True)
                next_cid += 1
        log.info(
            "per_bs_dirichlet partition (alpha=%.3f, sub_per_bs=%d, seed=%s): "
            "%d shards (= %d bs × %d sub); rows=%s",
            alpha, sub_per_bs, seed, len(shards),
            len(bs_ids_sorted), sub_per_bs,
            {c: len(s) for c, s in shards.items()},
        )
        return shards

    raise ValueError(f"unknown partition mode: {mode}")
