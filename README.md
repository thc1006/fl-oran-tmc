# fl-oran-tmc

Local PyTorch federated-learning pipeline for ColO-RAN O-RAN slice SLA
forecasting. Built as the experimental substrate for an IEEE TMC submission
on multi-algorithm FL under realistic non-IID splits.

> **Status: private during review.** The repository is kept private until
> camera-ready. Badges, release tags, and DOI will be added at submission
> acceptance.

## What this is

A clean-history reboot of the v1–v4 exploratory codebase (`colosseum-oran-federated-slicing`), scoped down to what the TMC paper needs:

- **6 FL algorithms** in a unified registry: `FedAvg`, `FedProx`, `FedAdam`,
  `SCAFFOLD`, `FedDyn`, `MOON`.
- **Dirichlet non-IID partition** over the `slice_id` column (NIID-Bench
  convention; alpha sweep planned).
- **OOD split** by `training_config` id (tr0-21 train / tr22-24 val /
  tr25-27 test) to prevent the target-leakage that invalidated v1.
- **PyTorch-native**, single-machine (RTX 4080, 16 GiB VRAM, Python 3.12,
  PyTorch 2.10 + CUDA 12.8). **No Flower, no TFF.**

The v1–v4 code is preserved under `src/fl_oran/` untouched; v5 extensions
live alongside it. See `docs/ADR-001-v5-tmc-paper-plan.md` for the full
design record.

## Current milestone state

| Milestone | Scope | Status |
|-----------|-------|--------|
| M1 | Dirichlet partition, `FLAlgorithm` registry, FedAvg, FedProx | done |
| M2 | FedAdam, SCAFFOLD, FedDyn + `run_local_sgd` helper | done |
| M3a | MOON with caller-supplied `encode_fn` | done |
| M3b | Sweep orchestrator + ForecasterV2 encode_fn + pilot | in progress |
| M4 | Multi-seed × alpha × algorithm sweep + aggregation + tables | pending |

**124 tests passing.** No v1–v4 regression (89/89 legacy tests preserved).

## Algorithm registry

```python
from fl_oran.federated.algorithms import get_algorithm

algo = get_algorithm("fedprox")(max_steps=50, batch_size=64, mu=0.01)
# Algorithms share this contract:
#   algo.client_update(client_id, local_model, client_tensors, loss_fn,
#                      current_lr, device, round_idx) -> ClientUpdate
#   algo.server_aggregate(global_state, updates) -> dict[str, Tensor]
# Stateful algorithms (SCAFFOLD, FedDyn, FedAdam, MOON) carry their own
# persistent state on the instance.
```

See `src/fl_oran/federated/algorithms/__init__.py` for the `FLAlgorithm`
Protocol and the rationale for which kwargs were intentionally omitted.

## Quick start (development)

```bash
# The venv is shared with the upstream colosseum-oran-federated-slicing
# repo via a symlink so PyTorch isn't installed twice.
source .venv/bin/activate

# Run the test suite (~4 s; no training, no GPU required):
pytest

# Run v5-only tests:
pytest tests/test_v5_*.py

# Coverage report (html in artifacts/coverage/):
pytest --cov=src/fl_oran
```

### Paper-claim source tests

`tests/test_paper_claims_sources.py` validates each numerical claim in
`docs/PAPER_DRAFT.md` against the underlying ground-truth artifacts.
The artifacts are gitignored (computed locally, not in CI); on a fresh
checkout these tests will SKIP with an informative regenerate hint.

```bash
# Regenerate paper artifacts before running the claim-source tests:
python scripts/step1_fact_finding.py     # → artifacts/step1_factfinding.json
python scripts/step2_mechanism_search.py # → artifacts/step2_mechanism_search.json

# Then the full suite runs (claim-source tests no longer skip):
pytest
```

**Training runs are not yet wired up** — the M3b orchestrator is in
progress. `experiments/run_v3_centralized.py`, `run_v3_fl_iid.py`,
`run_v3_fl_noniid.py`, `run_v4_all_seeds.py` replay the v3/v4 baselines
and will be kept as-is for TMC baseline comparisons.

## Data

ColO-RAN raw dataset (8.9 GB) is symlinked as `raw/`; the unified
processed parquet (18M rows) is symlinked as
`data/coloran_raw_unified.parquet`. Neither is tracked by git. The
symlinks resolve to the upstream `colosseum-oran-federated-slicing` repo
— see `.gitignore` for the full exclusion list.

Dataset citation (and the reason the target was reformulated in v2 after a
leakage audit):

> M. Polese *et al.*, "ColO-RAN: Developing Machine Learning-based xApps
> for Open RAN Closed-loop Control," **IEEE TMC**, 2022.
> <https://github.com/wineslab/colosseum-oran-coloran-dataset>

## Repository layout

```
src/fl_oran/               Main package (v1–v4 preserved, v5 additive)
├── data_v2/                Dirichlet partition, OOD split, target builder v2
├── federated/
│   ├── aggregation.py      Shared FedAvg weighted-average primitive
│   ├── client.py           ClientUpdate dataclass
│   └── algorithms/         FLAlgorithm Protocol + 6 algorithm classes
├── models/                 ForecasterV2 (embeddings + LSTM)
├── training/               v3/v4 trainers (do not modify)
└── utils/                  seed, device, AMP helpers

tests/                     124 tests; conftest.py has shared fixtures
docs/
├── ADR-001-v5-tmc-paper-plan.md   Full v5 plan + 16 decisions + history
└── README.md                      ADR index
experiments/
├── run_v3_*.py             v3 baselines (kept for comparison)
└── run_v4_all_seeds.py     v4 multi-seed baseline
artifacts/
├── RESULTS_V3.md           v3 centralized/IID/non-IID numbers
└── RESULTS_V4.md           v4 multi-seed numbers (baseline of record)
```

New contributors (and Claude) should read `CLAUDE.md` at the root first
— it captures the naming conventions and the seven hard rules that keep
v5 work from regressing v1–v4.

## License

Released under **AGPL-3.0** — see `LICENSE`. The ColO-RAN dataset is
distributed by WINES Lab under AGPL-3.0; derivative work on the dataset
inherits the license obligation.

## Acknowledgements

- The ColO-RAN dataset and the TMC 2022 paper that introduced it (Polese
  et al.) are the substrate this work builds on.
- The v5 algorithm set was refined against NIID-Bench, ERFO-2025, and
  the Reddi/Acar/Karimireddy/Li papers — full citations in
  `docs/ADR-001-v5-tmc-paper-plan.md`.
