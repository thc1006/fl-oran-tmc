# fl-oran-tmc

Local PyTorch federated-learning pipeline for ColO-RAN O-RAN slice SLA
forecasting. Built as the experimental substrate for an IEEE JSAC submission
on multi-algorithm FL under realistic non-IID splits.
(Venue switched from IEEE TMC to IEEE JSAC on 2026-05-05 — see
`docs/ADR-001-v5-tmc-paper-plan.md` Revision History. The `tmc` in this
repo's name is preserved for historical continuity with v1-v4 artifacts.)

> **Status: private during review.** The repository is kept private until
> camera-ready. The submission-ready paper (v0.9.2) is archived on Zenodo:
> **DOI: [10.5281/zenodo.20075433](https://doi.org/10.5281/zenodo.20075433)**
> — this is the preprint / submission deposit; an accepted-version DOI
> will be minted after JSAC acceptance.

## What this is

A clean-history reboot of the v1–v4 exploratory codebase (`colosseum-oran-federated-slicing`), scoped down to what the JSAC paper needs (paper title: *Federated O-RAN Slice SLA Prediction: A Cross-Architecture Empirical Benchmark on Colosseum/ColO-RAN*):

- **10 FL algorithms** in a unified registry: `FedAvg`, `FedProx`, `FedAdam`,
  `SCAFFOLD`, `FedDyn`, `FedBN`, `FedSWA`, `FedSCAM`, `FedGMT`, `FedMoSWA`
  (MOON deferred per ADR-001 D-16; raises `NotImplementedError` at dispatch).
- **5 sequence architectures**: LSTM (`ForecasterV2`), Mamba-S6, Spiking-SSM
  for the 3-arch core panel (Phase 5, 900 cells); xLSTM-sLSTM (Beck et al.,
  NeurIPS 2024) and Mamba-3 (Lahoti et al., arXiv 2603.15569) added in the
  Path D extended sweep (360 additional cells, 2026-05-18 prep). All five
  share an identical encoder and a structurally-identical classifier head
  (`Linear → ReLU → Linear`); only the temporal trunk differs. Total
  parameter count is matched within ±10% across all 5 archs (ADR-001 D-20
  parity constraint); xLSTM's head shell happens to consume `hidden_size=48`
  directly (no Mamba-style `out_proj` bottleneck), while the 3 core archs
  and Mamba-3 narrow to 32 before the head — this is the architecture-level
  confounder that the param-count parity rule controls for.
- **6 client partitions**: natural-by-BS (7 ColO-RAN gNBs as clients),
  Dirichlet `α ∈ {0.05, 0.10, 0.50, 1.00, 5.00}`, plus controlled
  `random_split` and `per_bs_dirichlet` ablation modes.
- **OOD split** by `training_config` id (tr0-21 train / tr22-24 val /
  tr25-27 test) to prevent the target-leakage that invalidated v1.
- **NVML per-round training-energy** measurement on commodity GPU,
  idle-baseline subtracted.
- **PyTorch-native**, single-machine. Stage 1 + Stage 2 core sweep
  collected on RTX 4080 (sm_89); current dev workstation is RTX 4060 Ti
  (sm_89, 16 GiB VRAM) post 2026-05-16 migration. Path D core sweep
  runs on a separate 4× Tesla V100-SXM2-32GB cluster. Python 3.14 +
  PyTorch 2.11 + CUDA 12.8 (4060) / CUDA 12.1 (V100). **No Flower, no TFF.**

The v1–v4 code is preserved under `src/fl_oran/` untouched; v5–v7
extensions live alongside it. See `docs/ADR-001-v5-tmc-paper-plan.md`
for the full design record (filename retains the historical
`tmc-paper-plan` slug — venue switched to JSAC on 2026-05-05; see
ADR-001 Revision History).

## Current state (as of 2026-05-18)

| Stage | Scope | Status |
|-----------|-------|--------|
| M1–M5 (v5) | Algorithm registry + 150-cell FL benchmark | done (see `docs/RESULTS_V5_FINAL.md`) |
| Stage 1 (v6) | 3-arch centralized sweep (3 archs × 10 seeds = 30 cells) | done (see `docs/RESULTS_V6_STAGE1.md`) |
| Stage 2 / Phase 5 (v7) | 3-arch × 5-algo × 6-partition × 10-seed FL sweep (900 cells) | done (paper Figs 1–3 + `docs/RESULTS_V7_PHASE5.md`) |
| Phase 6 (v7) | Per-BS Dirichlet ablation + R2 reviewer feedback | done (paper Tables 4 & 6 + §7.1.5) |
| Path D core (v7) | 3-arch × 3 SAM-family-algo × 6-partition × 10-seed (540 cells) | in progress on V100 cluster |
| Path D extension prep | 5-arch spec + pilot launcher + paper draft (xLSTM + Mamba-3) | done (PRs #18–#23) |
| Path D extended sweep | 2-new-arch × 3-algo × 6-partition × 10-seed (360 cells) | pending V100 release |

**334+ tests passing** across v1–v7 (89 legacy v1–v4 + 42 v5 FL + 90 v6 arch + 162+ v7 FL × arch + R1/R2 audit invariants + paper-claim sources + 5 new pin tests for xLSTM/Mamba-3). All test infrastructure is in `tests/`; CI uses `pytest --no-cov` for the canonical run.

**Paper artifact**: `paper/main.tex` (21 pp) + `paper/supplementary.tex`
(3 pp) + 67-entry `paper/bibliography.bib`. Submission-ready PDF
deposited to Zenodo at **DOI [10.5281/zenodo.20075433](https://doi.org/10.5281/zenodo.20075433)**
under tag `v0.9.2-submission-ready`.

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

**Training is fully wired up** as of v6/v7 (commits `c63b...` onwards).
The Stage 2 / Phase 5 entry point is `experiments/run_v7_phase_sweep.py
--spec experiments/specs/<spec>.yaml --skip-completed`. See
`experiments/specs/` for the available sweep specifications (`stage2_full.yaml`,
`path_d_full.yaml`, `path_d_extended_pilot.yaml`, etc.). The v3/v4
legacy entry points (`run_v3_centralized.py`, `run_v4_all_seeds.py`)
are kept as-is for baseline reproducibility.

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
src/fl_oran/               Main package (v1–v4 preserved, v5–v7 additive)
├── data_v2/                Dirichlet partition, OOD split, target builder v2
├── federated/
│   ├── aggregation.py      Shared FedAvg weighted-average primitive
│   ├── client.py           ClientUpdate dataclass
│   └── algorithms/         FLAlgorithm Protocol + 10 algorithm classes
├── models/                 ForecasterV2 (LSTM) + MambaForecaster +
│                           SpikingForecaster + xLSTMForecaster (NEW) +
│                           Mamba3Forecaster (NEW)
├── training/               v3/v4 trainers (frozen) + v6 centralized +
│                           v7 federated
└── utils/                  seed, device, AMP helpers

tests/                      334+ tests across v1–v7; conftest.py has shared fixtures
docs/
├── ADR-001-v5-tmc-paper-plan.md   Plan + decision log (16+ decisions, history)
├── ADR-002-phase6-fedswa.md       Mechanism-based FedSWA rejection
├── PAPER_NOTES_{XLSTM,MAMBA3}.md  Per-arch design rationale
├── PAPER_5ARCH_EXTENSION_DRAFT.md Paste-ready §3+§4+§7 5-arch text
└── RESULTS_*.md                   Aggregated paper-grade tables (committed)
experiments/
├── run_v3_*.py / run_v4_all_seeds.py    v3/v4 baselines (frozen)
├── run_v5_*.py                          v5 FL algorithm sweeps (frozen, M5)
├── run_v6_arch_sweep.py                 Stage 1 centralized 3-arch sweep
├── run_v7_phase_sweep.py                Stage 2 / Path D spec-driven launcher
└── specs/                               sweep specifications (.yaml)
artifacts/
├── RESULTS_V4.md                        v3/v4 baseline numbers (committed)
└── (all run outputs gitignored)         see `.gitignore` for the full pattern
paper/
├── main.tex / supplementary.tex         JSAC submission source
├── bibliography.bib                     67 BibTeX entries
└── main.pdf                             v0.9.2-submission-ready (Zenodo)
scripts/
├── aggregate_v7_results.py              Phase 5 / Stage 2 results aggregator
│                                        (generates docs/RESULTS_V7_PHASE5.md
│                                        — sourced by paper Figs 1–3)
├── aggregate_path_d.py                  Path D paper §7 generator
│                                        (paired-bootstrap CI95 tables)
├── sweep_dashboard.py                   Live Path D sweep dashboard
└── v100_*_launcher.sh                   V100 cluster launchers
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
