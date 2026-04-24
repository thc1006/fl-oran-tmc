# fl_oran — Project context for Claude

Local PyTorch federated-learning pipeline for ColO-RAN O-RAN slice SLA prediction.
Current state: v4 pipeline complete (86 tests pass, multi-seed results in `artifacts/RESULTS_V4.md`).
Next phase: v5 (IEEE TMC paper) — see `docs/ADR-001-v5-tmc-paper-plan.md`.

## Build / test commands

```bash
source .venv/bin/activate                 # always activate first
pytest                                    # all tests (~4 s, currently 89 passing)
pytest tests/test_v5_*.py                 # v5-only tests (not yet written)
python experiments/run_v3_centralized.py  # centralized baseline
python experiments/run_v4_all_seeds.py    # v4 multi-seed (done, don't re-run)
```

Hardware: single RTX 4080, 16 GiB VRAM, Ubuntu 24.04, Python 3.12.3, PyTorch 2.10 + CUDA 12.8.
Data: `data/coloran_raw_unified.parquet` (18M rows, symlinked from `raw/colosseum-oran-coloran-dataset-master/`).

## Naming conventions (existing, keep consistent)

| Layer | Pattern | Example |
|-------|---------|---------|
| Client trainer | `train_one_client_*` | `train_one_client_capped`, `train_one_client_cuda_graph` |
| Experiment runner | `run_<variant>` | `run_centralized`, `run_federated` |
| Data fit/apply | `fit_<thing>`, `apply_<thing>` | `fit_continuous_scaler`, `apply_continuous_scaler` |
| Target builder | `add_<task>_target` | `add_classification_target`, `add_regression_target` |
| Model | `<Name>V<n>` | `ForecasterV2`, `MLPv107_2` |

When adding new code, match these patterns. Do **not** invent new naming styles.

## Hard rules (do not violate)

1. **Do not redefine functions that exist elsewhere.** Single-source list is in ADR-001 D-3. Import, don't reimplement.
2. **Do not refactor v1–v4 code** during v5 work unless a v5 bug requires it. "Could be cleaner" is not a reason. Real bugs with failing tests are.
3. **Do not use fancy syntax for its own sake.** No `match/case`, no PEP 695 generics, no `@override`, no `type` aliases, no walrus outside hot loops. See ADR-001 D-8.
4. **Do not handle bugs by suppressing them.** No `try/except: pass`, no `nan_to_num` to hide NaN sources, no `filterwarnings("ignore")`, no `--no-verify`. Follow the 7-step debug protocol (ADR-001 D-10).
5. **Do not create new CLAUDE.md / ADR files for minor updates.** Update existing ones in place with a Revision History entry.
6. **Do not run training without asking.** Plan and code only unless explicitly told to execute.
7. **Do not use emojis in source code or docs** unless the user explicitly asks.

## Debug protocol (short version; full in ADR-001 D-10)

1. Read the full stack trace, identify file/line.
2. Reproduce deterministically (pin seed, batch, data).
3. Bisect: remove features until minimal failure remains.
4. Hypothesise in one sentence **before** editing.
5. Fix root cause, not symptom.
6. Add a regression test.

## File layout (authoritative)

```
src/fl_oran/
├── data/, data_v2/, data_raw/   # loaders, scalers, sequences, partitions
├── models/                      # MLPv106, MLPv107_2, LSTMMultiOutput, ForecasterV2
├── federated/                   # aggregation, client(s), dp, algorithms/ (v5, new)
├── training/                    # trainer (v1-v4), centralized_v3, fl_v3
├── baselines/                   # persistence, gbm
├── evaluation/, utils/          # metrics, seed, gpu helpers
├── cli.py, logging_utils.py, config.py
experiments/
├── run_v106.py, run_v107_1.py, run_v107_2.py  (v1 faithful replay)
├── run_v3_centralized.py, run_v3_fl_iid.py, run_v3_fl_noniid.py
├── run_v4_all_seeds.py  (multi-seed)
└── run_v5_algorithm_sweep.py  (to be written — see ADR-001 M3)
tests/  (89 passing; add v5 tests per ADR-001 §3)
docs/ADR-001-v5-tmc-paper-plan.md  ← read before starting v5 work
artifacts/RESULTS_V4.md  ← current benchmark state
```

## Current v5 workstream (as of 2026-04-25)

Read `docs/ADR-001-v5-tmc-paper-plan.md` for full plan. Milestones M1–M4 over ~6 weeks.
Next concrete step: write `tests/test_v5_dirichlet_partition.py` (TDD red), then extend
`src/fl_oran/data_v2/partition.py` to handle `mode="dirichlet"`.

## What's in `artifacts/`

Do not delete. Contains all v1–v4 model checkpoints, logs, history CSVs, summary JSONs,
and `RESULTS_V4.md`. v5 outputs go under `artifacts/v5_sweep/` (new subdirectory).

## How to ask for help

Prefer specific questions over broad ones. Good: *"v5 Dirichlet partition should use
`numpy.random.Generator.dirichlet` or `torch.distributions.Dirichlet`? Which matches
our existing partition.py style?"* Bad: *"how should I do non-IID?"*
