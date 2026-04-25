# fl_oran — Project context for Claude

Local PyTorch pipeline for ColO-RAN O-RAN slice SLA forecasting. Started as FL benchmark; pivoted 2026-04-25 to two-stage Spiking-SSM benchmark (Path B per ADR-001 D-19/D-20/D-21).

**Current state**:
- M5 (FL benchmark, 150 cells, 6 algos × 5 seeds × 5 alphas, 2 h 53 min) — complete; numbers preserved as Stage 2 ablation baseline.
- **Active workstream: Path B Stage 1 — centralized 3-architecture (LSTM / Mamba / Spiking-SSM) benchmark on ColO-RAN.**
- Stage 2 (FL × 3-architecture × 7-algorithm including FedBN) is **conditional on Stage 1 GO/NO-GO** at S1-W4 per ADR-001 D-21.

**Why pivoted**: FL-on-ColO-RAN angle was preempted by FL-DRAM (Springer 2026-03) and SliceFed (arxiv 2603.11390, 2026-03). Three-path Fermi analysis (2026-04-25) gave p(TMC) ≈ 8% for the original FL-only pitch vs ~9% with much lower variance for the centralized Spiking-SSM pitch. Stage 1 deliverables become Stage 2 ablation baselines, so no work is wasted on either branch.

Plan: `docs/ADR-001-v5-tmc-paper-plan.md` (read D-19 onwards for the pivot; D-1 to D-18 are M1-M5 history).
M5 paper-grade results (preserved, used as Stage 2 ablation): `docs/RESULTS_V5_FINAL.md`.
Stage 1 results destination (not yet created): `docs/RESULTS_V6_STAGE1.md`.

## Build / test commands

```bash
source .venv/bin/activate                  # always activate first
pytest                                     # 131 passing (M5 baseline); Stage 1 will add v6 arch tests
pytest tests/test_v5_*.py                  # v5 FL algorithm tests (preserved)
pytest tests/test_v6_*.py                  # Stage 1 architecture tests (to be written S1-W1)

# v5 (preserved, do not re-run unless audit needed):
python experiments/run_v5_algorithm_sweep.py --algorithm fedavg --alpha 0.5 --seed 42 ...
python experiments/run_v5_sweep_matrix.py --seeds 42 --alphas 0.5 --algo-spec 'fedavg:{}' ...
python experiments/run_moon_hpo.py --seed 42 --alpha 0.5 --mus 0.1 0.5 1.0 5.0 10.0 --taus 0.1 0.5 1.0
python scripts/aggregate_v5_results.py     # rebuilds RESULTS_V5.md from cells
./scripts/run_full_sweep.sh                # 150 cells, ~2 h 53 min — DONE, do not re-run

# Stage 1 (S1-W2 main sweep, per ADR-001 D-20):
python experiments/run_v6_arch_sweep.py --arch lstm,mamba,spiking \
  --seeds 42,0,1,2,3,7,11,13,17,23 --total-steps 5000 \
  --val-every 250 --sample-ratio 1.0
# Stage 1 (S1-W3 D-21 recovery if Spiking C1 fails — T_inner=5):
python experiments/run_v6_arch_sweep.py --arch spiking \
  --seeds 42,0,1,2,3,7,11,13,17,23 --total-steps 5000 \
  --spiking-t-inner 5 --output-suffix _t5
# Stage 1 aggregator + paper-grade markdown:
python scripts/aggregate_v6_results.py    # → docs/RESULTS_V6_STAGE1.md + aggregated.json
pytest tests/test_v6_*.py --no-cov         # 35 v6 tests; see ADR-001 D-20 TDD plan
```

Hardware: single RTX 4080, 16 GiB VRAM, Ubuntu 24.04, Python 3.12.3, PyTorch 2.10 + CUDA 12.8.
Data: `data/coloran_raw_unified.parquet` (18M rows, symlinked from `raw/colosseum-oran-coloran-dataset-master/`).
Repo: `fl-oran-tmc` (this repo); v1-v4 history in `colosseum-oran-federated-slicing` (upstream).

**Stage 1 dependencies (dep-sanity verified 2026-04-25, see ADR D-20)**:
```bash
VIRTUAL_ENV=/home/thc1006/dev/fl-oran-tmc/.venv uv pip install 'snntorch>=0.9' 'fvcore>=0.1.5' wheel ninja packaging
```
Outcome: `snntorch==0.9.4` + `fvcore==0.1.5.post20221221` work. `mamba-ssm` NOT installed (system has CUDA runtime via PyTorch but no `nvcc`; no pre-built cu128 wheel). **Active fallback**: implement `MambaS6Block` in pure PyTorch in-tree at `models/mamba_forecaster.py` (~150 LoC). No external Mamba dependency.

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
8. **Do not delete `artifacts/v5_sweep/` or `docs/RESULTS_V5_FINAL.md`** during Stage 1 cleanup. M5 numbers are the Stage 2 paper Table 3 source. `RESULTS_V5_FINAL.md` is committed; `artifacts/v5_sweep/` is gitignored and exists only on local disk — back up before any disk-heavy ops.

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
├── data/, data_v2/, data_raw/   # loaders, scalers, sequences, partitions (reused by Stage 1)
├── models/
│   ├── forecaster_v2.py         # ForecasterV2 (LSTM, Stage 1 baseline; reused as-is)
│   ├── mamba_forecaster.py      # Stage 1: MambaForecaster — Mamba-SSM backbone + same encoder/head (S1-W1 deliverable; first-of-kind so no V<n> suffix per D-20)
│   └── spiking_forecaster.py    # Stage 1: SpikingForecaster — LIF + selective scan + same encoder/head (S1-W1 deliverable)
├── federated/                   # aggregation, client(s), dp (used by Stage 2 only)
│   └── algorithms/              # FLAlgorithm Protocol, REGISTRY, fedavg/fedprox/
│                                # fedadam/scaffold/feddyn/moon, _local_loop helper.
│                                # Stage 2 will add fedbn.py.
├── training/                    # trainer (v1-v4), centralized_v3 (Stage 1 reuses), fl_v3,
│                                # fl_v5: V5Config, run_v5_sweep, forecaster_encode_fn
├── baselines/                   # persistence, gbm
├── evaluation/, utils/          # metrics, seed, gpu helpers (FLOPs/spike-count adders in S1-W3)
├── cli.py, logging_utils.py, config.py
experiments/
├── run_v3_centralized.py, run_v3_fl_iid.py, run_v3_fl_noniid.py  (v3 baselines)
├── run_v4_all_seeds.py                  (v4 multi-seed)
├── run_v5_algorithm_sweep.py            (v5 single-cell CLI; FL benchmark)
├── run_v5_sweep_matrix.py               (v5 multi-cell driver, shares data prep)
├── run_moon_hpo.py                      (v5 MOON hyperparameter grid)
├── run_v6_arch_sweep.py                 (Stage 1: 3-arch centralized; S1-W2 deliverable)
└── run_v7_fl_arch_sweep.py              (Stage 2 conditional: 3-arch × 7-algo FL; S2 deliverable)
scripts/
├── run_pilot.sh, run_seed_checkpoint.sh, run_full_sweep.sh, run_moon_hpo.sh
├── aggregate_v5_results.py              (v5 FL post-sweep paper Markdown generator; preserved)
└── aggregate_v6_results.py              (Stage 1 post-sweep generator; S1-W3 deliverable)
tests/  131 passing (89 v1-v4 baseline + 42 v5 TDD); Stage 1 adds 8 v6_*.py tests per D-20
docs/
├── ADR-001-v5-tmc-paper-plan.md         ← read D-19/D-20/D-21 for current direction
├── RESULTS_V5_PRELIM.md                 ← preserved (5-seed × α=0.5 + s42 α-curve snapshot)
├── RESULTS_V5_FINAL.md                  ← preserved (150-cell M5 result; Stage 2 ablation source)
└── RESULTS_V6_STAGE1.md                 ← Stage 1 paper-grade table (S1-W4 deliverable)
artifacts/
├── RESULTS_V4.md                        ← v3/v4 baseline numbers (do not compare to v5/v6 directly)
├── v5_sweep/                            ← M5 outputs preserved
└── v6_arch_sweep/                       ← Stage 1 outputs (created S1-W2)
```

## Current workstream — Path B Stage 1 (active as of 2026-04-25 post-Fermi-pivot)

M1-M5 (FL benchmark) done; pivoted to two-stage Spiking-SSM benchmark per ADR-001 D-19/D-20/D-21.

**Stage 1 — centralized 3-architecture benchmark (~5 weeks calendar, ~6 hr GPU)**:

1. **S1-W1 — Scaffolding** (P0; dep-sanity already done): mamba-ssm unavailable, fallback active. Create `src/fl_oran/models/mamba_forecaster.py` with in-tree `MambaS6Block` (~150 LoC) and `src/fl_oran/models/spiking_forecaster.py` with `SpikingSSMBlock` wrapping `snntorch.Leaky` (~180 LoC) per ADR D-20 explicit block configs. **TDD per individual test** (not all 8 at once): for each of `lif_neuron` → `mamba_shape` → `spiking_shape` → `param_count` → `spike_count` → `energy_metric` → `centralized_smoke` → `arch_swap_isolation_weak`, write the failing test first, implement minimum production code to pass, then refactor; commit at end of each cycle.

2. **S1-W2 — Centralized 3-arch sweep** (P0): Create `experiments/run_v6_arch_sweep.py` (3 archs × **10 seeds** × 5000 gradient steps, centralized only, reuse `training/centralized_v3.py::run_centralized` with the per-arch hyperparameter overrides pinned in ADR-001 D-20). Sweep wall-time ~6 hr on RTX 4080 (LSTM/Mamba ~10 min/run, Spiking ~30 min/run). Outputs to `artifacts/v6_arch_sweep/<arch>_s<seed>/{summary.json,history.csv,best_state.pt,energy.json}`.

3. **S1-W3 — Energy + statistics** (P0): Wire `fvcore.nn.FlopCountAnalysis` for LSTM/Mamba dense MACs. Instrument LIF layers with spike counters. Compute `sops` per arch on test set (10K random samples). Compute **paired-bootstrap CI95** (n_boot=10000) of `delta_auc(Spiking, LSTM)` and `delta_auc(Mamba, LSTM)` across 10 seeds per D-21. Generate energy-ratio Pareto + per-layer spike-rate heatmap.

4. **S1-W4 — Short paper draft + GO/NO-GO** (P0): Draft `docs/RESULTS_V6_STAGE1.md`. Draft IoTJ/TNSM short paper (6-8 pages, title in ADR-001 §8). Apply ADR-001 D-21 GO/NO-GO criteria (paired-bootstrap CI95 thresholds, not Wilcoxon p-values). Emit one of {Spiking-led GO, Mamba-led GO, NO-GO}. Append decision as new D-21 Outcome row.

**Stage 2 (conditional, only if S1-W4 = Spiking-led GO or Mamba-led GO)**: Add FedBN as 7th algorithm (`federated/algorithms/fedbn.py`, closes M6/D-17 gap). Integrate the chosen primary architecture and `MambaForecaster` (always retained as ablation) into the existing FL registry. Run **1050-cell sweep** (3 archs × 7 algos × 10 seeds × 5 alphas, ~14 hr GPU). M1-M5 LSTM × 6-algo numbers reused as Stage 2 paper Table 3 (legacy ablation). Submit to TMC; fallback TNSM Special Issue.

## Known caveats (paper writers must address in both Stage 1 and Stage 2)

**Stage 1 specific**:
- **No neuromorphic hardware**: energy is reported as theoretical `sops × 0.9pJ` vs `flops × 4.6pJ` using Horowitz 2014 45nm CMOS coefficients. Paper Limitations section must explicitly state this is upper-bound estimate, not deployment claim.
- **HiSTM (arxiv 2508.09184) is the closest Mamba-on-cellular precedent**: differentiate by dataset (ColO-RAN simulator, not Milan/Trentino traffic) + task (binary SLA classification, not regression) + spiking variant.
- **SpikySpace (arxiv 2601.02411) is the closest Spiking-SSM-on-time-series precedent**: differentiate by domain (RAN telemetry, not generic time series) and the slice-aware contextual setting.
- **Surrogate gradient + Adam stability is unverified**: S1-W1 first sanity is 1 epoch convergence on ColO-RAN val split. If divergent, escalate per ADR-001 Risk Register row.

**Stage 2 specific (only if reached)**:
- **FedBN missing from M5**: see ADR-001 D-17. Closed in Stage 2 sweep.
- **v4 vs v5 numbers not directly comparable**: v4 uses `sample_ratio=0.2` + 50K total gradient steps; v5 uses `sample_ratio=1.0` + 5K steps. Stage 2 reuses v5 setup.
- **MOON HPO test-leakage** (cosmetic): chosen (μ=0.1, τ=1.0) was selected by `test_auc` sort, not `best_val_auc`. Re-sorting by val gives identical winner (verified). Fix the script before Stage 2 sweep for principle.
- **n_clients=5 chosen by NIID-Bench convention, not by ColO-RAN physical structure** (7 gNBs). Stage 2 adds n_clients=7 (`mode="iid"`, bs_id partition) as ablation.
- **20 rounds is short for FL papers** (typical 100-1000). Stage 2 includes 100-round ablation at α=0.5 for SCAFFOLD/FedAdam to characterise convergence.

## What's in `artifacts/`

Do not delete. Contains all v1–v4 model checkpoints, logs, history CSVs, summary JSONs,
and `RESULTS_V4.md`. v5 outputs (M1-M5 FL benchmark, gitignored, **Stage 2 ablation source**) live under
`artifacts/v5_sweep/`. Stage 1 outputs will live under `artifacts/v6_arch_sweep/`
(also gitignored). Both are listed in `.gitignore`. The aggregated, paper-grade tables
(`docs/RESULTS_V5_FINAL.md`, future `docs/RESULTS_V6_STAGE1.md`) **are** committed and
serve as the recoverable source of truth if the gitignored raw cells are lost.

## How to ask for help

Prefer specific questions over broad ones. Good: *"v5 Dirichlet partition should use
`numpy.random.Generator.dirichlet` or `torch.distributions.Dirichlet`? Which matches
our existing partition.py style?"* Bad: *"how should I do non-IID?"*
