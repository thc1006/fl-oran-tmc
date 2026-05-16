# fl_oran — Project context for Claude

Local PyTorch pipeline for ColO-RAN O-RAN slice SLA forecasting. Started as FL benchmark; pivoted 2026-04-25 to two-stage Spiking-SSM benchmark (Path B per ADR-001 D-19/D-20/D-21); Stage 2 GO Spiking-led; pivoted again 2026-04-27 to "test the separability assumption" framing; venue switched 2026-05-05 to **IEEE JSAC** (was TMC).

**Revision History (top-level project state)**:

| Date | Change | Reason |
|------|--------|--------|
| 2026-04-25 | M5 done (150 FL cells, 2h53min, 6 algos × 5 seeds × 5 alphas) | TMC baseline |
| 2026-04-25→26 | Stage 1 done; D-21 GO Spiking-led after `spiking_expand2` audit at matched-25k sparsity-aware | Tier B.2 audit pass |
| 2026-04-27 | Venue locked IEEE TMC primary / OJ-COMS fallback (later switched to JSAC); narrative locked "separability assumption test"; Phase 5 = 900-cell FULL sweep planned | 3-round lit-review reconfirmed niche |
| 2026-04-28 | Stage A: lr-1e-4 spiking bug fixed (5e-4), num_rounds 20→100, FedDyn default `option_ii` (canonical diverges under Adam at 100r), B2 NVML integration | C1/C2/C4 control audit revealed undertraining |
| 2026-04-30→05-01 | Phase 5 v2 launched (900 cells = 3 archs × 5 algos × 6 partitions × 10 seeds), ADR-002 v3 REJECTED Phase 6 (FedSWA) | 6-layer mechanism-based rejection |
| 2026-05-02 | Step 1+2 measurement: dataset has **3 slices** (not 4), **3 schedulers** (not 4), **17 features** (not 29), **30.9% pos rate** (not 8-12%), bs↔slice KL=0; original §7.1 mechanism narrative invalidated | Fact-finding before paper writing |
| 2026-05-04→07 | Paper SPLIT to main (11,892w) + supplementary (1,757w); S11 LaTeX migration A→I complete; PR #1 merged | JSAC submission prep |
| 2026-05-05 | **Venue switched IEEE TMC → IEEE JSAC** (commit `18640f4`) — fallback OJ-COMS | Reviewer-fit for FL × O-RAN slicing |
| 2026-05-07→08 | R1/R2 reviewer-feedback rounds (PR #11-#17): ORCID corrected `0009-…7115-0149` → `0000-0001-7421-8027`, author name `Hao-Chun` → `Hsiu-Chi`, REM-1..4 + R34 fixes, v0.9.2 submission-ready tag, Zenodo deposit DOI **10.5281/zenodo.20075433** | Post-submission polish |
| 2026-05-16 | Project rsync'd 19 GB colosseum + 425 MB fl-oran-tmc + 96 MB .claude state to `4060-dev` (Tailscale 100.119.71.41); current machine | Capacity offload for future training |

**Current state (2026-05-16, this machine = `4060-dev`)**:
- Stage 1 + Stage 2 (Phase 5 900-cell sweep) + Phase 6 (per-BS Dirichlet ablation, R2 §7.1.5) + R2 reviewer polish (PRs #11-#17) all merged into `main`. Paper at v0.9.2-submission-ready, PDFs deposited to Zenodo.
- **No active long sweep.** Local `artifacts/v7_stage2_full/` has the 900 raw cells (195 MB, gitignored). Aggregator output is committed at `docs/RESULTS_V7_PHASE5.md`.
- Likely next-task surface: (a) reviewer responses if JSAC returns comments, (b) Stage 1 standalone short paper `docs/PAPER_V6_STAGE1.md` (currently in-prep), (c) follow-up research per `docs/FUTURE_WORK_RESEARCH.md`.

Plan + decision log: `docs/ADR-001-v5-tmc-paper-plan.md` (read D-19/D-20/D-21/D-22; the "tmc-paper-plan" in the filename is historical — venue is now JSAC).
M5 FL-benchmark results (Stage 2 ablation Table 3): `docs/RESULTS_V5_FINAL.md`.
Stage 1 centralized 3-arch results: `docs/RESULTS_V6_STAGE1.md` + `docs/RESULTS_V6_STAGE1_ANALYSIS.md` (paper-grade ≈420 lines).
Stage 2 FL × arch sweep results: `docs/RESULTS_V7_PHASE5.md` (paper Table 4 source).
Phase 6 FedSWA rejection rationale: `docs/ADR-002-phase6-fedswa.md`.
JSAC paper artifacts: `paper/main.tex` (130 KB, 21 pp) + `paper/supplementary.tex` (15.9 KB, 3 pp) + `paper/bibliography.bib` (61 entries).

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

# Stage 1 (DONE; 30 cells × 10 seeds; aggregator already ran):
python experiments/run_v6_arch_sweep.py --arch lstm,mamba,spiking_expand2 ...
python scripts/aggregate_v6_results.py    # → docs/RESULTS_V6_STAGE1.md
pytest tests/test_v6_*.py --no-cov         # 90 v6 tests after Phase 0 latency/EDP

# Stage 2 (DONE; Phase 5 900-cell sweep; aggregator already ran):
python experiments/run_v7_phase_sweep.py --spec experiments/specs/stage2_full.yaml \
  --skip-completed   # primary spec-driven launcher (Phase 1.5f)
python scripts/aggregate_v7_results.py --sweep-dir artifacts/v7_stage2_full \
  --output docs/RESULTS_V7_PHASE5.md
pytest tests/test_v7_*.py --no-cov         # 162+ v7 tests after Phase 1.5g

# Full test suite (605+ tests post-R2; ~9 s with --no-cov):
pytest --no-cov

# Paper claim-source regeneration (gated by `tests/test_paper_claims_sources.py` skip-or-run):
python scripts/step1_fact_finding.py        # → artifacts/step1_factfinding.json
python scripts/step2_mechanism_search.py    # → artifacts/step2_mechanism_search.json
python scripts/phase5_paper_figures.py      # regenerates artifacts/figures/*.{pdf,svg}
python scripts/phase5_dashboard.py          # artifacts/phase5_dashboard.{html,png}

# Paper LaTeX build (paper/ dir; needs texlive-full):
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd paper && pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

Hardware: **single RTX 4060 Ti, 16 GiB VRAM** (was RTX 4080 16 GiB until 2026-05-16 migration), Ubuntu 26.04, kernel 7.0.0-15, 20 CPU cores, ~30 GB RAM (was 32 cores / 128 GB on 4080-0), driver 595.58.03, CUDA 13.2 runtime. Python 3.12 + PyTorch 2.10 (`.venv/` is a symlink to the upstream colosseum-oran-federated-slicing venv and must be rebuilt against the local PyTorch+CUDA on this machine if it was created on the old box).
Data: `data/coloran_raw_unified.parquet` (18M rows, symlinked from `raw/colosseum-oran-coloran-dataset-master/`).
Repo: `fl-oran-tmc` (this repo); v1-v4 history in `colosseum-oran-federated-slicing` (upstream; the upstream repo is ALSO present on this machine at `/home/thc1006/dev/colosseum-oran-federated-slicing/` — symlinks resolve).

**Stage 1 dependencies (dep-sanity verified 2026-04-25, see ADR D-20; same applies on 4060-dev)**:
```bash
VIRTUAL_ENV=/home/thc1006/dev/fl-oran-tmc/.venv uv pip install 'snntorch>=0.9' 'fvcore>=0.1.5' 'nvidia-ml-py' wheel ninja packaging
```
Outcome: `snntorch==0.9.4` + `fvcore==0.1.5.post20221221` + `nvidia-ml-py` (replaces deprecated `pynvml`) work. `mamba-ssm` NOT installed. **Active fallback**: pure-PyTorch `MambaS6Block` in `models/mamba_forecaster.py`. No external Mamba dep.

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
├── run_v3_*.py / run_v4_all_seeds.py     (v3/v4 baselines; do not modify per D-9)
├── run_v5_algorithm_sweep.py / run_v5_sweep_matrix.py / run_moon_hpo.py  (M5 FL benchmark)
├── run_v6_arch_sweep.py                  (Stage 1: 3-arch centralized; DONE)
├── run_v7_fl_arch_sweep.py               (Stage 2 single-cell CLI)
├── run_v7_fl_arch_sweep_matrix.py        (Stage 2 multi-cell driver, joblib + SharedSplits)
├── run_v7_phase_sweep.py                 (Stage 2 spec-driven launcher; PRIMARY entry; Phase 1.5f)
├── run_p1_centralized_lstm.py            (P1.5 centralized LSTM 5-seed CI95 for §6.7)
├── run_p1_tr_embedding_check.py          (P0 tr embedding audit)
├── run_p2_inference_latency.py / run_p2_loto_cluster_bootstrap.py  (R2 reviewer adds)
├── run_r2_post_hoc_per_bs_finetune.py / run_rem1_local_only_per_bs_lstm.py  (R2 follow-ups)
├── preregistered/                         (P1 prereg YAMLs: naive_baselines, tr_embedding, fedbn)
└── specs/                                 (Phase 1.5g spec YAMLs: phase2_min, phase3a/c/e, stage2_full, r2_*, r34_fedswa_*, r2_no_tr_ablation, p1_*, ablation_random_split)
scripts/
├── aggregate_v5_results.py / aggregate_v6_results.py / aggregate_v7_results.py  (paper Markdown gens)
├── _v6_cell_metadata.py / _v7_cell_metadata.py / _v7_spec_loader.py             (canonical name builders + spec loader)
├── measure_v6_gpu_energy.py / recompute_v6_energy.py                            (NVML + idempotent re-aggregation)
├── phase5_dashboard.py / phase5_paper_figures.py                                (paper Fig 1-3 generators)
├── step1_fact_finding.py / step2_mechanism_search.py                            (paper claim-source generators)
├── baseline_last_bler.py / baseline_logreg.py                                   (naive baselines for §6/§7 contrasts)
├── check_preregistered.py / oom_watchdog.py                                     (preregistered-YAML auditor + nvidia-smi-aware OOM guard)
└── v100_*.sh                                                                    (V100 cluster launchers; cluster is at colosseum-oran-federated-slicing peer)
tests/  605+ passing (89 v1-v4 + 42 v5 + 90 v6 + 162+ v7 + R1/R2/P0/P1 audit invariants + paper-claim sources)
docs/
├── ADR-001-v5-tmc-paper-plan.md         ← decision log; read D-19/D-20/D-21/D-22 (note: "tmc-paper-plan" is historical filename — venue is JSAC since 2026-05-05)
├── ADR-002-phase6-fedswa.md             ← REJECTED v3 (do not implement FedSWA; mechanism-based dismissal in §related-work + §threats)
├── RESULTS_V5_FINAL.md                  ← M5 paper-grade FL-benchmark table (Stage 2 paper Table 3 legacy ablation)
├── RESULTS_V6_STAGE1.md                 ← Stage 1 centralized 3-arch paper table
├── RESULTS_V6_STAGE1_ANALYSIS.md        ← Stage 1 paper-grade analysis ≈420 lines
├── RESULTS_V7_PHASE5.md                 ← Stage 2 FL × arch 900-cell sweep paper Table 4
├── PAPER_DRAFT.md                       ← Markdown source of truth (now mirrored to paper/main.tex)
├── PAPER_SUPPLEMENTARY.md               ← Supplementary App. A-D (mirrored to paper/supplementary.tex)
├── PAPER_V6_STAGE1.md                   ← Stage 1 standalone short paper (in-prep)
├── PAPER_CONTRIBUTION_CLAIM.md          ← §1 contribution audit log
├── FUTURE_WORK_RESEARCH.md / FUTURE_STUDY.md   ← candidate next-paper directions
└── archive/                             ← superseded notes with `status: superseded` frontmatter
paper/
├── main.tex (130 KB, 21 pp) / supplementary.tex (15.9 KB, 3 pp) / bibliography.bib (61 entries)
└── main.pdf / supplementary.pdf         ← v0.9.2-submission-ready; deposited at Zenodo DOI 10.5281/zenodo.20075433
artifacts/
├── RESULTS_V4.md                        ← v3/v4 baseline numbers
├── v5_sweep/                            ← M5 outputs (gitignored)
├── v6_arch_sweep/                       ← Stage 1 outputs (gitignored)
├── v6_arch_sweep_audit/                 ← Stage 1 audit/recompute outputs (gitignored)
├── v7_stage2_full/                      ← Phase 5 900-cell Stage 2 main sweep (gitignored; 195 MB)
├── v7_phase2_min/, v7_phase3a_stress/, v7_phase3e_envelope/   ← scoped FL sweeps (gitignored)
├── v7_phase6_per_bs_dirichlet/, v7_phase6_threshold/          ← Phase 6 ablations (gitignored)
├── v7_ablation_random_split/, v7_arch_smoke/, v7_control_extended/  ← R2 + control experiments (gitignored)
├── audit/                               ← AUDIT_PLAYBOOK + invariant artifacts
├── baselines/                           ← naive baselines outputs
├── figures/                             ← paper Fig PDFs + SVGs (algo_ranking, interaction_heatmap, pareto, results_table.csv)
├── p1_*/, p2_*/                         ← P0+P1+P2 audit + inference latency + LOTO bootstrap
├── r2_*, r34_*                          ← R2 reviewer-feedback follow-ups
├── step1_factfinding.{json,md} / step2_mechanism_search.{json,md}   ← paper-claim sources
└── phase5_dashboard.{html,png}          ← interactive Phase 5 dashboard
```

## Past workstreams (DONE — not active)

M1-M5 FL benchmark → Stage 1 centralized 3-arch → Stage 2 FL × arch 900-cell sweep → Phase 6 ablations → R1/R2 reviewer-feedback polish → v0.9.2 submission tag → Zenodo deposit. Decision history is in ADR-001 Revision History table. **No active long-running sweep on this machine.**

## Current workstream (2026-05-16 onwards) — TBD

The transfer to `4060-dev` (this machine) was the last item closed. Awaiting user direction on:
- (a) Stage 1 standalone short paper finalisation (`docs/PAPER_V6_STAGE1.md` is in-prep, IoTJ/TNSM target).
- (b) Reviewer response handling if/when JSAC returns comments.
- (c) Follow-up research directions per `docs/FUTURE_WORK_RESEARCH.md`.
- (d) Any new Stage 1/2 ablations on this 4060 Ti machine.

## Known caveats already addressed in paper

**Closed via committed paper text** (see `docs/PAPER_DRAFT.md` + `paper/main.tex` for §-anchors):
- **No neuromorphic hardware** → §discussion applicability-boundary; Horowitz 2014 coefficients cited explicitly; theoretical-vs-NVML-real inversion analysed (§6.8 + §8 L15 ratios).
- **FedBN gap from M5** → closed by Phase 1 FedBN 30-cell sweep (commit `86b97b8`); FedBN bit-identical to FedAvg on our 3 archs (no BN), documented §6 contribution C1+C2+C3.
- **n_clients = 7 gNBs** (was 5) → Phase 5 uses n=7 natural-by-BS + Dirichlet α∈{0.05..5}.
- **20 rounds → 100 rounds** (Stage A 2026-04-28); spiking lr 1e-4 → 5e-4; FedDyn `option_ii` default (canonical diverges).
- **Step 1+2 measurement** invalidated original §7.1 mechanism (sparse-positive + per-client pos_weight + bs↔slice correlation all false). Replaced with "bs-conditioned channel-state signal preservation" hypothesis backed by §7.1.1 random_split V100 ablation (LSTM/Mamba/Spiking all drop −0.17~0.19 AUC when bs grouping is broken).
- **MOON deferred entirely in fl_v7 (Phase 1.5)** — `_select_algorithm` raises NotImplementedError for non-LSTM MOON.
- **Phase 6 FedSWA** → REJECTED per ADR-002 v3; mechanism-based §related-work + §threats paragraphs already drafted.
- **R1/R2 reviewer rounds** → all closed by PR #11-#17; v0.9.2-submission-ready.

## What's in `artifacts/`

Do not delete. v1-v4 baselines preserved (`RESULTS_V4.md` is the only artifact-level
committed README). All v5/v6/v7 raw cells are gitignored; the aggregated paper-grade
tables (`docs/RESULTS_V5_FINAL.md`, `RESULTS_V6_STAGE1.md`, `RESULTS_V6_STAGE1_ANALYSIS.md`,
`RESULTS_V7_PHASE5.md`) are committed and act as the recoverable source of truth if raw
cells are lost. The 900-cell `v7_stage2_full/` (195 MB) is the largest local-only artifact;
re-generating from scratch on RTX 4060 Ti is ~80-100 hr GPU.

## How to ask for help

Prefer specific questions over broad ones. Good: *"v5 Dirichlet partition should use
`numpy.random.Generator.dirichlet` or `torch.distributions.Dirichlet`? Which matches
our existing partition.py style?"* Bad: *"how should I do non-IID?"*
