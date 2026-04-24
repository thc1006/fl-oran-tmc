# ADR-001: v5 Pipeline Extension for IEEE TMC Submission

- Status: **Accepted (with revisions 2026-04-25)**
- Authors: thc1006 + assistant
- Last updated: 2026-04-25

## Revision History

| Date | Change | Reason |
|------|--------|--------|
| 2026-04-25 a.m. | Initial draft (4 algorithms, 3 seeds, 60 runs) | Planning for TMC submission |
| 2026-04-25 p.m. | Literature review: extended to 6 algorithms, 5 α values, 5 seeds, 150 runs; added D-11 through D-15 | ERFO 2025 benchmarks against 9 baselines; NIID-Bench default α=0.5; reviewers expect FedDyn + FedAdam in 2025+ |
| 2026-04-25 late | Clarify D-4 (algorithm reuse mechanism), D-11 (scaler sharing implementation); add FedAvg dispatch-regression test to §3 | Pre-M1 review pass found 2 ambiguities + 1 missing regression test |
| 2026-04-25 eve | M1 complete (Dirichlet partition, FLAlgorithm registry, FedAvg, FedProx). Migrated to fresh repo `fl-oran-tmc`. | M1 deliverables done; clean-history repo for TMC supplementary material |
| 2026-04-25 night | M2 partial: FedAdam, SCAFFOLD, FedDyn landed (3/4). MOON deferred to M3. See D-16 below. | Rule-of-three refactor done via `run_local_sgd` helper; MOON is not pure plumbing |

### D-16. MOON deferred from M2 to M3

MOON (Li et al. 2021) requires a client-side contrastive loss on model
**representations**, not on raw weights. This needs the model to expose a
dedicated ``encode(x) → z`` method (or an intermediate-layer hook) with a
stable output shape and a documented projection head. That is a
paper-level design decision — different choices (penultimate layer vs a
separate projection MLP; detached vs through-graph previous-round
representation) materially change the result.

Shipping MOON without that decision would bake in an arbitrary interface
that downstream tests and the sweep script would depend on. Instead: M2
ships the four algorithms whose contracts are purely plumbing (FedAvg,
FedProx, FedAdam, SCAFFOLD, FedDyn — five once we count FedAvg).
**MOON moves to M3** alongside the orchestrator work, where the
representation API can be designed and reviewed together.

Sweep count stays at 5 algorithms × 5 α values × 5 seeds = 125 runs for
M3; MOON adds the 6th algorithm for a final 150-run matrix if the
representation API lands cleanly.

For full decision-by-decision audit trail see `git log --follow docs/ADR-001-v5-tmc-paper-plan.md`.

---

## 1. Context

v4 produced 3-seed Centralized / FL-IID / FL-NonIID results on ColO-RAN SLA forecasting (see `artifacts/RESULTS_V4.md`). 86 TDD tests pass. For IEEE TMC (target venue), v4 has two gaps: **only FedAvg** (reviewers expect 6+ algorithms in 2025+), and **only one non-IID setting** (reviewers expect Dirichlet α sweep). v5 closes these without re-engineering v1–v4. **Scope is deliberately bounded** — causal inference, DP, Byzantine robustness, and multi-dataset work are **out of scope** and deferred.

---

## 2. Decisions

### D-1. Algorithm set: 6 canonical FL algorithms

`FedAvg, FedProx, SCAFFOLD, MOON, FedDyn, FedAdam`. FedDyn and FedAdam were added after literature review (ERFO 2025 benchmarks 9; TMC reviewers expect ≥6).

### D-2. Module placement

- New algorithms → `src/fl_oran/federated/algorithms/`, one file per algorithm (≤ 200 LoC each).
- Dirichlet partition → extend existing `data_v2/partition.py` with `mode="dirichlet"` (no new file).
- v5 runner → `experiments/run_v5_algorithm_sweep.py` (one script, CLI-driven, no copy-paste per algorithm).

### D-3. Single source of truth (no function duplication)

These functions exist **exactly once** and must be **imported**, never reimplemented:

| Function | Location |
|----------|----------|
| `weighted_average_state_dicts` | `federated/aggregation.py` |
| `fit_continuous_scaler`, `federated_fit_scaler`, `apply_continuous_scaler` | `data_v2/encoders.py` |
| `ForecasterV2`, `FeatureSchema` | `models/forecaster_v2.py`, `data_v2/encoders.py` |
| `train_one_client_capped` | `federated/client_v2.py` |
| `build_run_sequences`, `ood_split_by_tr` | `data_v2/sequences.py`, `data_v2/split.py` |
| `_metrics`, `_batched_predict` | `training/centralized_v3.py` |
| `add_classification_target` | `data_v2/targets_v2.py` |

### D-4. FL algorithm interface (stateful, explicit)

Revised from original to accommodate SCAFFOLD/FedAdam server state and MOON/SCAFFOLD client state:

```python
@dataclass
class FLAlgorithmState:
    name: str
    server_state: dict         # FedAdam m/v, SCAFFOLD c_global
    client_states: dict[int, dict]  # SCAFFOLD c_local, MOON prev_local

class FLAlgorithm(Protocol):
    def init_state(self, model) -> FLAlgorithmState: ...
    def local_train(self, state, cid, model, x_cat, x_cont, y, cfg) -> ClientUpdate: ...
    def server_aggregate(self, state, updates, global_state) -> tuple[dict, FLAlgorithmState]: ...
```

Registry is a plain dict:
```python
REGISTRY = {"fedavg": FedAvg(), "fedprox": FedProx(mu=0.01), ...}
```
No ABC, no factory pattern.

**Reuse mandate**: each algorithm's `local_train` is a thin wrapper around `train_one_client_capped` (D-3). Algorithm-specific contributions are injected as closures, not as duplicate training loops:

| Algorithm | Injection mechanism |
|-----------|--------------------|
| FedAvg | vanilla `loss_fn` — no modification |
| FedProx | `loss_fn` adds `μ·‖w − w_global‖²` per step |
| FedDyn | `loss_fn` adds dynamic regularizer term; `server_aggregate` updates the per-client regularizer state |
| MOON | `loss_fn` adds contrastive term using cached previous-local-model features |
| SCAFFOLD | `grad_hook` (new optional param on `train_one_client_capped`) applies control-variate correction to gradients post-backward |
| FedAdam | no local change; `server_aggregate` replaces FedAvg with Adam-style m/v updates |

`train_one_client_capped` gains **one** optional parameter in M2: `grad_hook: Callable[[Module], None] | None = None`. This is a v5-driven additive change (D-9 permits additive changes with tests). No other v1–v4 function signatures change.

### D-5. Dirichlet α values (specified)

`α ∈ {0.05, 0.1, 0.5, 1.0, 10.0}`. 0.5 matches NIID-Bench default; 0.05 and 10.0 span extreme non-IID and near-IID. 5 values × 6 algorithms × 5 seeds = **150 runs**.

### D-6. Config: extend V3Config, do not create V5Config

Add fields to V3Config with defaults: `algorithm: str = "fedavg"`, `fedprox_mu: float = 0.01`, `fedadam_server_lr: float = 0.01`, `feddyn_alpha: float = 0.01`, `moon_temperature: float = 0.5`, `dirichlet_alpha: float | None = None`.

### D-7. Output layout

`artifacts/v5_sweep/<algorithm>_a<alpha>_s<seed>/{summary.json,history.csv,best_state.pt}`. Aggregation script produces `artifacts/v5_sweep/aggregated.json` + heatmap PNG.

### D-8. Language/tooling constraints

Python 3.12 (current venv). No `match/case`, no PEP 695 generics, no `@override`, no `type` keyword aliases, no `@dataclass(slots=True, kw_only=True)` without perf justification, no walrus outside hot loops, no async, no DI frameworks. Type annotations: `list[X]`, `dict[K, V]`, `X | None` only.

### D-9. Refactor discipline

**Do not refactor v1–v4 during v5 work** unless v5 reveals a concrete, testable correctness bug. "I notice this could be cleaner" is **not** a trigger. Real trigger example: "function returns wrong result for α=0.01 due to off-by-one". Refactors go in separate commits with failing test first.

### D-10. Debugging protocol

1. Read full stack trace. Identify originating file/line.
2. Reproduce deterministically. Pin seed, data, batch.
3. Bisect. Remove features until minimal example still fails.
4. Hypothesise **before** editing. Write one sentence stating what's wrong.
5. Fix root cause, not symptom. No `nan_to_num`, no blanket try/except.
6. Add regression test that fails on main and passes on fix.
7. Never use `try/except: pass`, `filterwarnings("ignore")`, `--no-verify`.

### D-11. Shared scaler instance across algorithms

All 6 algorithms share the **same** `federated_fit_scaler` output per `(seed, α)` pair. No per-algorithm re-fitting (would bias comparison).

**Implementation**: achieved via **deterministic fitting**. `federated_fit_scaler` is a pure function of (data partition, schema) — given identical inputs it returns identical mean/std (up to float32 precision). Because each of the 6 algorithm runs within the same `(seed, α)` receives the identical partitioned data, re-fitting inside each run produces bit-equivalent scaler output. **No explicit cross-run caching is required**; a regression test (see §3) pins this determinism.

### D-12. Statistical testing: 5 seeds + pairwise Wilcoxon

3 seeds (v4) was too few. v5 uses 5 seeds per configuration. Report mean ± std AND pairwise Wilcoxon signed-rank test on AUC across seeds for algorithm-vs-algorithm comparisons. Significance threshold: p < 0.05.

### D-13. Minimal ablation study

Three ablations, each 1 seed × best-α-from-sweep × Centralized-only (no FL) to isolate architecture contribution:
- Remove embedding layer (one-hot categoricals instead)
- Remove trend features (raw inputs only)
- Reduce seq_len from 5 to 1 (feed-forward instead of LSTM)

Results go in paper Table 2. Total added runs: 3.

### D-14. MOON first-round fallback

First time a client is selected, previous-local-model is undefined. Contrastive loss term is set to 0 for that round (algorithm degrades to FedAvg for that client). Test: `test_v5_moon_first_round_fallback`.

### D-15. Reproducibility commitment

On paper acceptance: release `src/`, `experiments/`, `tests/`, `docs/`, plus aggregated sweep outputs (not raw `best_state.pt` files due to size) under AGPL-3.0. Seeds listed in paper. Hardware (RTX 4080) listed in paper. Exact commit hash referenced in paper.

---

## 3. TDD Plan (test-first, ordered)

Write tests in this order. Each fails before corresponding implementation exists:

| Test file | Key assertions |
|-----------|----------------|
| `test_v5_dirichlet_partition.py` | α=100 → near-uniform; α=0.01 → concentrated; total rows preserved; seed reproducible |
| `test_v5_registry.py` | REGISTRY has all 6 keys; each implements protocol |
| `test_v5_fedprox.py` | μ=0 reduces to FedAvg; proximal term ≥ 0; global state not mutated |
| `test_v5_moon.py` | Identical representations → contrastive = 0; first round fallback works |
| `test_v5_scaffold.py` | Control variates correct shape; zeroed controls → FedAvg equivalent |
| `test_v5_feddyn.py` | α=0 reduces to FedAvg; regularizer ≥ 0 |
| `test_v5_fedadam.py` | Server m/v updated correctly; β=0 reduces to SGD-server |
| `test_v5_end_to_end_smoke.py` | Each algorithm: 3 clients × 500 rows × 3 rounds, loss decreases, no crash |
| `test_v5_fedavg_dispatch_regression.py` | FedAvg-via-dispatch produces final loss identical (atol=1e-5) to direct FedAvg path on pinned synthetic data + seed; guards against silent drift when `algorithm="fedavg"` config replaces the direct code path |
| `test_v5_scaler_determinism.py` | `federated_fit_scaler` on identical (data, schema) returns bit-equivalent mean/std; guards the D-11 "no caching needed" claim |

Detailed per-test specifications live in each test file's docstring, not here.

---

## 4. Implementation Milestones (no training runs in this ADR)

| Week | Milestone | Content |
|------|-----------|---------|
| 1 | M1 Foundation | Dirichlet partition + registry skeleton + FedProx (TDD each) |
| 2 | M2 More algorithms | SCAFFOLD + MOON + FedDyn + FedAdam (TDD each) |
| 3 | M3 Sweep | Pilot (1 algorithm × 1 α × 1 seed), then full 150-run sweep + aggregation script |
| 4–6 | M4 Writing | Paper draft, iterate, submit |

Sweep estimated wall-time: 150 runs × ~4 min avg (SCAFFOLD/MOON slower than FedAvg) ≈ **10 h**. Run overnight.

---

## 5. Risk Register (high-likelihood items only)

| Risk | Mitigation |
|------|-----------|
| SCAFFOLD doubles communication cost + needs stateful clients | Store control variates on CPU; move to GPU per-round; test no-op when zeroed |
| MOON fails at very low α (< 0.05) due to slice-absent clients | Minimum sample floor per client (50 rows); skip affected α if needed |
| Over-engineering ("let me refactor aggregation.py while I'm here") | **D-9 forbids this**. Trust the ADR. |
| One run crashes mid-sweep | try/except at runner level; log + continue; don't abort batch |
| 150 runs takes > 12 hours | Reduce to 3 seeds if necessary (documented downgrade) |

---

## 6. Consequences

**Positive**: v4 artifacts stay valid; reviewers get the standard FL-algorithm comparison they expect; 5-seed + Wilcoxon meets 2025 statistical bar; scope is bounded (paper sprint not research rabbit hole).

**Negative**: Does **not** enable NeurIPS/ICML main — accepted trade-off (prior discussion). SCAFFOLD/MOON implementations each add state complexity. ~10 h GPU time for full sweep.

**Neutral**: Codebase +600–900 LoC across 9 new files. Test count 89 → ~120.

---

## 7. Alternatives Considered

1. Keep v4 and submit as-is → Rejected: insufficient algorithm comparison for TMC.
2. Only 3 algorithms (FedProx/SCAFFOLD/MOON) → Rejected: post-literature-review, 2025 bar is 6+.
3. Use Flower/FedML framework → Rejected: our CUDA-graph + federated-scaler customisations don't fit without adapter overhead.
4. Full rewrite into `v5/` sub-package → Rejected: violates D-9.
5. Add SliceAvg custom algorithm as primary contribution → Deferred to optional section 7 in paper, only if sweep results justify it.

---

## 8. Paper Outline (frozen early to anchor experiments)

**Title**: *Federated SLA-Violation Forecasting in O-RAN: A Leakage-Free Benchmark and Non-IID Algorithm Comparison*

**Sections**: (1) Intro, (2) Background on ColO-RAN / O-RAN slicing / FL, (3) **Target leakage audit** (v4 finding: `allocation_efficiency = 0.5·throughput_eff + 0.3·qos + 0.2·prb_util`), (4) Clean task definition + OOD split, (5) Methodology (ForecasterV2, federated scaler), (6) **Algorithm comparison under Dirichlet non-IID**, (7) Results + analysis, (8) Ablation + limitations, (9) Conclusion.

**Target**: IEEE TMC. Fallback: IEEE TNSM. Key figures: (a) leakage audit scatter, (b) 6×5 algorithm×α heatmap, (c) training curves per algorithm. Key tables: (a) baselines vs FL algorithms at α=0.5, (b) ablation results, (c) pairwise Wilcoxon p-values.
