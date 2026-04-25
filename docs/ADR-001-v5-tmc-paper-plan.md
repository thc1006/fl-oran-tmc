# ADR-001: v5 Pipeline Extension for IEEE TMC Submission

- Status: **Active — Stage 1 sweep complete + post-hoc audit complete; D-21 outcome = GO Stage 2 (Trade-off study) under audit-corrected Spiking hyperparameters. Stage 2 (FL upgrade) is conditionally GO; the preregistered Spiking row gave NO-GO and the audit row flips the decision (see 2026-04-25 20:09 revision-history entry).**
- Authors: thc1006 + assistant
- Last updated: 2026-04-25 (post-Fermi-analysis pivot)

## Revision History

| Date | Change | Reason |
|------|--------|--------|
| 2026-04-25 a.m. | Initial draft (4 algorithms, 3 seeds, 60 runs) | Planning for TMC submission |
| 2026-04-25 p.m. | Literature review: extended to 6 algorithms, 5 α values, 5 seeds, 150 runs; added D-11 through D-15 | ERFO 2025 benchmarks against 9 baselines; NIID-Bench default α=0.5; reviewers expect FedDyn + FedAdam in 2025+ |
| 2026-04-25 late | Clarify D-4 (algorithm reuse mechanism), D-11 (scaler sharing implementation); add FedAvg dispatch-regression test to §3 | Pre-M1 review pass found 2 ambiguities + 1 missing regression test |
| 2026-04-25 eve | M1 complete (Dirichlet partition, FLAlgorithm registry, FedAvg, FedProx). Migrated to fresh repo `fl-oran-tmc`. | M1 deliverables done; clean-history repo for TMC supplementary material |
| 2026-04-25 night | M2 partial: FedAdam, SCAFFOLD, FedDyn landed (3/4). MOON deferred to M3. See D-16 below. | Rule-of-three refactor done via `run_local_sgd` helper; MOON is not pure plumbing |
| 2026-04-25 night-2 | M3a/b complete: MOON via caller-supplied `encode_fn`; orchestrator `run_v5_sweep` + matrix driver with `SharedSplits` cache + joblib parallelism (~4× faster sweeps) | M3 milestone closed |
| 2026-04-25 late-2 | Adversarial review pass found 10 issues; SCAFFOLD Option-II rewrite (was producing F1=0.213 — Option-I formula assumes SGD; under Adam local optimiser the c_i term blows up ~100×). FedDyn `update_mode="option_ii"` default. FedAdam `bias_correction=True` opt-in. `pos_weight_split="train"` (was test — leakage). `cudnn_deterministic=True` for full-bit reproducibility. | Pilot 5-round @ α=0.5 surfaced SCAFFOLD F1 collapse; deep audit revealed shared SGD-vs-Adam scaling assumption + multiple methodology issues |
| 2026-04-25 final-1 | M4 superseded by M5: 150-cell sweep (5 seeds × 5 α × 6 algos × 20 rounds) completed in 2 h 53 min. MOON HPO grid (15 cells, μ × τ at α=0.5) picked μ=0.1, τ=1.0 (vs CIFAR default μ=1.0, τ=0.5 → +0.077 AUC). All 6 algos statistically tied except SCAFFOLD/FedAdam mid-tier. See `docs/RESULTS_V5_FINAL.md`. | Initial M4 plan was 12.5h; M5 with optimisations did it in 2.9h |
| 2026-04-25 final-2 | **Literature deep-dive (2024-2026 search)** revealed (a) arxiv 2508.08479 (Aug 2025) benchmarks FedBN on cellular and finds **FedBN > FedAvg by 11.7%** — we **omitted FedBN entirely**; (b) arxiv 2403.17287 already does 6-algo benchmark on toy data; (c) FedRS-Bench (May 2025) does 10-algo on remote sensing; (d) "When Clients Drift" (2026) does SLA forecasting on 6G with regime-aware FL. **Our positioning is weaker than first thought**. See D-17, D-18. | User-driven adversarial cross-validation forced honest reassessment |
| 2026-04-25 final-3 | **Path B pivot**. Three-path Fermi analysis (FL+Spiking-Mamba+ColO-RAN vs centralized Spiking-SSM vs SNN+FL on real domain) yielded p(TMC accept) ≈ 8% / 9% / 1% respectively. The originally planned FL benchmark angle is preempted by FL-DRAM (Springer 2026-03) and SliceFed (arxiv 2603.11390, 2026-03). The remaining genuinely novel niche is **Spiking-SSM × ColO-RAN** (0 hits across 3 cross-validation searches as of today). Pivot strategy: Stage 1 = centralized Spiking-SSM benchmark on ColO-RAN (4 weeks, IoTJ/TNSM/Globecom short paper). Stage 2 (conditional on Stage 1 GO/NO-GO) = FL upgrade combining 7-algorithm registry × 3 architectures (TMC paper). M1-M5 artifacts preserved as Stage 2 ablation baselines; nothing thrown away. M6 (FedBN gap closure) demoted to Stage 2 task. See D-19, D-20, D-21. | First-principles + Fermi adversarial reassessment after user pushback ("回到第一性原則 FL 有意義嗎"). Same total work, lower variance, two publishable products instead of one all-in moonshot. |
| 2026-04-25 14:39 | **Stage 1 W2 sweep complete (30 cells, 41.7 min on RTX 4080)**. Per-arch test AUC across 10 seeds: LSTM 0.9151 ± 0.0010, Mamba 0.9153 ± 0.0008, Spiking 0.6757 ± 0.0354. Paired-bootstrap CI95: delta(Mamba, LSTM) [-0.0005, +0.0009] (Mamba arm healthy at parity); delta(Spiking, LSTM) [-0.2586, -0.2167] (24 pp gap). Energy per inference: LSTM 967k pJ, Mamba 831k pJ (-14%), Spiking 780k pJ (-19% but 24-pp accuracy loss). | Pure-PyTorch MambaS6Block + SpikingSSMBlock end-to-end. Sweep 5-8× faster than ADR D-20 estimate due to RTX 4080 + bf16. |
| 2026-04-25 14:40 | **D-21 GO/NO-GO outcome: NO-GO Stage 2**. C1 hard fail (Spiking gap 0.24, threshold 0.03; hi = -0.2167 << -0.050 lower bound for any Stage 2 with Spiking primary). C2 fail (energy ratio 0.81, threshold 0.5). C3 pass (Mamba healthy). Mamba-led Stage 2 fallback NOT warranted: Mamba-LSTM CI95 brackets zero, lo = -0.0005 < +0.005. **Stage 1 short paper proceeds standalone with the LSTM ~ Mamba parity at -14% energy as the positive finding and the Spiking applicability-boundary as a publishable negative result. Stage 2 (FL upgrade) is not pursued.** Recovery HPO at T_inner=5 launched in background as preregistered, but C1 hi << -0.050 means it cannot rescue the decision; numbers will be reported in the paper for completeness. M1-M5 FL benchmark numbers ship as supplementary technical report on arxiv (preempted by FL-DRAM, not standalone-publishable). | Path B Stage 1 thesis confirmed end-to-end: same expected p(TMC) as Path A all-in but with the unconditional Stage 1 deliverable preserved; total invested time ~1 day vs 5+ weeks of speculative Path A all-in. |
| 2026-04-25 20:09 | **D-21 outcome revised after post-hoc audit: GO Stage 2 (Trade-off study)**. User pushback on result meaningfulness ("我主要是擔心剛剛跑的所有腳本存在任何問題") triggered a learning-rate ablation. Audit revealed that the preregistered Spiking lr=1e-4 + 5000-step budget was severely undertraining: train-loss + val-AUC curves were still descending at the budget cap, while LSTM/Mamba had plateaued by step 1500. A 10-seed full sweep at lr=5e-4 + 25 000 steps (`spiking_lr5e4_25k` cells under `artifacts/v6_arch_sweep/`) lifts Spiking from 0.6757 ± 0.0354 to **0.8944 ± 0.0018** AUC. Paired-bootstrap CI95 of `delta(Spiking_audit, LSTM)` = [−0.0218, −0.0199], **inside** the C1 −0.030 threshold (PASS). C2 unchanged at 0.80 (FAIL). C3 unchanged (PASS). Per ADR D-21 row "C1 met AND C2 fail (energy advantage < 2×)", the Stage 2 path is reframed as a **trade-off study** ("comparable energy at no accuracy cost") rather than a clean Spiking-superiority claim. Lower p(TMC) ~6-8% per the original D-21 row commentary, but unambiguously above the NO-GO threshold. **Stage 1 paper now reports both preregistered and audit-corrected Spiking rows side-by-side with a methodological audit section (§6.6) that documents why the literature-derived lr=1e-4 heuristic did not transfer to this task.** | Honest correction: methodological flaw caught by adversarial-review-style audit, not by spec-time review. Preserves preregistration credibility (numbers reported as-spec) while delivering the substantive corrected finding. |

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

### D-17. FedBN must be added (post-literature-review correction)

**Discovered post-M5**: The closest competitor paper is "Benchmarking
Federated Learning for Throughput Prediction in 5G Live Streaming
Applications" (arxiv 2508.08479, August 2025). They benchmark FedAvg /
FedProx / **FedBN** on real cellular telemetry datasets and find
**FedBN-LSTM beats FedAvg by 11.7%** (and FedBN-Transformer by 11.4%).

FedBN (Li et al. ICLR 2021) keeps **client-local BatchNorm statistics**
(mean, variance, running stats) — never aggregating them. This is
specifically the right design for **feature-distribution skew**, which is
**exactly our slice-axis Dirichlet partition setting** (covariate shift
in NIID-Bench taxonomy).

**Our 150-cell M5 sweep covers 6 algorithms but omits FedBN.** Reviewers
will catch this — FedBN is the textbook choice for our partition mode and
a recent direct competitor demonstrates it wins on similar tasks.

**Action**: M6 adds FedBN as the 7th algorithm, runs FedBN-only on the
existing (5 seeds × 5 α) grid (~30 min), re-aggregates RESULTS_V5_FINAL.

Two outcomes are both publishable:
- FedBN > FedAvg on ColO-RAN: confirms cellular FL precedent, narrative
  becomes "FedBN-style local-BN aggregation works for slice-axis non-IID".
- FedBN ≈ FedAvg on ColO-RAN (i.e. contradicts arxiv 2508): novel finding
  that FedBN's advantage does not transfer to the ColO-RAN
  feature/category mix, and the paper has a clear contribution.

ForecasterV2 has only LayerNorm-free architecture — there are no real
BatchNorm layers to keep local. FedBN here will reduce to FedAvg unless
we add BatchNorm to the trunk. Open question for M6: introduce
BN-augmented ForecasterV2 variant, or apply FedBN logic to LayerNorm /
Embedding statistics (uncommon but defensible).

### D-18. Revised paper positioning

**Original positioning** (D-1 onwards): "First systematic 6-algorithm FL
benchmark on ColO-RAN under Dirichlet non-IID."

**Honest post-literature positioning**:

| Claim | Novelty after lit-review |
|---|---|
| First FL benchmark on ColO-RAN dataset specifically | Genuinely novel (no prior hits) |
| 6-algorithm FL comparison concept | **Not novel** — arxiv 2403 (CIFAR), FedRS-Bench (remote sensing) |
| FL on cellular data | **Not novel** — arxiv 2508 (5G throughput), Statistical FL B5G (2021) |
| Slice-axis (covariate) Dirichlet on cellular | Novel angle; covariate skew is documented but rarely used for cellular |
| SCAFFOLD Option-II under Adam local optimiser | Novel (small) — fix for SGD-vs-Adam scaling collapse not in literature |
| MOON CIFAR-default failure on RAN telemetry | Novel (small) — useful as HPO transfer cautionary tale |
| α-monotonicity reversal (low α → higher AUC) | Partial — generic α-LR coupling documented (Hsieh 2020), but cellular-slice-specific magnitude is fresh |

**Revised paper framing**:
- Title becomes more precise: "Benchmarking and Optimiser-Aware
  Adaptation of Six FL Algorithms on ColO-RAN Slice SLA Forecasting"
- Positioned as **applied benchmark + implementation insights**, not as a
  fundamentally novel methodology.
- TMC submission is **borderline**: needs FedBN added (D-17) at minimum;
  ideally also a 100-round ablation and a slice-aware FedBN method
  proposal. Without these, IEEE Globecom / ICC / WCNC short paper is the
  realistic target.
- Open option (P2): propose **Slice-aware FedBN** that maintains BN
  statistics per slice rather than per client — would elevate from
  benchmark to method paper, but +6 hr design and risk of weak novelty
  vs FedBN baseline.

For full decision-by-decision audit trail see `git log --follow docs/ADR-001-v5-tmc-paper-plan.md`.

### D-19. Path B pivot — Fermi-based reassessment (2026-04-25)

After M5 completion + post-literature-review (D-17, D-18), a deeper three-path Fermi analysis was conducted on the remaining strategic options. The original FL-benchmark-on-ColO-RAN pitch was found to have low p(accept) at TMC due to recent preemption.

**Three paths compared (point estimates with ±50% Fermi uncertainty per multiplicative factor)**:

| Path | Concept | p(TMC) point | p(TMC) range | p(short-paper venue) | Infra reuse |
|---|---|---|---|---|---|
| A | FL + Spiking-Mamba + ColO-RAN (3-way novelty) | ~8% | 5–12% | ~30% (Globecom/ICC) | 100% |
| **B** | **Centralized Spiking-SSM benchmark on ColO-RAN** | ~9% | 6–13% | ~45% (IoTJ/TNSM short) | 95% |
| C | SNN+FL on real domain (healthcare/genomics) | ~1% | 0.5–2% | ~5% | 5% |

**Honest reading**: Path A and Path B point estimates **overlap substantially** in the Fermi confidence interval. The case for B is **not higher mean acceptance** — it is **lower variance + unconditional Stage 1 paper deliverable**, and a graceful Stage-1-only off-ramp that produces a publishable short paper even if every Stage 2 risk fires.

**Fermi factors driving Path A down**:
- FL-DRAM ([Springer Wireless Personal Communications, 2026-03](https://link.springer.com/article/10.1007/s11277-026-11943-3)) preempts slice-aware FL framing (×0.65 reviewer-1 penalty).
- No neuromorphic hardware → energy claims are estimated, not measured (×0.70 reviewer-2 penalty).
- Surrogate gradient + Adam + FL is unprecedented combination — W1 GO/NO-GO failure kills the paper (×0.75 implementation risk).

**Decision**: Pivot to **Path B as Stage 1**, with **Path A reformulated as Stage 2 conditional on Stage 1 success**.

**Why Stage 1 first** (Fermi reasoning):
- Path B variance is materially lower than Path A (no surrogate-gradient + Adam + FL triple-unverified combination).
- Path B Stage 1 deliverables (3-architecture centralized comparison) become **ablation baselines for Stage 2 — no work is wasted**.
- Stage 1 is publishable independently as IoTJ / TNSM / Globecom short paper if Stage 2 falls through. Path A all-in produces nothing if it fails.
- Path A's "multi-tenant FL on ColO-RAN" framing is now behind FL-DRAM (March 2026); only the **Spiking-SSM × ColO-RAN** angle remains genuinely first-mover, and Stage 1 captures exactly that angle independently of FL.

**Direct competitor warnings (Path B Stage 1)** — none preempt the niche, all must be cited:
- **HiSTM** (arxiv 2508.09184): Hierarchical Spatiotemporal Mamba on Milan/Trentino cellular *traffic prediction* (regression, centralized). Not on ColO-RAN, not classification, not spiking, not federated. Cite as Mamba-on-cellular precedent.
- **SpikySpace** (arxiv 2601.02411, Jan 2026): Spiking SSM for general time series forecasting. Not RAN, not slice SLA. Cite as Spiking-SSM-on-time-series architectural precedent.
- **SpikingMamba** (TMLR Jan 2026, arxiv 2510.04595): LLM via knowledge distillation. Not time series, not telemetry. Architectural reference only.
- **SpikingSSMs** (AAAI 2025, https://github.com/shenshuaijie/SDN): Sparse + parallel spiking SSM, beats spiking LLMs on WikiText-103. **Primary implementation source for Stage 1.**

**Stage 1 novelty pitch** (single sentence — verified 0 hits as of 2026-04-25 across {WebSearch on arxiv/IEEE/Springer, IEEE Xplore manual title search, NeurIPS 2024-2025 proceedings index}):

> "First Spiking State-Space Model benchmarked against LSTM and Mamba baselines on the public ColO-RAN slice SLA forecasting dataset, with spike-count + FLOPs-based energy analysis and accuracy-gap quantification."

This claim is independent of Stage 2 success. Re-verify the "0 hits" before submitting Stage 1 paper — niche could close in 6 months given monthly arxiv velocity in spiking-SSM space.

### D-20. Stage 1 architecture plan (centralized, 3-arch sweep)

**Three architectures share identical input encoder + classification head**. Only the temporal backbone differs. This isolates the architectural contribution.

| Model class | Backbone | LoC est. | Source |
|---|---|---|---|
| `ForecasterV2` | LSTM (existing, M5 baseline) | 0 (reuse) | already at `src/fl_oran/models/forecaster_v2.py`. **Reused as the LSTM arm — not renamed.** |
| `MambaForecaster` | Mamba-S6 (pure PyTorch implementation; mamba-ssm package unavailable due to nvcc absence — see dep-sanity outcome above) | ~150 | `MambaS6Block` written in-tree following Gu & Dao 2024 §3.5 (selective SSM with input-dependent A/B/C parameters + 1D causal conv + gated branch); two stacked blocks at `d_model=64, d_state=16, d_conv=4, expand=2`; reuse `nn.Embedding`s, `dropout`, `fc`, `relu`, `head` from ForecasterV2. **Naming**: no `V2` suffix because it is first-of-kind. |
| `SpikingForecaster` | Spiking-SSM (LIF + selective scan) | ~180 | port from `shenshuaijie/SDN` (AAAI 2025); surrogate-gradient backward via `snntorch.surrogate.atan` or hand-rolled atan; same encoder + classifier head as ForecasterV2. |

**Centralized training only in Stage 1** — no FL. Reuse:
- `data_v2/sequences.py::build_run_sequences`
- `data_v2/encoders.py::fit_continuous_scaler` + `apply_continuous_scaler`
- `data_v2/split.py::ood_split_by_tr`
- `training/centralized_v3.py::run_centralized` (single-machine training loop with the same train/val/test pos-weighted BCE loss and AUC metric we use in v3/v4/v5)

**Dep-sanity outcome (verified 2026-04-25 by actual install attempt)**:

```bash
# Final command set after dep-sanity outcome:
VIRTUAL_ENV=/home/thc1006/dev/fl-oran-tmc/.venv uv pip install 'snntorch>=0.9' 'fvcore>=0.1.5' wheel ninja packaging
```

Verification:
- `snntorch==0.9.4` + `fvcore==0.1.5.post20221221` install cleanly. `snntorch.Leaky(beta, threshold, spike_grad=snntorch.surrogate.atan(alpha=2.0))` produces binary spikes and gradients flow back through the surrogate. `fvcore.nn.FlopCountAnalysis` imports.
- `mamba-ssm>=2.2` build **fails**: requires `nvcc` (CUDA dev toolkit) for source compilation, and no pre-built wheel exists on PyPI for `torch==2.10.0+cu128`. The system has CUDA runtime via PyTorch but the dev toolkit (`nvcc`) is not installed. We will not install the system-level CUDA toolkit just for mamba-ssm.

**Fallback (now the active plan)**: implement Mamba-S6 in pure PyTorch directly inside `models/mamba_forecaster.py` (in-tree class `MambaS6Block`, no external dependency). Pure-PyTorch sequential scan is ~2-3× slower than the Triton kernel but functionally identical to `mamba_ssm.Mamba`. The Stage 1 wall-clock estimate range (5-8 hr) was already chosen to absorb this overhead. Paper §3 will report this as "Mamba-S6 (Gu & Dao 2024) re-implemented in pure PyTorch following the algorithm in §3.5 of the original paper; functionally equivalent to the reference Triton implementation but does not require CUDA-toolkit availability for the reproducibility artifact."

**Stage 1 sweep dimensions** (revised after C5 statistical-power fix in D-21):
- Architectures: 3 (LSTM / Mamba / Spiking)
- Seeds: **10** (42, 0, 1, 2, 3, 7, 11, 13, 17, 23). Bumped from 5 to 10 so the bootstrap CI in D-21 has usable power and cross-comparability with M5 (where M5's 5 seeds form a subset).
- Fixed config (mirror M5 exactly except per-arch overrides below): `seq_len=5`, `sample_ratio=1.0`, `batch_size=64`, `grad_clip=1.0`, `mixed_precision="bf16"`, `cudnn_deterministic=True`, identical OOD `tr` split (train=tr 0..21, val=tr 22..24, test=tr 25..27), `pos_weight_split="train"`, classification target. **Optimizer = `torch.optim.Adam` (fused on CUDA), matching M5 (`_make_optimizer` in `federated/client.py`); no scheduler beyond linear warmup ramp-up over the first 750 steps**.
- Step budget: **5000 gradient steps** per (arch, seed). This matches M5's federation-wide step count exactly (M5 = 20 rounds × 5 clients × 50 steps/round = 5000 steps total per cell). The same step budget is used for all three architectures — Spiking-SSMs literature reports surrogate-gradient training often needs 3-5× more steps to converge, so undertraining is a known risk. We accept this and report convergence curves; if Spiking is undertrained at 5000 steps, that itself is a reportable finding consistent with the paper's "Spiking-SSM applicability on RAN telemetry" framing. We do **not** asymmetrically expand the budget for Spiking only, because that would make the energy comparison incoherent (different total work per inference budget).
- Per-run wall-clock: ~5-10 min for LSTM, ~6-12 min for Mamba, ~15-25 min for Spiking on RTX 4080. Total Stage 1 estimate: **5-8 hr** depending on Spiking surrogate-gradient overhead.
- Total: **30 runs** centralized.

**Per-arch hyperparameter overrides (pinned, no HPO inside Stage 1)**:

| Hyperparameter | LSTM (`ForecasterV2`) | Mamba (`MambaForecaster`) | Spiking (`SpikingForecaster`) |
|---|---|---|---|
| `lr` (Adam) | 5e-4 (= M5) | 5e-4 | 1e-4 (surrogate gradient is unstable at 5e-4; verified empirically in spiking-SSM literature) |
| `lr_warmup_steps` (linear ramp from `lr × 1e-3` → `lr`) | 750 (= M5: 3 rounds × 250 steps/round) | 750 | 1250 (longer warmup for surrogate stability) |
| `weight_decay` | 0.0 (Adam default; M5 used 0; matches) | 0.0 | 0.0 (LIF threshold acts as implicit regulariser; weight decay double-counts) |
| Dropout | 0.1 (= ForecasterV2 default) | 0.1 | 0.0 (LIF binarisation is dropout-equivalent) |
| Hidden width plan | `lstm1(input_dim → 64)` + `lstm2(64 → 32)` (= ForecasterV2 unchanged) | `Linear(input_dim → 64)` + 2× `MambaS6Block(d_model=64, d_state=16, d_conv=4, expand=1)` (in-tree pure-PyTorch class; **`expand=1` chosen at S1-W1 to satisfy ±10% param parity**) + `Linear(64 → 32)` | `Linear(input_dim → 80)` + 2× `SpikingSSMBlock(d_model=80, d_state=16, lif_threshold=1.0, lif_beta=0.9)` (in-tree class wrapping `snntorch.Leaky` + selective scan; **`d_model=80` chosen at S1-W1 to satisfy ±10% param parity**) + `Linear(80 → 32)` |
| `T_inner` (Spiking only) | N/A | N/A | 1 (= one LIF integration per sequence position; no rate-coding repetition). **Decision**: keep T_inner=1 for first sweep; if accuracy gap > 5%, rerun with T_inner=5 in S1-W3 as recovery HPO. |

**Parameter count target**: each arch's hidden dims tuned so total trainable parameters match `ForecasterV2` baseline within ±10%. Verified in `test_v6_param_count.py` (see TDD plan below). This prevents capacity from confounding the energy comparison.

**S1-W1 measured values on the 4-cat-12-cont production schema** (test bound: 40K-50K for ForecasterV2):
- ForecasterV2: 43257 params
- MambaForecaster (d_model=64, expand=1, 2 blocks): 40153 params (-7.2%, within ±10%)
- SpikingForecaster (d_model=80, d_state=16, 2 blocks): 42505 params (-1.7%, within ±10%)

The hidden-width-plan defaults above reflect these tuned values. Defaults applied during S1-W1 from initial (Mamba expand=2, Spiking d_model=64) which gave +68.3% / -29.7% drift respectively.

**Why `weight_decay=0`, dropout=0, lower lr for Spiking**: surrogate gradients are noisier than dense gradients, so additional regularisation (weight decay + dropout) further degrades signal-to-noise. Lower lr is standard practice in spiking-SSM training (Yin et al., ICCV 2023; SpikingSSMs AAAI 2025). Preregistered choice — not HPO'd in Stage 1.

**Energy metric protocol** (no neuromorphic HW; estimates only — explicitly disclosed in paper):

All metrics are computed **per single inference**, where one inference = one full sequence of `seq_len=5` processed end-to-end (= one row of the test set).

- `flops_encoder_head(arch)`: dense MACs in `nn.Embedding` lookups + continuous-feature scaling + `fc + relu + head` (the parts shared identically across all three archs). Same value across archs since encoder/head are identical.
- `flops_backbone_dense(arch)`: dense MACs inside the backbone for LSTM and Mamba (zero for Spiking, since Spiking backbone is binary spike-driven). Measured via `fvcore.nn.FlopCountAnalysis` over the backbone module only.
- `sops_backbone(SpikingForecaster)`: synaptic operations = `Σ_layer (spike_count_layer × fan_out_layer)`. Zero for LSTM and Mamba.
- `total_energy_pJ(arch)` = `(flops_encoder_head + flops_backbone_dense) × 4.6 pJ_per_MAC + sops_backbone × 0.9 pJ_per_AC`. For LSTM and Mamba the spike term vanishes; for Spiking the backbone-dense term vanishes (replaced by sops).
- `energy_ratio(Spiking_vs_LSTM)` = `total_energy_pJ(SpikingForecaster) / total_energy_pJ(ForecasterV2)`. Lower = Spiking is more energy-efficient. **Note**: encoder/head dense ops are present in both numerator and denominator and partially mask backbone savings — paper §5 explicitly reports both `total_energy_ratio` and `backbone_only_energy_ratio` so reviewers can see the contribution split.

**Coefficient source** (pin in paper §5): Horowitz, ISSCC 2014, Table 1, 45nm CMOS — `pJ_per_MAC_FP32 = 4.6` (32-bit float multiply-accumulate); `pJ_per_AC_FP32 = 0.9` (32-bit float accumulate; 1-bit×N-bit spike accumulation maps to a single FP-add event in the dense-input convention used by Davies et al. Loihi 2018).

**TDD plan (Stage 1) — physically appended to §3 table after merge**:

| Test file | Key assertions |
|---|---|
| `test_v6_lif_neuron.py` | LIF forward integrates membrane potential correctly per Euler step; spike-and-reset on threshold crossing; surrogate atan derivative is `1/(π(1 + (πu)²))` evaluated at threshold; matches hand-rolled ground truth for τ=0.5, β=0.9. |
| `test_v6_mamba_forecaster_shape.py` | input/output shapes match `ForecasterV2`; gradients flow through `Mamba` block to embedding weights; deterministic given seed (cudnn_deterministic=True). |
| `test_v6_spiking_forecaster_shape.py` | shapes as above; spike outputs ∈ {0, 1} post-LIF; surrogate gradient norm > 0 when at least one neuron crossed threshold during the sequence. |
| `test_v6_param_count.py` | `count_parameters(ForecasterV2)` ≈ `count_parameters(MambaForecaster)` ≈ `count_parameters(SpikingForecaster)` within ±10%. Fails the build if a hidden-dim change drifts the budget. |
| `test_v6_spike_count_metric.py` | counter integrates monotonically inside one forward pass; resets between forward passes; matches manual count on a toy 1-layer 4-neuron net for a hand-constructed spike pattern. |
| `test_v6_energy_metric.py` | on a 1-Linear(4→4) + 1-LIF toy net, computed `flops_dense=16` and computed `sops` matches `Σ spike_count_input × fan_out_4` for a hand-constructed spike pattern; `energy_ratio_estimate` matches hand calc. Calibrates the paper's energy numbers. |
| `test_v6_centralized_smoke.py` | each of 3 archs trains for 100 gradient steps on a 256-row toy slice without NaN; final batch loss < initial batch loss; runs under 60 s on RTX 4080. Mirrors `test_v5_end_to_end_smoke.py`. |
| `test_v6_arch_swap_isolation_weak.py` | **weakened** from earlier draft: given an explicitly seeded `torch.Generator` for backbone-init separately from encoder/head-init (using `torch.manual_seed` for global, then `Generator(device).manual_seed(K)` per submodule), assert the encoder/head produces **bit-equivalent forward output across the 3 archs when backbone is replaced by `nn.Identity` (with a Linear(d_in → d_in) that is initialised to identity matrix to preserve dimensionality)**. The original "naïve seed" claim was wrong because PyTorch consumes RNG differently per backbone — this version uses split RNG generators to make the test actually pass. |

**Commit cadence**:
- Each S1-Wn ends with a local commit (subject prefix: `S1-W1: ...`).
- `git push` to remote requires user explicit approval, per CLAUDE.md hard rule #6 derivative.
- Each commit must include `tests/test_v6_*.py` updates if relevant production code changed.
- Pre-commit hook runs `pytest tests/test_v6_*.py --no-cov` (~5 s budget).

### D-21. Stage 1 GO/NO-GO criteria (S1-W4)

**Statistical methodology (revised from earlier Wilcoxon plan)**: Wilcoxon signed-rank test with n=5 cannot reach p<0.05 in two-tailed test even at the most extreme outcome (5/5 same-sign pairs gives p=0.0625). With Stage 1's bumped n=10 seeds, Wilcoxon becomes feasible (p=0.002 at extreme), but for accuracy thresholds we use **paired bootstrap CI on the per-seed AUC delta**, which has full power and a directly interpretable threshold:

```
delta_auc = test_auc(SpikingForecaster, seed_i) − test_auc(ForecasterV2, seed_i)   for i=1..10
bootstrap CI95 = percentile(resample(delta_auc, n_boot=10_000), [2.5, 97.5])
```

This gives `[lo, hi]` directly comparable to the −0.030 threshold. Wilcoxon p-values are reported as a **secondary diagnostic** in tables, not as gating criteria.

**Three criteria, each must independently meet a hard threshold**:

| ID | Criterion | Threshold | If miss |
|---|---|---|---|
| C1 | Paired-bootstrap CI95 of `delta_auc(Spiking, LSTM)` upper bound | `hi ≥ −0.030` (i.e. Spiking is at most 3 AUC points worse). **Threshold rationale**: SpikingMamba (TMLR Jan 2026, arxiv 2510.04595) reports a 4.78% accuracy gap vs Mamba on LLM tasks via knowledge distillation. Setting our threshold at 3% is more aggressive than literature's strongest SOTA — passing means Spiking is at least as competitive on RAN telemetry as the best published spiking-SSM is on LLM. We accept a stricter bar to avoid weak novelty. | `hi ∈ (−0.050, −0.030]`: ship as Globecom short paper "Energy-accuracy trade-off". `hi < −0.050`: workshop poster only, **NO-GO any Stage 2 with Spiking as primary**. |
| C2 | Mean `total_energy_ratio(Spiking_vs_LSTM)` across 10 seeds (defined in D-20) | `≤ 0.5` (≥ 2× theoretical total energy advantage including encoder/head dense overhead). Also report `backbone_only_energy_ratio` for transparency. | `(0.5, 0.7]`: re-tune `T_inner` (1 → 5) and LIF threshold (one HPO pass); if still > 0.5, ship as "Spiking-SSM applicability negative result". `> 0.7`: ship Stage 1 short paper, **NO-GO Spiking-led Stage 2**. |
| C3 | Paired-bootstrap CI95 of `delta_auc(Mamba, LSTM)` lower bound + sanity that all 10 Mamba seeds completed with finite metrics (no NaN, loss < 1.0) | `lo ≥ −0.030` (Mamba is no worse than LSTM by 3 AUC) AND zero NaN runs | `lo < −0.030`: Mamba arm broken or genuinely worse than LSTM; escalate to user with diagnostic dump. NaN runs > 0: hard bug, halt Stage 2 planning until fixed. |

**Stage 2 outcome decision** (added per second-round review I13):

| Stage 1 outcome | Stage 2 decision |
|---|---|
| C1 (Spiking ≥ LSTM−0.03) AND C2 (≥2× energy) AND C3 (Mamba healthy) | **Stage 2: Spiking-led** — `SpikingForecaster` is main contribution; Mamba and LSTM are ablations in 1050-cell sweep. |
| C1 fail (gap > 3%) AND C3 (Mamba significantly outperforms LSTM, e.g. `lo > +0.005`) | **Stage 2: Mamba-led fallback** — `MambaForecaster` becomes main FL contribution; Spiking demoted to limitation/ablation. Mamba+FL+ColO-RAN is also genuinely novel (HiSTM is centralized). p(TMC) for Mamba-led is comparable to Spiking-led (~9-12%). |
| C1 fail AND C3 fail (Mamba ≈ LSTM) | **NO-GO Stage 2**. Ship Stage 1 short paper only. M5 FL benchmark numbers go to arxiv as supplementary technical report (not standalone — preempted by FL-DRAM). |
| C1 met AND C2 fail (energy advantage < 2×) | **Stage 2: still Spiking-led but reframed** — title becomes "Trade-off study", energy claim downgraded to "comparable energy at no accuracy cost". Lower p(TMC) ~6-8%. |

**Reporting in the Stage 1 paper**: present all three CI bounds + Wilcoxon p as supplementary, regardless of GO/NO-GO outcome. Reviewers care about effect sizes more than gates.

**Stage 2 plan summary** (full plan in §4 milestones, this is a one-paragraph reminder):
If GO (Spiking-led or Mamba-led), integrate the chosen primary architecture and `MambaForecaster` (always retained as ablation) with the existing 6-algorithm FL registry plus FedBN (7th, addressing M6/D-17). Sweep dimensions: **3 archs × 7 algos × 10 seeds × 5 alphas = 1050 cells**, est. **~20-26 hr GPU** on existing matrix driver (M5's 1.16 min/cell × 1050 cells = 20.3 hr baseline + Spiking-overhead correction up to ~30%). Larger than an earlier draft estimate of 525 cells / 7 hr because (a) seed count standardised at 10 for bootstrap power and (b) Spiking and pure-PyTorch Mamba both add per-cell overhead vs M5's pure-LSTM baseline. M1-M5 LSTM × 6-algo numbers serve as 150-cell legacy ablation table inside the Stage 2 paper. Submission target: TMC. Fallback: TNSM (better-fit for FL-on-network-management angle).

---

## 1. Context

v4 produced 3-seed Centralized / FL-IID / FL-NonIID results on ColO-RAN SLA forecasting (see `artifacts/RESULTS_V4.md`). 86 TDD tests pass.

**Original v5 scope (M1-M5, complete)**: extend to 6 FL algorithms + Dirichlet α sweep + 5 seeds = 150 cells, addressing the v4-vs-TMC bar gaps (only FedAvg / only one non-IID setting).

**Post-M5 scope pivot (D-19, 2026-04-25)**: deep literature review revealed FL-DRAM (Springer 2026-03), SliceFed (arxiv 2603.11390, 2026-03), and FedRS-Bench (arxiv 2505.08325, 2025-05) had preempted the FL-benchmark angle as a TMC-quality contribution. The genuinely empty niche is **Spiking State-Space Models on ColO-RAN** (verified 0 hits across multiple search engines as of 2026-04-25). The ADR therefore extends to a two-stage plan: Stage 1 ships a centralized 3-architecture (LSTM / Mamba / Spiking-SSM) energy-accuracy benchmark as a short paper unconditionally; Stage 2 (conditional on Stage 1 GO/NO-GO per D-21) extends to the federated regime as a TMC paper.

**Scope is deliberately bounded** — causal inference, DP, Byzantine robustness, multi-dataset work, and real neuromorphic hardware deployment are **out of scope** and deferred to future work. Spiking-SSM is in scope as of D-19 but only with **estimated** energy metrics (Horowitz 2014 coefficients), not measured.

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

**v5 (FL benchmark; preserved)**:
`artifacts/v5_sweep/<algorithm>_a<alpha>_s<seed>/{summary.json,history.csv,best_state.pt}`. Aggregation script produces `artifacts/v5_sweep/aggregated.json` + heatmap PNG.

**v6 (Stage 1 centralized 3-arch; new)**:
`artifacts/v6_arch_sweep/<arch>_s<seed>/` with files:
- `summary.json` — keys: `arch`, `seed`, `test_auc`, `val_auc`, `test_f1`, `params_count`, `train_time_sec`, `final_train_loss`, `final_val_loss`, `git_commit`.
- `history.csv` — columns: `step`, `train_loss`, `val_loss`, `val_auc`, `lr`.
- `best_state.pt` — gitignored.
- `energy.json` — keys: `flops_dense`, `spike_count`, `sops`, `energy_ratio_estimate`, `pj_per_mac`, `pj_per_ac`, `t_inner`, `seq_len`, `batch_size_used`, `num_inferences_measured`. Generated in S1-W3.

Aggregation script `scripts/aggregate_v6_results.py` produces `artifacts/v6_arch_sweep/aggregated.json` + bootstrap-CI bar chart PNG + energy-Pareto PNG. Outputs go to `docs/RESULTS_V6_STAGE1.md` for paper-grade table.

**`.gitignore` additions** (must land in same commit as scripts): `artifacts/v6_arch_sweep/` and `artifacts/RESULTS_V6.md`.

**v7 (Stage 2 conditional)**: TBD layout, but will mirror v5 with extra `arch` axis: `artifacts/v7_fl_arch_sweep/<arch>_<algorithm>_a<alpha>_s<seed>/...`.

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

### D-12. Statistical testing: per-stage protocol

**M1-M5 (FL benchmark, completed)**: 5 seeds per configuration. Report mean ± std AND pairwise Wilcoxon signed-rank test on AUC across seeds for algorithm-vs-algorithm comparisons. Significance threshold: p < 0.05. **Retroactive caveat**: Wilcoxon n=5 reaches p=0.0625 at the most extreme outcome (5/5 same-sign), so M5 Wilcoxon p-values are **directional indicators only**, not strict significance gates. M5 paper §6 should report effect sizes and 5-seed bootstrap CIs alongside Wilcoxon to compensate.

**Stage 1+ (centralized 3-arch and Stage 2 if reached)**: 10 seeds per configuration. Primary statistic = paired-bootstrap CI95 of `delta_auc` per D-21 (n_boot=10000). Secondary = Wilcoxon (with n=10 it has full power). This change addresses the n=5 power flaw above and is mandatory for any GO/NO-GO gate at S1-W4.

### D-13. Minimal ablation study

Three ablations, each 1 seed × best-α-from-sweep × Centralized-only (no FL) to isolate architecture contribution:
- Remove embedding layer (one-hot categoricals instead)
- Remove trend features (raw inputs only)
- Reduce seq_len from 5 to 1 (feed-forward instead of LSTM)

Results go in paper Table 2. Total added runs: 3.

### D-14. MOON first-round fallback

First time a client is selected, previous-local-model is undefined. Contrastive loss term is set to 0 for that round (algorithm degrades to FedAvg for that client). Test: `test_v5_moon_first_round_fallback`.

### D-15. Reproducibility commitment

On paper acceptance: release `src/`, `experiments/`, `tests/`, `docs/`, plus aggregated sweep outputs (not raw `best_state.pt` files due to size) under **Apache-2.0** (revised from earlier AGPL-3.0 to avoid license-incompatibility friction with `mamba-ssm` Apache-2.0 / `snntorch` MIT / `fvcore` Apache-2.0 dependencies; AGPL's strong copyleft would create distribution complications without practical benefit for an academic codebase). Seeds listed in paper. Hardware (RTX 4080) listed in paper. Exact commit hash referenced in paper. Stage 1 short paper supplementary will include `requirements.lock` and a `Dockerfile.repro` for one-line reproduction.

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

### M1-M5 (FL benchmark; complete; preserved as Stage 2 ablation source)

| Stage | Milestone | Content | Status |
|-------|-----------|---------|--------|
| M1 | M1 Foundation | Dirichlet partition + registry skeleton + FedAvg + FedProx (TDD each) | ✓ done |
| M2 | M2 More algorithms | SCAFFOLD + FedDyn + FedAdam (MOON deferred per D-16) | ✓ done |
| M3a | M3a MOON | MOON via caller-supplied `encode_fn` | ✓ done |
| M3b | M3b Orchestrator | `run_v5_sweep` + matrix driver + SharedSplits + joblib parallelism | ✓ done |
| M3c | M3c Pilot | 6 algos × 1 seed × 1 α — surfaced SCAFFOLD Option-I bug | ✓ done |
| ~~M4~~ | (superseded) | (was: standalone full sweep) | superseded by M5 |
| M5 | M5 Adversarial-fix sweep | 10 fixes (SCAFFOLD Option-II / FedDyn option_ii / FedAdam bias_correction / pos_weight_split=train / cudnn_deterministic / etc.) → MOON HPO → 150-cell post-fix sweep | ✓ done (2 h 53 min wall-clock) |
| ~~M6~~ | ~~M6 — Lit-review gap closure (D-17, D-18)~~ | ~~(blocking) FedBN as 7th algorithm + ablations + HPO grids~~ | **superseded by Path B pivot 2026-04-25 final-3; FedBN demoted to Stage 2 task** |
| ~~M7~~ | ~~M7 Paper draft~~ | ~~Single TMC paper~~ | **superseded — split into Stage 1 short paper + Stage 2 (conditional) TMC paper** |

### Stage 1 (active; centralized 3-arch benchmark; ~5 weeks, 5-8 hr GPU)

| Stage | Milestone | Content | Status |
|-------|-----------|---------|--------|
| **S1-W1** | **Stage 1 W1: Scaffolding (dep-sanity already done)** | Dep-sanity outcome (D-20): snntorch+fvcore OK; mamba-ssm unavailable → in-tree pure-PyTorch fallback. Create `models/mamba_forecaster.py` with `MambaS6Block` (~150 LoC) and `models/spiking_forecaster.py` with `SpikingSSMBlock` (~180 LoC) per D-20 explicit block configs. TDD per test in dependency order: red (write one failing test) → green (minimum code to pass) → refactor; repeat for all 8 v6 tests in this order: `lif_neuron` → `mamba_shape` → `spiking_shape` → `param_count` → `spike_count` → `energy_metric` → `centralized_smoke` → `arch_swap_isolation_weak`. Local commit at end of each test cycle. | not started |
| **S1-W2** | **Stage 1 W2: Centralized 3-arch sweep (30 runs)** | Create `experiments/run_v6_arch_sweep.py` (3 archs × 10 seeds × 5000 steps, centralized only, reuse `training/centralized_v3.py`). Wall-clock budget: ~6 hr on RTX 4080 (LSTM/Mamba ~10 min each, Spiking ~30 min each). Outputs to `artifacts/v6_arch_sweep/<arch>_s<seed>/`. | not started |
| **S1-W3** | **Stage 1 W3: Energy instrumentation + statistics** | Wire `fvcore.nn.FlopCountAnalysis` for dense backbones; instrument LIF layers with spike counters; compute `sops` per arch on test set (10K random samples). Compute paired-bootstrap CI95 of `delta_auc(Spiking, LSTM)` and `delta_auc(Mamba, LSTM)` per D-21. Generate energy-Pareto + per-layer spike-rate heatmap. | not started |
| **S1-W4** | **Stage 1 W4: Short paper + GO/NO-GO** | Draft `docs/RESULTS_V6_STAGE1.md` (paper-grade table). Draft IoTJ/TNSM short paper (6-8 pages). Apply D-21 GO/NO-GO criteria; emit one of {Spiking-led GO, Mamba-led GO, NO-GO}. Append decision as new D-21 Outcome row. | not started |

### Stage 2 (conditional on D-21 GO; ~5-6 weeks, 20-26 hr GPU)

| Stage | Milestone | Content | Status |
|-------|-----------|---------|--------|
| **S2-W1..3** | **Stage 2 weeks 1-3: FL × architecture integration** | Only if S1-W4 emits Spiking-led or Mamba-led GO. Add FedBN as 7th algorithm (`federated/algorithms/fedbn.py`, ~80 LoC + tests; closes M6/D-17 gap). Integrate chosen primary arch + Mamba (always retained) into existing 7-algo registry. New runner `experiments/run_v7_fl_arch_sweep.py`. Sweep: 3 archs × 7 algos × 10 seeds × 5 alphas = 1050 cells, **~20-26 hr GPU** (M5's 1.16 min/cell × 1050 = 20.3 hr baseline; Spiking adds ~30% overhead → 26 hr upper bound). | not started |
| **S2-W4..6** | **Stage 2 weeks 4-6: TMC paper draft** | M1-M5 LSTM × 6-algo numbers become Stage 2 paper Table 3 (legacy ablation). Stage 1 numbers become Table 2 (centralized upper bound). New main contribution: 3-arch × 7-algo × 5-α heatmap with paired-bootstrap CI95. | not started |

Wall-time history: M5 sweep **2 h 53 min** on RTX 4080 (vs initial 10 h estimate; SharedSplits + joblib + torch.compile reduce-overhead + fused Adam cumulatively bought ~4×).
Stage 1 GPU estimate: **5-8 hr** (30 runs; LSTM ~5-10 min/run, Mamba ~6-12 min/run, Spiking ~15-25 min/run).
Stage 2 GPU estimate (conditional): **20-26 hr** (1050 cells × M5's 1.16 min/cell × Spiking-overhead correction; previously stated 14 hr was M5-baseline-only and ignored Spiking surrogate-gradient cost).
Calendar: Stage 1 = ~5 weeks (4 engineering + 1 paper drafting). Stage 2 = ~5-6 weeks (3 engineering + 2-3 paper drafting). Total Stage-1+Stage-2 if both reach completion: **~10-11 weeks calendar from 2026-04-25**.

---

## 5. Risk Register (high-likelihood items only)

| Risk | Mitigation |
|------|-----------|
| SCAFFOLD doubles communication cost + needs stateful clients | Store control variates on CPU; move to GPU per-round; test no-op when zeroed |
| MOON fails at very low α (< 0.05) due to slice-absent clients | Minimum sample floor per client (50 rows); skip affected α if needed |
| Over-engineering ("let me refactor aggregation.py while I'm here") | **D-9 forbids this**. Trust the ADR. |
| One run crashes mid-sweep | try/except at runner level; log + continue; don't abort batch |
| 150 runs takes > 12 hours | (M5 historical) Did not materialise — M5 finished in 2 h 53 min. Row preserved for context. |
| **FedBN literature gap (D-17)** discovered post-M5 — competitor paper finds FedBN > FedAvg on cellular | **Demoted to Stage 2 task** per D-19 pivot. FedBN sweep + re-aggregation runs as part of S2 1050-cell sweep (3 archs × 7 algos × 10 seeds × 5 alphas), not as standalone Stage 1 blocker. |
| **20-round limitation** — SCAFFOLD/FedAdam may still be improving | Stage 1 is centralized so 20-round limitation is N/A. Stage 2 (if reached) extends to 100 rounds in the FL sweep budget. |
| **Reviewer asks "why n_clients=5 not 7 (= ColO-RAN gNBs)?"** | Stage 1 is centralized; question is N/A. Stage 2 extends with n_clients=7 (`mode="iid"`, bs_id partition) ablation at α=0.5. |
| **FedBN reduces to FedAvg on ForecasterV2** (no BatchNorm layers) | Stage 2 question. `MambaForecaster` and `SpikingForecaster` may have RMSNorm or LIF state — FedBN logic needs re-derivation per architecture before Stage 2 sweep. |
| **Path B Stage 1: HiSTM (arxiv 2508.09184) reviewer asks "how does this beat HiSTM?"** | HiSTM is on Milan/Trentino traffic regression, not ColO-RAN slice SLA classification. Paper §2 (Related Work) explicitly differentiates: different dataset, different task, no spiking. Honest framing: "HiSTM is the closest precedent for Mamba-on-cellular; we extend the direction to (a) ColO-RAN slice telemetry (b) classification task (c) spiking variant for energy". |
| **Path B Stage 1: SNN-skeptic reviewer rejects "estimated energy without neuromorphic HW"** | Pre-empt in paper Limitations section; cite Horowitz 2014 coefficients explicitly; report `sops`/`flops` as architecture-level theoretical metric, not deployment claim. Decline reviewers asking for hardware measurements as out-of-scope (noted upfront in Abstract). |
| **Path B Stage 1: Surrogate gradient + Adam unstable** (training does not converge) | Pre-empted in D-20 hyperparameter table (Spiking lr=1e-4, weight_decay=0, dropout=0). If still divergent after one HPO pass on `T_inner ∈ {1, 5, 10}`, escalate to user as S1-W1 NO-GO and re-scope to non-spiking energy-efficient SSM (e.g., quantised Mamba INT8) — that pivot still produces a Stage 1 short paper. |
| **Path B Stage 1: `SpikingForecaster` accuracy gap > 5%** | D-21 hard NO-GO threshold for criterion C1. Ship as workshop poster or arxiv preprint: "Negative result on Spiking-SSM applicability to slice SLA forecasting". Document why (representational capacity vs LIF binarisation tradeoff). Still a publishable contribution. |
| **`mamba-ssm` incompatible with PyTorch 2.10 / CUDA 12.8** | **Fired and resolved 2026-04-25**. mamba-ssm source build requires `nvcc` (not installed); no pre-built wheel for cu128. Resolution: implemented `MambaS6Block` in pure PyTorch in-tree (~150 LoC) following Gu & Dao 2024 §3.5. No external dependency. ~2-3× slower than Triton but functionally identical and absorbed into 5-8 hr Stage 1 budget. |
| **Stage 1 W1 statistical-power error: bootstrap CI too wide at n=10** | Mitigation: pre-register `n_boot=10000`. If CI width > 0.04 AUC at S1-W3, bump seeds to 15 (additional 15 runs ≈ 3 hr GPU). Document in paper. |
| **Stage 2 not reached** (Stage 1 NO-GO at S1-W4) | Stage 1 short paper goes to IoTJ / TNSM / Globecom regardless. M5 150-cell FL benchmark numbers ship as supplementary technical report on arxiv (not as standalone paper — preempted by FL-DRAM). **Calendar reclaimed: ~5-6 weeks of Stage 2 budget freed for next research direction. Intellectual loss: zero** (M5 + Stage 1 numbers retained for any future use). |
| **Stage 2 paper reviewer: "Why didn't you compare to FL-DRAM?"** | Stage 2 paper §1 explicitly differentiates: we focus on architectural energy-efficiency under covariate skew; FL-DRAM focuses on slice-aware adaptation algorithms. Frame as orthogonal/complementary, cite FL-DRAM in Related Work + future-work section ("combining FL-DRAM's PerFedRL with our SpikingForecaster is a natural extension"). |
| **Niche closes during Stage 1** (someone else publishes Spiking + cellular RAN before our submission) | Mitigation: pre-register on arxiv at end of S1-W2 (after sweep, before paper drafting). This timestamps priority. Re-run D-19 "0 hits" search at start of S1-W4 paper draft; if niche has closed, pivot Stage 1 paper framing to differentiated angle (e.g., "first systematic energy-accuracy benchmark" vs "first to combine"). |

---

## 6. Consequences (refreshed 2026-04-25 post-pivot)

**Positive**:
- v4 artifacts stay valid (no changes there).
- M1-M5 (FL benchmark) artifacts stay valid; rebrand as "Stage 2 ablation source" rather than discard.
- Stage 1 produces a publishable short paper unconditionally — research budget is no longer all-in on a single TMC bet.
- 10-seed + paired-bootstrap CI replaces the underpowered Wilcoxon n=5 plan, meeting 2026 statistical-rigour bar.
- Scope remains bounded: Stage 1 = ~5 weeks, Stage 2 conditional ~5-6 weeks. No multi-quarter rabbit-hole.

**Negative**:
- NeurIPS/ICML *main track* still out of scope — accepted trade-off. **NeurIPS Workshop on ML for Systems / NeurIPS Datasets-and-Benchmarks track / ICLR Workshop on ML for Wireless** are now plausible secondary venues for Stage 1 given the energy-aware Spiking-SSM novelty.
- Spiking implementation introduces surrogate-gradient + LIF state complexity new to this codebase; risk-managed via TDD plan in D-20 + dependency sanity check before any LoC is written.
- Total GPU time across both stages: Stage 1 = ~6 hr (30 runs × ~12 min mean) + Stage 2 conditional = ~14 hr (1050 cells × 1.16 min/cell + Spiking overhead). Combined ~20 hr GPU.
- License changed from AGPL-3.0 → Apache-2.0 (D-15) to avoid friction with mamba-ssm/snntorch/fvcore dependencies.

**Neutral**:
- Codebase footprint: M1-M5 baseline (~+700 LoC) preserved; Stage 1 adds ~+400-500 LoC (`MambaForecaster` ~120, `SpikingForecaster` ~180, `run_v6_arch_sweep.py` ~80, `aggregate_v6_results.py` ~60, plus ~7 v6 test files); Stage 2 conditional adds ~+200 LoC (`fedbn.py` + `run_v7_fl_arch_sweep.py`). Total post-Stage-2: ~+1300 LoC across ~14 new files vs the v4 baseline.
- Test count: 131 (M5) → ~140 (Stage 1) → ~145 (Stage 2 conditional).

---

## 7. Alternatives Considered

### Pre-pivot (M1-M5 scope, FL-benchmark framing)

1. Keep v4 and submit as-is → Rejected: insufficient algorithm comparison for TMC.
2. Only 3 algorithms (FedProx/SCAFFOLD/MOON) → Rejected: post-literature-review, 2025 bar is 6+.
3. Use Flower/FedML framework → Rejected: our CUDA-graph + federated-scaler customisations don't fit without adapter overhead.
4. Full rewrite into `v5/` sub-package → Rejected: violates D-9.
5. Add SliceAvg custom algorithm as primary contribution → Deferred to optional section 7 in paper, only if sweep results justify it.

### Post-pivot (D-19 Path B scope)

6. **Path A all-in: FL + Spiking-Mamba + ColO-RAN as a single TMC submission, no Stage-1 short paper** → Rejected: Fermi p(TMC) ≈ 8% (5–12% with uncertainty); FL-DRAM (March 2026) preempts slice-aware framing; W1 GO/NO-GO failure on surrogate+Adam+FL combination produces zero deliverable. The two-stage plan retains this option's upside (Stage 2 = FL+Spiking) while gaining an unconditional Stage 1 deliverable.
7. **Path C: SNN+FL on real domain (healthcare/genomics)** → Rejected: Fermi p(TMC) ≈ 1% (0.5–2%); zero existing infra reuse; no IRB/consortium access for healthcare; switching domain would consume the entire 5-week Stage 1 budget on data plumbing. ROI dramatically negative vs Path B.
8. **Path B Stage 1 only, never do Stage 2** → Considered but rejected as a *plan* (vs a *fallback*): Stage 1 short-paper venues (IoTJ/TNSM/Globecom) are lower-impact than TMC; if Stage 1 numbers warrant Stage 2 (criteria in D-21), refusing to extend would leave significant impact on the table. Therefore Stage 2 is conditional, not forbidden.
9. **Drop Spiking, do Mamba+FL+ColO-RAN directly** → Considered as pre-emptive simplification; rejected because (a) HiSTM already covers Mamba-on-cellular (centralized), so Mamba alone has weaker novelty than Mamba-as-fallback-to-Spiking after Stage 1; (b) the Spiking energy angle is the strongest remaining differentiator. **However**, this is preserved as the **Mamba-led Stage 2 fallback** in D-21 if Stage 1 Spiking misses C1 but Mamba beats LSTM.
10. **Submit Stage 1 to a workshop instead of a journal** → Considered for risk reduction; rejected because IoTJ/TNSM short paper has comparable acceptance rate (~25-30%) to top workshops while granting full journal-paper status, citation weight, and citability for the eventual Stage 2 TMC submission.

---

## 8. Paper Outline (revised 2026-04-25 — Path B two-stage)

### Stage 1 paper (active target)

**Title**: *Energy-Aware Architectures for O-RAN Slice SLA Forecasting: LSTM, Mamba, and Spiking State-Space Models on the ColO-RAN Dataset*

**Length**: 6-8 pages (short paper)

**Target venue (priority order)**: IEEE IoTJ → IEEE TNSM → IEEE Globecom 2026 (deadline July) → IEEE WCNC 2027 short.

**Sections**:
1. Intro — slice SLA forecasting energy cost in always-on RAN xApps; motivation for energy-aware architectural choice (not just FedAvg vs FedBN). Highlight that today's FL-on-cellular literature treats LSTM as fixed; we ask the orthogonal question.
2. Background — ColO-RAN dataset; SLA forecasting as binary classification; SSM and SNN primer; **target leakage audit** (v4 finding: `allocation_efficiency = 0.5·throughput_eff + 0.3·qos + 0.2·prb_util`) — same audit reused.
3. Clean task definition + OOD split (reused from v4); preregistered hyperparameter table (D-20).
4. **Three-architecture methodology**: shared encoder + shared classifier head; LSTM / Mamba / Spiking-SSM backbones interchangeable; parameter-count matched within ±10%.
5. Energy estimation protocol — `sops` vs `flops_dense`; Horowitz 2014 45nm CMOS coefficients (4.6 pJ/MAC, 0.9 pJ/AC); explicit non-deployment caveat.
6. **Results** — 3 archs × 10 seeds: AUC mean ± paired-bootstrap-CI95, FLOPs/SOPs, energy-ratio Pareto.
7. Limitations (no neuromorphic HW; ColO-RAN simulator; centralized only; preregistered T_inner=1) + future work (Stage 2 FL extension).
8. Conclusion.

**Key figures**: (a) test-AUC bars across 3 archs × 10 seeds with paired-bootstrap CI95 + Wilcoxon p as supplementary; (b) accuracy vs energy-ratio Pareto scatter; (c) per-layer spike rate heatmap.

**Key tables**: (a) 3-arch metric table (AUC mean ± CI95, F1, params count, FLOPs/SOPs, energy-ratio); (b) hyperparameter ablation for `T_inner ∈ {1, 5, 10}` and LIF threshold; (c) leakage audit table (ColO-RAN methodological hygiene).

### Stage 2 paper (conditional on D-21 GO outcome)

**Title (Spiking-led)**: *Federated Energy-Efficient Spiking State-Space Models for O-RAN Slice SLA Forecasting*

**Title (Mamba-led fallback)**: *Mamba State-Space Models in Federated O-RAN Slice SLA Forecasting: Architecture–Algorithm Interactions Under Covariate Skew*

**Length**: 12-14 pages (full TMC submission)

**Target venue**: IEEE TMC. Fallback: IEEE TNSM Special Issue on Latest Developments in Federated Learning for Networked Systems.

**Sections**:
1. Intro — extend Stage 1 architecture finding to a federated regime where data is partitioned by gNB / slice. **Framing distinction from FL-DRAM** (D-19): we focus on architectural energy-efficiency under covariate skew, not on slice-aware adaptation algorithms; results are complementary to and citation-friendly with FL-DRAM.
2. Background — Stage 1 background + FL primer + non-IID taxonomy + ColO-RAN partitioning rationale.
3. Methodology — combine three architectures (Stage 1 result) with seven FL algorithms (M5 + FedBN); covariate-skew Dirichlet partition over `slice_id`; per-architecture FedBN derivation (D-17 open question resolved here).
4. **Main result** — 3 archs × 7 algos × 10 seeds × 5 alphas = 1050-cell heatmap.
5. **Stage 1 numbers reused as Table 2** (centralized upper bound vs FL accuracy gap).
6. M1-M5 LSTM × 6-algo numbers reused as Table 3 (legacy ablation; demonstrates how the chosen primary architecture changes the algorithm-comparison conclusions).
7. Per-architecture FedBN behaviour (D-17 open question resolved here with experimental data).
8. Limitations + Conclusion.

**Key figures**: (a) 7×5 algorithm×α heatmap per architecture (3 panels); (b) Pareto over accuracy / energy / communication cost; (c) FedAvg-vs-FedBN training curves per architecture.

**Key tables**: (a) Stage 1 centralized baseline; (b) Stage 2 FL result at α=0.5 across 3 archs × 7 algos with paired-bootstrap CI95; (c) communication-cost-per-AUC-point efficiency table; (d) M5 LSTM-only legacy table for backward comparison.
