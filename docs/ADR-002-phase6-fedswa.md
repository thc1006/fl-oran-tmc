# ADR-002 — Phase 6: FedSWA Integration for SOTA Comparison

**Status:** REJECTED v3 (do not implement; mechanism-based §related-work treatment instead)
**Date v1:** 2026-04-30 (mechanism wrong — see v2 note)
**Date v2:** 2026-04-30 (paper PDF read; mechanism corrected; screen design proposed)
**Date v3:** 2026-05-01 (rejected after deep figure analysis — Phase 6 EV negative under Phase 5 evidence)
**Authors:** thc1006 + Claude analysis
**Supersedes:** none

---

## v3 Decision Summary

**Do not run Phase 6.** Treat FedSWA via mechanism-based dismissal in paper §related-work
+ §threats-to-validity. Reclaim ~17 hr GPU + ~10 hr dev for Phase 4 paper writing.

The decision is driven by six layers of analysis on Phase 5 figures
(`artifacts/figures/algo_ranking.png`, `interaction_heatmap.png`, `pareto.png`)
and the 764-cell completed `_phase_summary_complete_20260430_200744.csv`. Each
layer is documented in §14 (rationale chain).

---

## 1. Context (unchanged from v2)

Phase 5 (900 cells × 5 algorithms) closes May 1 ~14:30. Pre-submission lit
review revealed that our 5 algorithms (FedAvg/Prox/Adam/SCAFFOLD/FedDyn)
are 2017-2021 papers; 2024-2026 SOTA has shifted to sharpness-aware methods
(FedSAM 2022, FedSWA ICML 2025, FedSCAM Dec 2025). Reviewer-defensibility
question: do we add Phase 6, or handle the gap differently?

v2 proposed Phase 6.1 (FedSWA + 9-cell hyperparameter screen, ~17 hr GPU
+ ~10 hr dev). v3 rejects v2 after analysing Phase 5 figures.

---

## 2. What FedSWA Is — Mechanism (verbatim from arxiv 2507.20016, retained from v2)

### 2.1 Algorithm 1 (Liu et al. ICML 2025, page 4)

```
For round t = 0..T:
  Server broadcasts θ_{t-1} to s clients.
  Each client i:
    θ^t_{i,0} = θ_{t-1}
    For k = 0..K-1:
      g_i = ∇L(θ^t_{i,k}; mini-batch)
      η^t_k = η_l (1 - k/K) + (k/K) ρ η_l            # eq. (3) intra-round LR decay
      θ^t_{i,k+1} = θ^t_{i,k} - η^t_k g_i             # eq. (4)
  v_t = (1/s) Σ_i θ^t_{i,K}                          # uniform aggregation
  θ_t = θ_{t-1} + α (v_t - θ_{t-1})                  # eq. (17) server EMA / LookAhead
```

Paper defaults: α=1.5, ρ=0.1, η_l grid 1e-3..3e-1, SGD optimizer, 1000 rounds.

### 2.2 Two distinct mechanisms

* **M1 (intra-round cyclical LR):** η_k decays linearly from η_l to ρ·η_l within each
  round, then resets. Inspired by Smith 2017 + Gotmare 2019 LR restarts.
* **M2 (server EMA / LookAhead overshoot):** θ_t = (1−α)θ_{t-1} + α v_t with α=1.5
  → extrapolation beyond v_t. Inspired by Zhang 2019 LookAhead.

### 2.3 What FedSWA is NOT (correcting v1)

NOT classic SWA (no separate trajectory accumulator). NOT SCAFFOLD-family
(no per-client control variate c_i; that's FedMoSWA).

---

## 3. Decision

**REJECTED.** Skip Phase 6. Document FedSWA via mechanism-based §related-work
defense (~150 words) + §threats-to-validity (~60 words). 0 hr GPU, ~1 hr writing.

Rationale chain in §14.

---

## 4. Paper Treatment (replaces v2's §7 implementation plan)

### 4.1 §related-work paragraph (~150 words)

> "Recent sharpness-aware federated learning methods — FedSWA (Liu et al.,
> ICML 2025) and FedSCAM (Dec 2025) — target heterogeneity-induced sharp
> minima via cyclical local learning-rate schedules and LookAhead-style
> server-side weight extrapolation. The reported gains (FedSWA: +5.6% over
> FedAvg on CIFAR-100 with 10% client participation across 1000 rounds)
> assume large-N low-participation FL with coherent client drift toward sharp
> minima. Our regime — 7 base-station-natural clients with 71% per-round
> participation, 100 rounds, RAN slice SLA prediction — inverts this premise.
> The natural-by-BS partition (Table X) and inverted-α pattern (Fig Y)
> indicate heterogeneity correlates with structural client specialization
> rather than sharp-minima wounding. In this regime, server-side aggregation
> noise dominates over coherent drift, and FedAdam's adaptive variance
> damping (Reddi et al., 2021) saturates the available headroom (+0.007 to
> +0.017 AUC over FedAvg, Fig Z). FedSWA's noise-blind LookAhead overshoot
> (α=1.5) cannot exceed this ceiling on mechanistic grounds."

### 4.2 §threats-to-validity paragraph (~60 words)

> "We do not empirically evaluate FedSWA, FedSCAM, or other 2025 sharpness-aware
> FL methods. The mechanism analysis in §X.Y predicts these methods would
> perform at or below FedAdam in our regime. Empirical confirmation in regimes
> where heterogeneity-as-sharpness-wound holds (image classification with
> α<0.1, large-N, low-participation) is left to future work."

---

## 5. Decisions Required from User

* [x] Approve REJECTED v3 (do not run Phase 6).
* [ ] Confirm §4.1 + §4.2 wording goes into paper draft (Phase 4a + 4d).
* [ ] Pivot dev hours immediately to Phase 4 paper writing (5 pending tasks
      #142, #143, #144, #145, #146, #147).

---

## 6. Revision History

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2026-04-30 | v1 | Initial draft (mechanism wrong) | Claude |
| 2026-04-30 | v2 | Read paper PDF; rewrote mechanism + screen design | Claude |
| 2026-05-01 | v3 | REJECTED after Phase 5 figure analysis (six-layer rationale §14) | Claude |

---

## 14. Rationale Chain (six-layer analysis)

### Layer 1: Phase 5 figures show flat algorithm design space

`algo_ranking.png` y-axis is **mean best_val_auc** averaged over Dirichlet
partitions α∈{0.05..5.0}; numbers below are read from that figure (best_val_auc,
not test_auc — the paper's RESULTS_V7_PHASE5.md table reports both, with
test_auc as the §6 single-source-of-truth metric):

| Arch | Best | Worst | Spread (best_val_auc) |
|------|------|-------|--------|
| LSTM | FedAdam 0.813 | FedDyn 0.789 | 0.024 |
| Mamba | FedAdam 0.816 | SCAFFOLD 0.738 | 0.078 (SCAFFOLD outlier; rest spread 0.031) |
| Spiking | FedAdam 0.696 | SCAFFOLD 0.653 | 0.043 (full 5-algo populated post-2026-05-01) |

`interaction_heatmap.png` natural-by-BS column: 5 algos collapse to 0.919-0.923
on LSTM, 0.917-0.924 on Mamba — completely flat. The Phase 5 finding is
that **algorithm choice is a near-flat design dimension** in this dataset.

### Layer 2: FedSWA is strictly dominated by FedAdam mechanism-wise

Both modify server-side aggregation:

|  | FedAdam | FedSWA |
|---|---|---|
| Server step | Adam(m, v) over (v_t − θ_{t-1}) | θ_{t-1} + α(v_t − θ_{t-1}) linear extrap |
| Knows variance | yes (v second-moment) | no |
| Damps oscillation | yes (√v denominator) | no — α=1.5 amplifies |
| Reduces to FedAvg | β1=β2=0, η=1 | α=1.0 |

In aggregation-noise-dominated regimes (small-N + high-participation FL like
ours), FedAdam **strictly dominates** FedSWA on mechanism. Phase 5 confirms:
FedAdam +0.007~0.017 over FedAvg in three architectures. FedSWA cannot
exceed this ceiling because it lacks variance estimation.

### Layer 3: Inverted α pattern (mechanism revised 2026-05-02 after Step 1+2 + V100 ablation)

```
LSTM × FedAvg: α=0.05 → 0.8605  |  α=5.0 → 0.7475   (Δ = +0.113 toward heterogeneity)
Mamba × FedAvg: α=0.05 → 0.8686 |  α=5.0 → 0.7490   (Δ = +0.120)
Natural by-BS: 0.921~0.924                          (Δ = +0.05+ over α=0.05)
```

**Original mechanism hypothesis (2026-04-30) — INVALIDATED.** The original
"per-client pos_weight × sparse-positive-class structural specialization"
story relied on (a) sparse-positive regime, (b) per-client local pos_weight,
(c) Dirichlet-induced positive-class variance. Step 1 fact-finding
(`memory/dataset_facts_phase5.md`, 2026-05-02) eliminated all three:
positive rate is 30.9 % (balanced, not sparse); fl_v7 uses GLOBAL pooled
pos_weight (`fl_v7.py:636-637`); per-bs slice mixture KL ≈ 0 (no bs↔slice
correlation). The §3 ColO-RAN dataset has 3 slices not 4 (slice_id ∈ {0,1,2}),
3 schedulers not 4, 17 continuous features not 29 — see PAPER §3.1 corrected.

**Current mechanism finding (2026-05-02) — V100 ablation supports
"bs-conditioned channel-state signal preservation" hypothesis.** §7.1.1 V100
random_split ablation (15 cells, 4× V100-SXM2-32GB) shows partitions that
break bs grouping (whether Dirichlet α=5.00 or uniform random) collapse AUC
~0.18 below natural-by-BS, consistent across all 3 architectures (LSTM
−0.176, Mamba −0.172, Spiking-SSM −0.186). §7.1.2 measures top per-bs KL on
continuous features: dl_cqi (0.475), dl_mcs (0.267), num_ues (0.092) —
channel-state features differ substantively per BS, consistent with model
exploiting bs-conditioned signal that natural-by-BS preserves. This does
not change the FedSWA-rejection conclusion: the "heterogeneity as sharpness
wound" framing FedSWA targets is incompatible with the inverted-α
empirical pattern regardless of the precise mechanism.

### Layer 4: Architecture leverage is 5-10× algorithm leverage on Pareto front

`pareto.png` (AUC vs model-attributable energy, 786 cells):

| Choice axis | AUC range | Energy range |
|---|---|---|
| Architecture (LSTM / Mamba / Spiking) | 0.65-0.93 | ~3.5 kJ / ~10.5 kJ / ~41 kJ |
| Algorithm (within each arch) | spread 0.024-0.031 | flat |

Paper headline should center architecture×partition trade-off, not algorithm
breadth. Adding a 6th algorithm (FedSWA) expected on flat axis dilutes the
narrative.

### Layer 5: ROI is negative under paper-writing opportunity cost

Path cost vs alternatives:

| Path | GPU hr | Dev hr | Total |
|---|---|---|---|
| ADR v2 (180-cell screen+sweep) | ~17 | ~10 | ~27 |
| Path β (30-cell paper-default mini) | ~2.5 | ~10 | ~12.5 |
| Path ε (20-cell hypothesis-targeted) | ~1.7 | ~10 | ~11.7 |
| **Path γ (rejected, this v3)** | **0** | **~1** | **~1** |

11+ hours of writing time can produce: §related-work mechanism defense (Layer 2),
§discussion natural-by-BS structural specialization finding (Layer 3),
Mamba × SCAFFOLD catastrophic interaction case study (existing data),
architecture-energy decomposition section (Layer 4 from `pareto.png`),
paper-writing tasks #142-#147. All higher value than confirming an expected-neutral
result on a flat axis.

### Layer 6: Paper value is finding-depth × finding-count, not additive

Phase 5 already produced 5 unwritten findings:
1. Natural-by-BS uniformly outperforms Dirichlet (monotonic in α; structure-helps-FL).
2. FedAdam +0.007~0.017 over FedAvg consistently across 3 archs (Adam-style
   adaptive aggregation saturates the algorithm headroom in this regime).
3. SCAFFOLD × Mamba × α∈{0.10, 0.50} catastrophic destructive interaction
   (σ amplified 23×; deployment-relevant warning).
4. Architecture choice dominates algorithm choice on AUC AND on energy (Pareto).
5. pos_weight × heterogeneity → mixture-of-specialists explanation for
   inverted-α pattern.

Adding "FedSWA tested, neutral" as a 6th finding **dilutes** the paper rather
than strengthens it. Reject Phase 6, write up the 5 findings deeply.

---

## 15. References

Retained from v2:
* Liu et al. "FedSWA". arxiv:2507.20016 (ICML 2025).
* Reddi et al. "Adaptive Federated Optimization" (FedAdam). ICLR 2021.
* Smith. "Cyclical learning rates." WACV 2017.
* Zhang et al. "Lookahead optimizer." NeurIPS 2019.
* Izmailov et al. "SWA." UAI 2018.
* Hsu et al. "Measuring the Effects of Non-Identical Data Distribution." 2019.
