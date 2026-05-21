# ADR-003 — "Conditional Structure, Not Distributional Skew": Mechanism + Reframe Research Program (Paper A)

**Status:** Proposed — program accepted by user 2026-05-21. Phase 1 (mechanism quantification, see `docs/PREREG-A1-mechanism.md`) is **documented but NOT executed**; any training run requires separate explicit user authorization (CLAUDE.md hard rule #6).
**Date:** 2026-05-21
**Authors:** thc1006 + Claude analysis
**Supersedes:** none.
**Builds on:** the JSAC benchmark paper (`paper/main.tex`, tag `v0.9.4-preprint`) and ADR-001 (Stage 1/2 program). Companion pre-registration: `docs/PREREG-A1-mechanism.md`.

---

## 0. Decision summary

Pursue a new research program — **Paper A** — that converts the JSAC benchmark's *empirical phenomenon* (the natural base-station partition uniformly outperforms every Dirichlet partition across all architectures and algorithms; "inverted heterogeneity") into a **pre-registered, falsifiable mechanistic claim** plus a **reframe** of what data heterogeneity *means* in physically-grounded federated learning.

**Thesis — "Conditional structure, not distributional skew."** FL theory was built on vision/NLP, where client heterogeneity is *distributional skew* over a **shared** concept `P(y|x)` — a nuisance to be mitigated (SCAFFOLD/FedDyn/FedProx; or engineered away toward IID). In RAN telemetry the conditional `P(SLA|KPI)` is **itself cell-conditioned by radio physics** (propagation and scheduling are cell-specific): the same KPI vector implies different SLA outcomes at different base stations. The heterogeneity across base stations is therefore **concept structure that is part of the predictive signal**, not nuisance skew. The natural per-cell partition *preserves* this structure inside each local update; synthetic IID re-partitioning *destroys* it. Hence the natural partition wins **because** it aligns clients with the true conditional structure, and the standard heterogeneity-robustness toolkit is solving the wrong problem for O-RAN.

Paper A is a **different kind of contribution** from the JSAC benchmark (see §4). A safe-offline-xApp-recommendation application (**Paper B**) follows it, gated by a coverage probe; Paper B is out of scope here except as a downstream dependency (§6).

This ADR records: the falsifiable hypotheses (§2), the claim ceiling (§3), the JSAC differentiation (§4, user-mandated), the positioning/citation decisions (§5), the program structure and gates (§6), and the kill criteria summary (§7). Exact experiments, metrics, output figures, and pre-registered thresholds live in PREREG-A1.

---

## 1. Context — verified May-2026 literature audit

The phenomenon (from the JSAC paper, `docs/RESULTS_V7_PHASE5.md`): natural-by-BS > all Dirichlet α, uniformly across {LSTM, Mamba, Spiking-SSM, xLSTM, Mamba-3} × {FedAvg, FedProx, FedAdam, SCAFFOLD, FedDyn} on OOD-by-`tr` test; architecture dominates algorithm; Mamba×SCAFFOLD catastrophic interaction at αdir∈{0.10,0.50}.

A multi-round web audit (2026-05-21, ≥5 corroborating sources per load-bearing claim) established:

- **The field treats non-IID as a problem to mitigate**, not a benefit. Canonical foil: NIID-Bench (arXiv 2102.02079) + multiple surveys (arXiv 2411.12377; MDPI 2224-2708/14/2/37; assessment studies arXiv 2503.17070, 2502.00182). ≥6 sources, high confidence.
- **An "engineer toward IID" school exists** — e.g. "Geographical Node Clustering to Guarantee Data IIDness in FL" (arXiv 2410.15693) groups clients *to achieve* IID-ness. This is the **direct reframe target / foil**.
- **PARTIAL PRECEDENT (must cite + differentiate):** "Federated Learning for 5G Base Station Traffic Forecasting" (arXiv 2211.15220) reports that, in real base-station (naturally non-IID) FL, **SOTA non-IID-robust aggregators do not outperform simple FedAvg**, and local preprocessing matters more. This overlaps with our "architecture dominates algorithm / robustness toolkit does not pay off" finding. **"Aggregators don't help in wireless FL" is therefore NOT our novel claim.**
- **The mechanism (client-conditional structure ⇒ cluster/personalize) is established** in clustered/personalized FL (Frontiers 2026 clustered-FL-wireless; prototype-based CFL; concept-drift FL FedPLC; FedBN arXiv 2102.07623). The bare mechanism is **not novel**; our contribution is the RAN-physics grounding + the *global-model* effect + cross-data generality + the reframe (§2, §4).
- **No public multi-BS RAN control-plane dataset matching ColO-RAN's schema exists** (client-side KPI datasets have the wrong schema; the Commercial Traffic Twinning dataset, arXiv 2409.16217, is 1 gNB / 8 UE / 2 slices). This sets the claim ceiling (§3).
- **TimeRAN (arXiv 2604.04271, 2026)** is a RAN time-series foundation model but is FL-agnostic (no partition/heterogeneity analysis) → a baseline, not a competitor.
- **Offline RL for O-RAN slicing is active since 2023** (arXiv 2312.10547; CQL selection arXiv 2603.03932) → relevant only to Paper B's differentiation.

**Critical caution.** The mechanism behind this phenomenon was **falsified once already**: the original §7.1 narrative (sparse-positive labels + per-client `pos_weight` + bs↔slice correlation) was invalidated by the Step-1/2 measurement (see `memory/dataset_facts_phase5.md`). A paper that *explains* the phenomenon therefore carries a high bar: the mechanistic claim must be **pre-registered and falsifiable** before any analysis is run, to prevent post-hoc story-fitting.

---

## 2. Thesis as falsifiable hypotheses

The thesis (§0) is decomposed into three pre-registered, falsifiable hypotheses. Operational tests and thresholds are in PREREG-A1 (E1–E4).

- **H1 — Concept shift dominates.** RAN base-station heterogeneity is primarily *concept shift* (`P_bs(SLA|KPI)` differs across bs at matched covariates), not merely *covariate shift* (`P_bs(KPI)` differs while `P(SLA|KPI)` is shared). *Falsified if* concept-shift magnitude at matched covariates is not significantly above a pre-registered floor (→ it is the standard covariate-shift non-IID, thesis rejected).
- **H2 — Mechanism: coherent supervision of a cell-indexed function.** Because `bs_id` is a model **input** (**verified in code:** embedded categorical, `forecaster_v2.py` / `V3_CATEGORICAL`), the global FedAvg model fits a cell-*indexed* conditional `P(SLA|KPI, bs)`. The natural partition gives each client *coherent* per-cell local updates → the aggregated global model absorbs the mixture-of-conditionals cleanly; Dirichlet/shuffle gives cell-blurred local updates → a worse global model. *Decisive test:* the **3-arm cell-conditioning ablation** (E2: no-bs / explicit-bs / bs-shuffled, via the existing `drop_categorical` lever) separates "cell as input index," "true cell identity vs embedding capacity," and "intrinsic channel structure." This is the **primary falsification gate** (§7); only E2 showing all arms tie (cell conditioning provides nothing) falsifies the program.
- **H3 — Traffic-invariant transfer.** The cell-conditional radio structure is `tr`-(traffic-config-)invariant; the natural-partition model learns it and transfers to unseen `tr` (the OOD test axis), whereas the shuffled model entangles `tr` with cell and transfers worse. *Predicts:* the natural-partition advantage grows with the train/test `tr`-distribution gap.

This resolves the **breakthrough puzzle** that makes Paper A more than a restatement of clustered FL: standard FL theory says concept shift *hurts* a single global model, yet here the global model trained on the concept-shifted natural partition generalizes *better*. H2 is the proposed resolution and the intellectual core.

---

## 3. Scope & claim ceiling (honest, pre-committed)

Evidence ladder and the maximum claim each rung supports:

| Data available | Maximum defensible claim |
|---|---|
| ColO-RAN only (have) | Concept-structure mechanism holds **within** the ColO-RAN multi-BS benchmark |
| + Twinning (arXiv 2409.16217) **by-UE/slice** | Mechanism is **conditioning-unit-general** (BS/UE/slice) and **survives real Madrid commercial traffic** |
| + self-generated multi-BS (srsRAN/ns-O-RAN/OAI) | **Cross-pipeline / cross-stack** structural property |
| operator multi-cell telemetry (**NOT available**) | operator-grade universality — **explicitly NOT claimed** |

**Decision:** Paper A claims **cross-pipeline + cross-granularity**, not "universal RAN law." The broader idea — that *conditional-structure-as-signal* generalizes to other physically-grounded FL domains (medical-across-scanners, IoT-across-environments) — is recorded as a **discussion-section hypothesis only**, never as a tested claim. Overclaiming here repeats the failure mode this ADR exists to prevent.

---

## 4. Relationship to the JSAC benchmark paper (user-mandated differentiation)

**Decision (user-confirmed 2026-05-21):** the JSAC benchmark paper is **submitted independently first**; Paper A **cites it** as the empirical basis. The JSAC paper is **not** refactored to fold in mechanism quantification, Twinning, or the offline-xApp work — that would explode scope and raise overlap.

The two papers differ in **kind of contribution**, which is the basis for the split:

| | JSAC benchmark paper (`v0.9.4`) | Paper A (this program) |
|---|---|---|
| Contribution kind | Reports the **phenomenon** + open reproducible **benchmark** | **Pre-registered mechanism verification** + **reframe** + cross-data generality |
| Core question | *What* happens across the 4-axis design space | *Why* it happens, and *whether* it is a structural property |
| Evidence | 900-cell sweep, paired-bootstrap, energy | E1–E4 (concept-shift decomposition, no-bs ablation, advantage∝structure, tr-transfer) + Twinning generality |
| Novelty | first comprehensive cross-arch FL benchmark on public ColO-RAN | concept-shift-as-signal reframe; the global-model mechanism (H2); physics grounding |

**Overlap-management rules (binding):**
1. Paper A must **not** present JSAC's benchmark tables as its own contribution; it cites them and builds new analysis on top.
2. Paper A's contribution statement must enumerate what is **new relative to JSAC** (mechanism, reframe, generality), in the contribution section and cover letter.
3. **If Paper A is submitted while the JSAC paper is not yet formally published**, Paper A must **disclose** the related submission / preprint status (engrXiv DOI + JSAC-under-review) and clearly state the added contributions — same discipline as the FedRMamba concurrent-work disclosure (`memory/jsac_v093_pending.md`).
4. Self-overlap text check (related-work / dataset / background) against the JSAC paper before submitting A.

---

## 5. Positioning & citation decisions

- **Cite + explicitly differentiate** arXiv 2211.15220 (BS-traffic FL: aggregators don't beat FedAvg). Frame: "the algorithm-insensitivity is consistent with prior wireless-FL observations [2211.15220]; our contribution is the *partition-side* inversion for the global model and its mechanism, not the aggregator finding."
- **Position against** NIID-Bench (arXiv 2102.02079) as the canonical "non-IID degrades" baseline our result inverts.
- **Use as reframe foil** arXiv 2410.15693 (verified abstract: clusters mobile IoT nodes geographically *to achieve near-IID-ness*; its 110× gain is on **grouping-cost / device-balance, not model accuracy** — cite as evidence of the "engineer-toward-IID" reflex, **not** as an accuracy claim) — the explicit opposite of our prescription.
- **Anchor mechanism prior art** in clustered/personalized/concept-drift FL (FedBN 2102.07623; FedPLC; prototype-CFL) — to claim the *grounded/global-model* version, not the bare idea.
- **Cite the covariate-vs-concept-shift decomposition prior art** (FL covariate shift arXiv 2306.05325; covariate/concept scores via feature norms/angles; Geometric Sensitivity Decomposition) as E1's *methodological basis* — our contribution is applying it to RAN telemetry + the finding, **not** the decomposition method.
- **TimeRAN (2604.04271)** = forecasting baseline and optional "does the structure appear in a foundation model's representation?" probe; **not** built upon as a competitor.
- Offline-RL-O-RAN (2312.10547; CQL 2603.03932) reserved for Paper B.

---

## 6. Program structure & gates

```
JSAC benchmark (v0.9.4, ready)  ── "phenomenon" ── submitted independently; cited by A
        │
        ▼
Paper A  ── "science": mechanism + reframe + generality
  Phase 0  Reproducibility check — re-evaluate reused checkpoints, no training  [GATE: gap reproduces]
  Phase 1  E2 primary gate (3-arm) + E1 secondary/descriptive + E3 (PREREG-A1)   [GATE: E2 SUPPORT vs HARD-KILL]
  Phase 2  Twinning by-UE/slice generality (real commercial traffic)
  Phase 3  (optional) self-generate multi-BS via srsRAN/ns-O-RAN/OAI  [only if a strong
           cross-stack claim is wanted; channel features (SINR/CQI/RSSI/BLER) + the BLER
           target are NOT standard E2SM-KPM → require custom MAC/PHY instrumentation]
        │  A establishes the structural principle that justifies B
        ▼
Paper B  ── "application": safe offline xApp recommendation     [GATE: action-coverage probe]
  contextual bandit (config-level constant actions, not RL); action = slice-PRB vector
  (cf. DORA/ORANSlice); pre/post-action variable hygiene; conservative in-support + OPE;
  differentiated from offline-RL-O-RAN by mechanism-grounding + FL-natural-partition link.
  If coverage insufficient: self-generate action-varied data (OAI/srsRAN) OR report a
  feasibility/negative result.
```

**Gates are pre-committed:** Phase 1 must clear the H1/H2 kill criteria before Phase 2/3 investment; Paper B starts only after the coverage probe (PREREG to be written as PREREG-B1 when Paper B is reached). Phase ordering is dependency-driven: B's justification *is* A's principle.

---

## 7. Kill criteria (summary; full thresholds in PREREG-A1)

- **Precondition (Phase-0):** the natural>shuffle advantage must **reproduce on this machine** (re-evaluate the reused checkpoints; no training). If not → fix reproduction before any mechanism claim.
- **PRIMARY gate = E2** (3-arm cell-conditioning ablation — no-bs / explicit-bs / bs-shuffled, under the fixed natural partition): **SUPPORT** if either `explicit-bs − bs-shuffled` or `no-bs − shuffle-partition` excludes 0 on ΔAUC **and** ΔNLL. **HARD-KILL / PIVOT** if all arms tie pairwise **and** `no-bs ≈ shuffle-partition` → the advantage is not cell-conditional structure (suspect covariate / training-dynamics).
- **INTERPRETATION = E1 (secondary, descriptive):** concept-share (vs a concept-homogenized placebo, with sensitivity over the decomposition) sets the claim *wording* (mixed vs concept-dominant). The 0.30/0.50/0.70 cut-points are **narrative bands, not survival gates**.
- **SUPPORTING = E3:** advantage∝structure correlation (pooled across bs×arch×seed; directional, given only 7 bs).

**Why E2 is the gate, not E1:** E2 has near-zero researcher degrees of freedom and is hard to contest; the concept-share ratio (E1) has a contestable `G_nonconcept` denominator (false-precision risk), so it is descriptive, not decisive. Full rule frozen in PREREG-A1 §8.

---

## 8. Consequences

**Positive.** A high-ceiling contribution grounded in physics, falsifiable, and reusable; a clean two-paper progression (phenomenon → science → application); strong defensive positioning against the established literature.

**Negative / risks.**
- *Salami / overlap with JSAC* — mitigated by §4 rules; must be enforced at submission.
- *Claim ceiling* — without operator data, generality is cross-pipeline, not universal (§3).
- *"Obvious mechanism" reviewer attack* — mitigated by H2 (the *global-model* puzzle) + the covariate-vs-concept decomposition (E1) making it non-trivial.
- *Solo bandwidth* — mitigated by cheap-first, gated execution; Phase 3 and Paper B are optional/contingent.
- *Once-falsified mechanism* — mitigated by pre-registration (PREREG-A1) and the locked kill criteria.

---

## Revision history

| Date | Change | Reason |
|------|--------|--------|
| 2026-05-21 | Initial ADR. Program accepted; JSAC-independent-first timing decided; H1–H3 + kill criteria + claim ceiling + positioning recorded. Phase 1 documented, not executed. | Post-JSAC-v0.9.4 strategic decision after a multi-round verified literature audit; user authorized writing this ADR + PREREG-A1 (documents only, no training). |
| 2026-05-21 (audit) | §5 foil precision (2410.15693's 110× is grouping-cost/device-balance, not accuracy) + added the covariate-vs-concept decomposition citations (arXiv 2306.05325 etc.) as E1's methodological basis; §7 HARD KILL reframed from an absolute floor to a relative concept-share test. 2211.15220 + 2410.15693 claims verified verbatim. | Line-by-line cross-validation vs verified web sources + repo state. |
| 2026-05-21 (restructure) | **Primary falsification gate moved to E2** (3-arm cell-conditioning ablation); **E1 concept-share demoted to secondary/descriptive** (placebo floor + sensitivity; cut-points are narrative bands, not gates); **Phase-0 reproducibility precondition added** (re-evaluate reused checkpoints, no training). Verified `bs_id` IS an embedded input → no-bs arm implementable via the existing `drop_categorical` lever. §2 H2, §6, §7 updated; full rule in PREREG-A1 §8. | User-approved reverse-opinion: E2 is less gameable than a concept-share ratio; don't let pre-registration become procrastination. |
