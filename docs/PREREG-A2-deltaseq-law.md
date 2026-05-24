# PREREG-A2 — Δ_seq governs the fragmentation AUC gap

Pre-registration for the completeness program of the standalone methods paper
*"Sequence Integrity Matters: Row-Level Client Partitioning Artifacts in Federated
RAN Time-Series Benchmarks."* Locked **2026-05-24**, BEFORE the V100 ColO-RAN sweep,
to prevent post-hoc target/threshold cherry-picking. Companion to PREREG-A1.

## Background (already established, not re-litigated here)

- Row-level client partitioning fragments per-client sliding windows (audit metric:
  intact fragmentation-score = 1.0 vs row ≈ 0.0002–0.0031 on **both** ColO-RAN and Twinning).
- ColO-RAN next-step BLER shows a large fragmentation AUC gap (+0.16); Twinning next-step
  CQI/MCS show ~0 gap — a REAL null (single-step diagnostic), not a bug.
- **Synthetic control** (`scripts/prea1/synth_seqessential_control.py`, 2026-05-24): with a
  tunable sequence-essentiality knob and pos-rate held ~0.5, the gap is monotone in
  Δ_seq, **Pearson +0.986**; the conjectured `gap ≤ Δ_seq` bound is FALSE (gap is sub-linear
  at low Δ_seq, super-linear at high — crosses y=Δ_seq). Δ_seq is a monotone **diagnostic**.

## Definitions

- **Δ_seq(target)** = AUC_intact_LSTM − AUC_single-step_logreg, where single-step logreg
  predicts the next-step label from the LAST window step's features only. (How much the
  trajectory adds over an instantaneous read.)
- **fragmentation gap(target)** = AUC_intact − AUC_row, intact = natural-by-BS / run-level
  partition (whole runs intact), row = row-level Dirichlet(α=1) (rows scattered).
- Every target's **source column is dropped from the model input features** (no label leak);
  the label is strictly next-step (k≥1), never inside the window.

## Hypotheses (confirmatory)

- **H1 (law).** Across the ColO-RAN target family below, gap is monotone increasing in Δ_seq;
  Spearman ρ(gap, Δ_seq) > 0.8 and the OLS slope CI95 (gap ~ Δ_seq) excludes 0.
- **H2 (diagnostic thresholds).** Targets with Δ_seq < 0.02 have gap < 0.02 (negligible);
  targets with Δ_seq > 0.15 have gap > 0.10.
- **H3 (shape, from synthetic).** gap is sub-linear at low Δ_seq and super-linear at high
  Δ_seq (crosses the y=Δ_seq line) — pre-registered from the synthetic Pearson 0.986 result.
- **H4 (architecture invariance).** A no-sequence MLP has Δ_seq ≈ 0 and gap ≈ 0 for ALL
  targets (sanity); LSTM / Mamba / Transformer fall on the same gap-vs-Δ_seq line on a
  vulnerable target.

## Target family (ColO-RAN, 17 V3_CONTINUOUS features, OOD-by-tr split)

~13 targets chosen a priori to span Δ_seq ∈ [0, ~0.3]:
1. next-step `ul_bler` > θ, θ ∈ {0.05, 0.10, 0.20}                 (3)
2. k-step-ahead `ul_bler` > 0.10, k ∈ {1, 2, 3, 5}                 (4)
3. next-step below/above median: `dl_cqi`, `dl_mcs`, `dl_buffer`, `tx_brate` (4)
   - `ul_sinr` was listed but DROPPED pre-results: median=0 with 64% of rows exactly 0, so
     `1[ul_sinr_{t+1} < median]` has pos-rate 0 and AUC is undefined. Technical exclusion
     (its fragmentation gap was never computed), not a result-based selection.
4. a smoothed/trend `ul_bler` target (5-step rolling mean exceeds θ) (1)

## Protocol

- Per target: drop the source column(s) from features; compute Δ_seq + gap.
- Partition modes: natural-by-BS (intact) + run_dirichlet(α=1) (intact) vs dirichlet(α=1)
  (row) vs random_split (row); 5 seeds; 100 rounds (matching the main ColO-RAN protocol).
- Fragmentation continuum (Tier-2 ⑤): for ≥2 vulnerable targets, also α ∈ {0.1, 1.0} to
  trace gap vs fragmentation-score.
- Stats: paired-bootstrap CI95 (n_boot=10k) on per-target gap; OLS gap~Δ_seq with slope CI;
  report all targets (no dropping).

## Decision rules

- **H1 confirmed** ⇒ Δ_seq is a validated diagnostic; it becomes the paper's central result.
- **H1 fails** (non-monotone, or ρ ≤ 0.8, or slope CI includes 0) ⇒ report honestly; the
  "diagnostic" claim weakens to "task-conditional, mechanism not yet predictable" — do NOT
  salvage by dropping targets.
- **H4 MLP sanity fails** (MLP shows a gap) ⇒ a bug in the fragmentation pipeline; halt and
  debug before any claim.

## Implementation note

Requires generalizing the target builder in `fl_v7` (target column + threshold/quantile +
horizon k), reusing the merged `--drop-continuous` plumbing. Pre-registered cell list +
aggregator to be added under `experiments/specs/` + `scripts/prea1/` before launch.
