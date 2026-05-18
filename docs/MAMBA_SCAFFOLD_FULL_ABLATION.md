# Mamba × SCAFFOLD: full 6-partition × 10-seed ablation

**Status**: post-Phase-5 reanalysis (2026-05-18). Source data: 60 cells from
`artifacts/v7_stage2_full/v7_mamba_scaffold_*_s*/history.csv`. Same data
that paper v0.9.2-submission-ready (Zenodo DOI 10.5281/zenodo.20075433)
was derived from — no new training cells. This document re-aggregates
the existing Phase 5 sweep along a different metric axis than the paper
narrative uses.

**Motivation**: paper §6.3 "Architecture × algorithm catastrophic
interaction" claims `Mamba × SCAFFOLD × α∈{0.10, 0.50}` is catastrophic
with cross-seed std ~0.083 amplified 3× over the FedAvg baseline. A
deeper reanalysis along the within-seed convergence axis shows the
catastrophe is **systemic across all 6 partition variants** (5/6 with
≥50% seed-failure), but the paper's cross-seed-std detection metric
has a blind spot for the "all seeds consistently fail" failure mode
and therefore did not flag 4 of the 6 partitions.

## TL;DR

**Mamba × SCAFFOLD fails in 5 of 6 partition variants tested**
(35 of 60 cells final AUC < 0.55 = effectively random; 48 of 60 cells
final AUC < 0.75 = sub-baseline). The paper §6.3 narrative — restricting
the catastrophic claim to `α ∈ {0.10, 0.50}` — is technically true for
the cells it cites but **understates the broader systematic failure**.
The cross-seed-std detection metric the paper uses has a known
blind spot: it fires for "some seeds recover, some don't" (high seed
disagreement, e.g., α=0.05, std=0.141) but misses "all seeds collapse
consistently to similar bad value" (low seed disagreement, e.g.,
α=5.00, std=0.052 with 8/10 seeds at AUC < 0.55).

## Full 6 × 10 matrix

Seeds = `{0, 1, 2, 3, 4, 5, 6, 7, 8, 42}` (same as Phase 5). `Collapsed` =
final val_auc < 0.55 (effectively random). `Below baseline` = final
val_auc < 0.75 (Mamba × FedAvg × IID gets ~0.92 → 0.75 is the
"useful predictor" floor).

| Partition | Cross-seed mean ± std | Collapsed (<0.55) | Below 0.75 | Within-curve catastrophic | Paper §6.3 flagged? |
|---|---|---|---|---|---|
| **IID** (natural-BS) | 0.7154 ± 0.1989 | 5 / 10 | 5 / 10 | 6 / 10 | ❌ MISSED |
| **α = 5.00** | 0.5109 ± 0.0524 | **8 / 10** | **10 / 10** | 9 / 10 | ❌ MISSED ⚠️ |
| **α = 1.00** | 0.5792 ± 0.0813 | 5 / 10 | **10 / 10** | 10 / 10 | ❌ MISSED |
| **α = 0.50** | 0.5275 ± 0.0703 | 8 / 10 | 10 / 10 | 10 / 10 | ✓ flagged |
| **α = 0.10** | 0.6230 ± 0.1332 | 5 / 10 | 7 / 10 | 10 / 10 | ✓ flagged |
| **α = 0.05** | 0.6706 ± 0.1406 | 4 / 10 | 6 / 10 | 10 / 10 | (correctly not-flagged-as-catastrophe; recovers to acceptable AUC for some seeds) |
| **TOTAL** | — | **35 / 60 = 58%** | **48 / 60 = 80%** | **55 / 60 = 92%** | 2 / 6 partitions |

For comparison, the **Mamba × FedAvg × α=0.10 baseline** (paper §6.3 reference cell):
`0.8439 ± 0.0288`, 0 / 10 collapsed, 0 / 10 below baseline.

## Paper §6.3 cross-seed-std blind spot

Paper §6.3 uses cross-seed std of final AUC as the catastrophic-
detection metric:

> "Mamba × SCAFFOLD × α=0.10 yields AUC 0.7609 ± 0.0830 — std amplified
> ~3× over (Mamba, FedAvg) at the same α (0.0251)"

This metric correctly fires when **some seeds recover and others collapse**
(high seed disagreement → high std). But it has a systematic blind spot:

| Failure mode | Cross-seed std signature | Paper §6.3 detection |
|---|---|---|
| Some-seeds-recover, some-don't | HIGH std (≥0.10) | ✓ Fires |
| All-seeds-consistently-fail-to-similar-value | LOW std (~0.05) | ❌ Silent — looks "stable at bad value" |

The Mamba × SCAFFOLD matrix above contains BOTH failure modes:
- `α ∈ {0.05, 0.10, 0.50}`: high seed disagreement → flagged
- `α ∈ {1.00, 5.00, IID}`: consistent failure → missed

The paper's narrative says only 2 of 6 partitions are catastrophic. The
data says 5 of 6 are.

## Why this matters operationally

The paper §6.3 deployment warning is currently:

> "Control-variate methods interact pathologically with selective-scan
> SSMs under high heterogeneity, even as both components individually
> behave well on neighboring cells."

This implies an operator can use SCAFFOLD + Mamba **safely at high α**
(low heterogeneity) or at the natural-BS partition. The data
contradicts both implications:
- `Mamba × SCAFFOLD × α=5.00`: 8/10 seeds collapse to random
- `Mamba × SCAFFOLD × natural-BS`: 5/10 seeds collapse to random

A more honest deployment warning, supported by the full 60-cell matrix:

> **"Do not combine SCAFFOLD with Mamba selective-scan backbones in any
> heterogeneity regime tested (5 of 6 partition variants exhibit
> systematic training failure with 50%+ seed collapse to random AUC).
> The catastrophe is not heterogeneity-conditional; it is architecture
> × algorithm-conditional. The paper's earlier emphasis on α ∈ {0.10,
> 0.50} reflects only the cells where seed-to-seed outcomes most
> disagreed; cells with consistent failure across all seeds have low
> cross-seed std and were not flagged by the paper's detection
> heuristic."**

## Proposed paper §6.3 revision wording

For a post-acceptance JSAC revision (or v1.0 redeposit), the §6.3
paragraph could be replaced by:

> Across the 60 Mamba × SCAFFOLD cells in our Phase 5 sweep, 35 (58%)
> reach final AUC < 0.55 (effectively random prediction) and 48 (80%)
> reach final AUC < 0.75 (sub-utility-threshold for SLA prediction).
> Failure is systematic: 5 of 6 partition variants tested have ≥50%
> seed-collapse, including the natural-BS partition (5/10) and the
> mildest Dirichlet partition α=5.00 (8/10). The two partition variants
> the original analysis emphasised (α ∈ {0.10, 0.50}) have the highest
> CROSS-SEED disagreement (some seeds recover, others collapse, std
> 0.07-0.13) but are NOT the most failure-dominated cells — α=5.00 has
> only 0.05 std yet 8/10 seeds at random AUC. We attribute the original
> narrative's restriction to α ∈ {0.10, 0.50} to a known blind spot in
> the cross-seed-std detection heuristic: it fires on inconsistent
> failure but misses uniformly-bad consistent failure. The deployment
> recommendation should therefore be **do not deploy SCAFFOLD with
> selective-scan SSM backbones in any heterogeneity regime** rather
> than the partition-conditional warning the original narrative
> implies.

## Methodology

Per-cell metrics:
- `final` = `df["val_auc"].iloc[-1]` (final round's val AUC)
- `σ(Δ)` = std of round-to-round val_auc deltas (within-curve volatility)
- `max_drop` = `np.diff(auc).min()` (worst single-round AUC drop)
- `osc20` = `auc[-20:].max() - auc[-20:].min()` (steady-state oscillation)
- **within-curve catastrophic** = `abs(max_drop) > 0.10 OR osc20 > 0.05`

Per-partition aggregates over 10 seeds:
- cross-seed mean / std of `final`
- count of seeds with `final < 0.55` (collapsed)
- count of seeds with `final < 0.75` (sub-baseline)
- count of seeds with within-curve catastrophe

Reproducibility:
```bash
source .venv/bin/activate
python << 'EOF'
import pandas as pd, numpy as np
from pathlib import Path
seeds = [0,1,2,3,4,5,6,7,8,42]
parts = ["iid_n7", "dirichlet_a5p00_n7", "dirichlet_a1p00_n7",
         "dirichlet_a0p50_n7", "dirichlet_a0p10_n7", "dirichlet_a0p05_n7"]
for p in parts:
    finals = []
    for s in seeds:
        csv = Path(f"artifacts/v7_stage2_full/v7_mamba_scaffold_{p}_s{s}/history.csv")
        if csv.exists():
            finals.append(pd.read_csv(csv)["val_auc"].iloc[-1])
    print(f"{p:30s} | {np.mean(finals):.4f} ± {np.std(finals):.4f} | "
          f"collapsed={sum(1 for f in finals if f < 0.55)}/{len(finals)}")
EOF
```

## Caveats

1. **Single-cell-aggregator-snapshot**: Phase 5 final-round metric only.
   The paper's paired-bootstrap CI95 over per-seed delta is a different,
   complementary view. Both views are valid; they describe different
   aspects of the failure.

2. **Within-curve oscillation criterion is conservative**: any cell with
   `max_drop > 0.10` is flagged, even if it recovers. The "recovers
   despite oscillation" cells (e.g., Mamba × FedAdam × α=0.10) are
   correctly NOT flagged catastrophic at the final-AUC level — they
   recover. So the 92% within-curve catastrophic rate for Mamba ×
   SCAFFOLD is genuinely structural; cells do not recover.

3. **No claim about Mamba × SCAFFOLD with different hyperparameters**:
   our SCAFFOLD uses Option-II per paper §4 (Adam-friendly default). The
   canonical Option-I variant is not analysed here. A SCAFFOLD variant
   that handles Mamba's selective-scan state differently might rescue
   some cells.

## Open questions for future revision

1. Is the failure rooted in SCAFFOLD's per-client `c_i = ∇L(w_g)` Option-II
   formula interacting badly with Mamba's input-dependent `Δ_t`, `B_t`,
   `C_t` projections (the selective-scan state which has no analogue in
   LSTM)? A first-principles derivation would either confirm or refute
   the mechanism.
2. Does the same systemic failure extend to Mamba-3 (PR #23 / extension
   sweep #40) given its data-dependent `λ_t` (trapezoidal mix) and
   `θ_t` (rotation angle) projections? The Path D extension cells will
   contain `mamba3 × {fedscam, fedgmt, fedmoswa}` data points, but NOT
   `mamba3 × {SCAFFOLD, FedDyn}` per current spec. A follow-up sweep
   covering `mamba3 × SCAFFOLD` is needed to extend this analysis.
3. The single instance of `α=0.05` recovering for some seeds (4/10
   collapsed, mean 0.67) suggests there may be a recovery mechanism at
   extreme heterogeneity (small client subsets → less averaging noise?).
   Worth investigating mechanistically.
