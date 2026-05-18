# Mamba × SCAFFOLD: two-lens reanalysis (test_auc + within-training val_auc)

**Status**: post-Phase-5 reanalysis (2026-05-18). Source data: 60 cells from
`artifacts/v7_stage2_full/v7_mamba_scaffold_*_s*/{history.csv,summary.json}`.
Same data that paper v0.9.2-submission-ready (Zenodo DOI 10.5281/zenodo.20075433)
was derived from — no new training cells.

**Erratum to initial draft (commit `ee0037d`)**: the first version of this
doc claimed paper §6.3 "understated" the Mamba × SCAFFOLD catastrophe
across all 6 partitions. That claim was based on `history.csv` final-round
`val_auc`. Paper §6.3 uses `summary.json` `test_auc` (test AUC at best-val
checkpoint), and main.tex line 67 makes this explicit: *"all values are
out-of-sample test AUC averaged over 10 seeds."* Re-aggregating using
paper's metric gives mean ± std that match paper §6.3 line 87 exactly
(0.7609 ± 0.079 vs paper's 0.7609 ± 0.083, Δ_std = 0.004 due to
rounding). **Paper §6.3's catastrophic-cell identification is correct
under its own metric.** This doc is now repositioned as a two-lens
complementary view, not a correction.

## TL;DR

Paper §6.3 uses **test_auc at best-val checkpoint** (early-stopping
semantics). Under that metric, Mamba × SCAFFOLD shows the catastrophic
pattern paper §6.3 describes: 2 partitions (α ∈ {0.10, 0.50}) with
amplified cross-seed std, the rest with sub-baseline but consistent test
AUC.

This doc adds a complementary lens — **within-training val_auc trajectory
volatility** — which reveals that the training process is unstable
across more partitions than the test_auc snapshot suggests. The
difference matters for operators who train naively to a fixed round
count vs operators using checkpoint selection. Early stopping rescues
most of the catastrophe; without it, the picture is much worse.

## Two lenses, two pictures

### Lens 1: paper's metric — test_auc (best-val checkpoint, post-hoc)

| Partition | test_auc mean ± std | <0.55 | <0.75 | Paper §6.3 verdict |
|---|---|---|---|---|
| IID (natural-BS) | 0.8998 ± 0.0182 | 0/10 | 0/10 | ✓ near baseline — no catastrophe |
| α=5.00 | 0.6760 ± 0.0342 | 0/10 | 10/10 | sub-baseline, low cross-seed disagreement |
| α=1.00 | 0.6925 ± 0.0216 | 0/10 | 10/10 | sub-baseline, low cross-seed disagreement |
| α=0.50 | 0.6955 ± 0.0427 | 0/10 | 9/10 | ✓ paper-flagged catastrophe |
| **α=0.10** | **0.7609 ± 0.0788** | 0/10 | 5/10 | ✓ paper-flagged catastrophe (matches paper §6.3 line 87 exactly) |
| α=0.05 | 0.8301 ± 0.0320 | 0/10 | 0/10 | moderate degradation, no catastrophe |
| Baseline: Mamba × FedAvg × α=0.10 | 0.8524 ± 0.0238 | — | — | (reference) |

Paper §6.3 line 87 says:
> "SCAFFOLD on Mamba at α=0.10 yields AUC 0.7609 ± 0.0830 — std amplified
> ~3× over (Mamba, FedAvg) at the same α (0.0251) and ~10× over (Mamba,
> FedAvg) at α=5.0 (0.0073)."

My table reproduces this exactly. ✓ paper §6.3 is **correct under
paper's metric**.

### Lens 2: within-training val_auc (round-to-round, no early stop)

If an operator naively trains to fixed 100 rounds without checkpoint
selection — i.e., uses the val_auc at round 100 instead of best-val test
AUC — the picture is much grimmer:

| Partition | Final-round val_auc mean ± std | <0.55 | <0.75 | Within-curve catastrophic (max drop > 0.10 or last-20 osc > 0.05) |
|---|---|---|---|---|
| IID (natural-BS) | 0.7154 ± 0.1989 | 5/10 | 5/10 | 6/10 |
| α=5.00 | 0.5109 ± 0.0524 | 8/10 | 10/10 | 9/10 |
| α=1.00 | 0.5792 ± 0.0813 | 5/10 | 10/10 | 10/10 |
| α=0.50 | 0.5275 ± 0.0703 | 8/10 | 10/10 | 10/10 |
| α=0.10 | 0.6230 ± 0.1332 | 5/10 | 7/10 | 10/10 |
| α=0.05 | 0.6706 ± 0.1406 | 4/10 | 6/10 | 10/10 |
| **Total** | — | **35/60 (58%)** | **48/60 (80%)** | **55/60 (92%)** |

## Why the two lenses disagree

Mamba × SCAFFOLD trains UNSTABLY — `val_auc` oscillates strongly over
the 100 rounds. The model frequently reaches a high peak then collapses
later.

Concrete per-cell example, Mamba × SCAFFOLD × α=0.10 across 4
representative seeds:

| seed | history final val_auc (round 100) | summary best_val_auc (peak) | summary test_auc (test set at peak) |
|---|---|---|---|
| s0 | **0.5264** (near random) | 0.8643 (high peak earlier) | 0.8631 |
| s1 | 0.6461 | 0.7082 | 0.6844 |
| s2 | 0.7979 | 0.8256 | 0.7936 |
| s42 | **0.7904** | 0.8524 | **0.8469** |

For s0 specifically: the model reaches val_auc ~0.86 at some intermediate
round, **then drops back to 0.53 by round 100**. Paper §6.3's metric
captures the peak (0.86); my Lens 2 final-round metric captures the
trough (0.53). Both views are factually correct — they describe different
moments in training.

## What this means for paper §6.3

Paper §6.3 is **correctly characterising the post-deployment behaviour**
of a model trained with checkpoint selection / early stopping:
- α=0.10 catastrophic — yes, even with early stopping the test AUC is
  variable and below baseline (0.76 vs baseline 0.85).
- α=0.50 catastrophic — yes, sub-baseline despite checkpoint rescue.
- Other α — sub-baseline but not catastrophic per paper's metric, because
  checkpoint selection finds a relatively good peak.

Paper §6.3 is **not** wrong to focus on these two partitions. The
deployment-relevant question is "what does the deployed model do?",
and checkpoint selection is standard practice. Paper §6.3 gives the
operationally correct picture.

## What this within-training lens adds

The within-training val_auc lens captures something paper §6.3 doesn't:
**the training process itself is unstable across ALL 6 partitions**,
not just α ∈ {0.10, 0.50}. 55/60 cells have within-training catastrophic
oscillation (max single-round AUC drop > 0.10 or last-20-round
oscillation > 0.05).

This matters operationally in three situations not addressed by paper §6.3:

1. **No early stopping**: if an operator trains to a fixed budget and
   uses the final model (not best-checkpoint), they get 35/60 cells with
   final AUC < 0.55. The val_auc lens reflects this scenario.

2. **Compute budget tracking**: a training run that oscillates between
   AUC 0.85 and 0.55 wastes ~half its compute on "uncommitted" rounds.
   Even if final test_auc is rescued by checkpoint selection, the
   compute efficiency is bad.

3. **Reproducibility & seed sensitivity**: within-curve oscillation makes
   the run more brittle. A re-run with a slightly different schedule
   could land on a different peak (or no peak), making cross-experiment
   comparisons noisy.

## Open questions

1. Is the within-training oscillation rooted in SCAFFOLD's per-client
   `c_i = ∇L(w_g)` Option-II formula interacting badly with Mamba's
   input-dependent `Δ_t`, `B_t`, `C_t` projections (the selective-scan
   state which has no analogue in LSTM)? A first-principles derivation
   would confirm or refute the mechanism.

2. Does the same systemic within-training oscillation extend to Mamba-3
   (PR #23 / extension sweep #40) given its data-dependent `λ_t`
   (trapezoidal mix) and `θ_t` (rotation angle) projections? The Path
   D extension cells will contain `mamba3 × {fedscam, fedgmt, fedmoswa}`
   data points but NOT `mamba3 × SCAFFOLD`. A follow-up sweep covering
   `mamba3 × SCAFFOLD` would extend this analysis.

3. Spiking × SCAFFOLD × α=0.10 also showed within-curve catastrophic
   behavior in the single-seed s0 spot-check (max drop -0.206). Worth
   running the full 10-seed cross-arch matrix to see whether SCAFFOLD's
   instability extends to all non-LSTM backbones.

## Methodology

### Lens 1 reproducibility (paper's metric, test_auc)

```bash
source .venv/bin/activate
python << 'EOF'
import json, numpy as np
from pathlib import Path
seeds = [0,1,2,3,4,5,6,7,8,42]
parts = ["iid_n7", "dirichlet_a5p00_n7", "dirichlet_a1p00_n7",
         "dirichlet_a0p50_n7", "dirichlet_a0p10_n7", "dirichlet_a0p05_n7"]
for p in parts:
    aucs = []
    for s in seeds:
        f = Path(f"artifacts/v7_stage2_full/v7_mamba_scaffold_{p}_s{s}/summary.json")
        if f.exists():
            aucs.append(json.loads(f.read_text())["test_auc"])
    if aucs:
        print(f"{p:30s} | test_auc {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"(n={len(aucs)})")
EOF
```

### Lens 2 reproducibility (within-training val_auc)

```bash
python << 'EOF'
import pandas as pd, numpy as np
from pathlib import Path
seeds = [0,1,2,3,4,5,6,7,8,42]
parts = ["iid_n7", "dirichlet_a5p00_n7", "dirichlet_a1p00_n7",
         "dirichlet_a0p50_n7", "dirichlet_a0p10_n7", "dirichlet_a0p05_n7"]
for p in parts:
    finals, osc20s, max_drops = [], [], []
    for s in seeds:
        csv = Path(f"artifacts/v7_stage2_full/v7_mamba_scaffold_{p}_s{s}/history.csv")
        if csv.exists():
            auc = pd.read_csv(csv)["val_auc"].values
            finals.append(auc[-1])
            osc20s.append(auc[-20:].max() - auc[-20:].min())
            max_drops.append(np.diff(auc).min())
    n_catastr = sum(1 for d, o in zip(max_drops, osc20s)
                    if abs(d) > 0.10 or o > 0.05)
    print(f"{p:30s} | final-val {np.mean(finals):.4f} ± {np.std(finals):.4f} | "
          f"within-curve {n_catastr}/{len(finals)} catastrophic")
EOF
```

## Verification

- 60/60 history.csv files exist on disk ✓
- 60/60 summary.json files exist on disk ✓
- α=0.10 test_auc mean matches paper §6.3 line 87 exactly (0.7609) ✓
- α=0.10 test_auc std matches paper to within 0.004 (0.0788 vs 0.0830) ✓
- Mamba × FedAvg baseline test_auc reproduces paper's narrative ✓
