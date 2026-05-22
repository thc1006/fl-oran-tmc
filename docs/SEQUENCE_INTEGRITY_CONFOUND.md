# Sequence-integrity confound under row-level client partitioning

**Status: RESOLVED 2026-05-22 — CONFOUND CONFIRMED.** The decisive `run_random` gate
(below) shows the natural-by-BS advantage is **sequence integrity, not BS-coherence**.
Consequently:
- **The JSAC §6 "inverted heterogeneity / natural-by-BS dominance as a RAN structural
  property" framing is an artifact** (DIRECTLY confirmed across the α-curve by the
  factorial below) and must be removed/reinterpreted before any submission. The HOLD
  (do not submit / do not claim it is a RAN structural property) **stands until that
  rewrite is done.**
- **Paper A's original thesis ("conditional structure, not distributional skew") is
  falsified** and stops; it pivots to the methodological contribution (a row-level
  partitioning artifact in FL-time-series benchmarks).

Opened 2026-05-22 after an adversarial review of the PREREG-A1 Phase-1 mechanism work;
resolved the same day by the `run_random` control.

---

## The confound in one sentence

The non-IID partitions that lose to natural-by-BS (`random_split`, `dirichlet`,
`per_bs_dirichlet`) are all **row-level**: they scatter a run's consecutive
timesteps across clients, so the per-client `build_run_sequences` builds
**temporally-broken windows**. `natural-by-BS` (partition by `bs_id`) keeps each
run intact. So the headline "+0.175 AUC natural-by-BS advantage" (and the
inverted-α finding) may be measuring **sequence integrity, not BS-coherence /
cell-conditional structure**.

## Evidence (code-verified + data + theory)

1. **Code (the mechanism).**
   - `data_v2/partition.py`: `random_split` = `rng.permutation(n_rows)` then split
     (row-level); `dirichlet` shuffles **row indices within each slice**
     (row-level); `per_bs_dirichlet` likewise scatters rows within a BS.
   - `training/fl_v7.py` (≈ lines 633→653): the order is **`_partition(split.train)`
     first, then per-client `build_run_sequences`** — so a scattered-row client
     windows over non-contiguous `step_idx`.
   - `data_v2/sequences.py`: `build_run_sequences` groups by `(run_id, slice_id)`,
     sorts by `step_idx`, and slides a window over **consecutive rows in the
     subset** with **no contiguity guard** — scattered rows → a "5-step window"
     spanning ~35 real steps.

2. **Centralized triangle (the data).** Same task/split:
   | setup | sequences | BS mixing | AUC |
   |---|---|---|---|
   | centralized | intact | fully mixed | **0.931 / 0.924** |
   | natural-by-BS FL | intact | BS-coherent | **0.916** |
   | random_split FL | corrupted | mixed | **0.740** |

   `random_split` is **IID-by-row** — FL theory says FedAvg on IID should ≈
   centralized (~0.93); it's at **0.74**. Centralized is **fully BS-mixed** yet
   scores 0.93 — so BS-mixing is *not* what hurts. The only thing separating
   centralized (0.93) from random_split (0.74), both BS-mixed, is intact vs
   row-shuffled sequences.

3. **Partition audit (`artifacts/prea1/run_random/partition_audit.json`).**
   `random_split` gives each client **7369 fragmented `(run,slice)` groups** with
   only ~1.2% fewer windows than intact — so the corruption is **temporal
   scrambling within windows**, not data loss.

## What this affects
- JSAC paper §6 headline (natural-by-BS dominance, inverted-α) — same row-level
  Dirichlet partitions.
- Paper A's thesis ("conditional structure, not distributional skew").
- The 900-cell sweep's **cross-partition** comparisons (within-natural arch
  comparisons are unaffected).
- §7.1.5 `per_bs_dirichlet` "control" (also row-level → also confounded).

## The decisive gate: `run_random` control

`partition_clients(mode="run_random")` — assign each **whole `(run_id, slice_id)`
group** to one client at random (greedy least-loaded). **Intact sequences, broken
BS-coherence.** Matched to natural-by-BS in groups/rows/valid-seq/tr (audited);
differs only in BS coherence (7 BS/client vs 1). Trained same-env as the E2
natural baseline (LSTM × FedAvg × 5 seeds, V100 fp16 eager, identical hypers).

**Pre-registered decision rule (locked before seeing results):**
- `run_random AUC ≈ natural-by-BS` (~0.90–0.92) → **confound confirmed**: the
  advantage is sequence integrity. JSAC must rewrite the partition
  interpretation; Paper A's original thesis stops (pivots to a methodological
  contribution: "row-level non-IID partitioning corrupts FL-time-series
  benchmarks").
- `run_random AUC ≈ random_split` (~0.74–0.80) → **thesis survives**: BS-coherence
  matters even with intact sequences; JSAC adds this control to strengthen.
- `run_random AUC` mid (~0.82–0.89) → **mixed**: sequence integrity explains part;
  needs run-level Dirichlet / run-level per-BS follow-ups (phase 2).

Comparison baseline: E2 natural-by-BS (same env, fp16-V100-eager) = **0.91588**
(`artifacts/prea1_e2/explicit_bs/`). Row-level floor (env-stable) = **0.740**.

## Gate progress
- `run_random` mode implemented (`partition.py`) + 8 unit tests green
  (`tests/test_v7_run_random_partition.py`) + V7Config naming (`runrandom`).
- Real-data partition audit passed (0 groups split, 200/200 contiguous; clean
  isolation vs natural).
- Single-cell-CLI integration smoke passed (eager, no triton; 0.863 AUC @ 8
  rounds; config field-by-field == E2 natural).
- **5-seed control DONE** (V100, fp16 eager, identical hypers, paired with E2-natural):

  | partition | sequences | BS | mean test-AUC (n=5) |
  |---|---|---|---|
  | natural-by-BS (E2 explicit) | intact | coherent | 0.91572 |
  | **run_random** | intact | **mixed** | **0.91583** |
  | random_split (floor) | corrupted | mixed | 0.740 |

  `natural − run_random = −0.00011`, paired bootstrap CI95 **[−0.00030, +0.00013]**
  (straddles 0); `run_random − floor = +0.17554`. Breaking BS-coherence with sequences
  intact costs ~0; the entire +0.175 "advantage" is recovered by intact sequences alone.
  Three-point closure: intact+coherent = 0.916, intact+mixed = 0.916, corrupted+mixed =
  0.74. **CONFOUND CONFIRMED** — the advantage is the intact-vs-corrupted-sequence axis,
  ≈0 from BS-coherence. (Per-cell results: `artifacts/prea1_run_random/`.)

## Decision & next steps (2026-05-22)
1. **JSAC**: do not submit until §6 is rewritten. The natural-by-BS / inverted-α results
   must be reframed as **sensitive to a partitioning artifact**, OR the non-IID baselines
   must be re-run with **sequence-preserving (run-level) partitioning** and the headline
   re-derived from whatever survives.
2. **Paper A**: original mechanism thesis falsified. Pivot to a methodological paper:
   *"Row-level non-IID client partitioning corrupts FL-time-series benchmarks; the
   apparent inverted heterogeneity in O-RAN slice-SLA FL is a sequence-integrity
   artifact."* — likely generalizes beyond this dataset.
3. **Downstream** (offline xApp recommender, Twinning): were predicated on the inversion
   being a real exploitable structure → re-scope before investing.
4. **Factorial confirmation (DONE 2026-05-22)** — run-level vs row-level Dirichlet,
   same-env (V100 fp16 eager, LSTM×FedAvg, 3 seeds {0,1,2}, identical hypers):

   | α | run_dirichlet (intact) | dirichlet (row, corrupt) | corruption cost |
   |---|---|---|---|
   | 0.1 | 0.91395 | 0.83685 | +0.077 |
   | 0.5 | 0.91577 | 0.79035 | +0.125 |
   | 1.0 | 0.91589 | 0.75480 | +0.161 |

   natural = 0.91569; random_split (corrupt-IID floor) = 0.73966. **run_dirichlet
   (intact) is FLAT at ~natural across all α** — heterogeneity does not hurt when
   sequences are intact. The row-level deficit, and its α-dependence (row-Dirichlet
   *improves* as α↓: 0.755→0.790→0.837), is sequence **fragmentation**: smaller α →
   fewer clients hold each run → less fragmentation → less corruption. So the "inverted
   heterogeneity" is a corruption-severity effect, not a real heterogeneity benefit.
   **The JSAC §6 inverted-α is now a DIRECTLY-DEMONSTRATED sequence-corruption artifact**
   (no longer an inference). Per-cell: `artifacts/prea1_factorial/`.

## Complete same-env factorial (the one-line summary)

| | intact sequences | corrupted (row-level) |
|---|---|---|
| coherent (natural-by-BS) | **0.916** | — |
| IID (run_random / random_split) | **0.916** | 0.740 |
| skewed α=0.1 | **0.914** | 0.837 |
| skewed α=0.5 | **0.916** | 0.790 |
| skewed α=1.0 | **0.916** | 0.755 |

The **intact column is flat (~0.916) regardless of coherence or skew**; the corrupted
column ranges 0.74–0.84 by fragmentation severity. **Sequence integrity is the sole
axis; BS-coherence and heterogeneity contribute ~0.** The natural-by-BS dominance and
the inverted-α are both artifacts of row-level partitioning of a time series.
