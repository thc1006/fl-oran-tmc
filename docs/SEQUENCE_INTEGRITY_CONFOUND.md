# Sequence-integrity confound under row-level client partitioning

**Status: OPEN — BLOCKING.** Until the decisive gate below resolves:
- **Do NOT submit the JSAC paper.**
- **Do NOT claim the inverted-heterogeneity / natural-by-BS dominance is a *structural property of RAN telemetry*** (in JSAC §6, Paper A, or any preprint/talk).

Opened 2026-05-22 after an adversarial review of the PREREG-A1 Phase-1 mechanism work.

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
- **5-seed control RUNNING** on V100 (`artifacts/prea1_run_random/`,
  `v7_lstm_fedavg_runrandom_n7_s{0,1,2,3,42}`). Result pending → update this file
  and lift/confirm the HOLD accordingly.
