# Sequence-Integrity Confound — Auditable Evidence Pack

**Status (2026-05-22):** RESOLVED CONFOUND, CONFIRMED across 3 architectures.
**Main-line claim (proven on this dataset):** *row-level client partitioning can create
severe sequence-integrity artifacts in federated RAN time-series benchmarks.* The apparent
"inverted heterogeneity" (natural-by-BS beating every Dirichlet α) on Colosseum/ColO-RAN is
such an artifact, not a structural property of RAN telemetry.
**Explicitly a hypothesis, NOT a proven claim:** *that this pitfall broadly affects other
published FL-time-series benchmarks.* Raising that to a claim requires replication on Twinning
or another RAN time-series corpus (see §7).

Superseded: the "cell-conditional structure" thesis (Paper A) is **falsified**; Paper B
(offline xApp) and Twinning are re-scoped to follow-ups, not active. See
`docs/SEQUENCE_INTEGRITY_CONFOUND.md` (HOLD note) and memory `paper-a-conditional-structure-program`.

Reframe branch: `paper-seq-integrity-rewrite` (8 commits `3ea8e84..0bafbe8`).

---

## 1. Implementation diffs + test results

### 1.1 Partition modes (`src/fl_oran/data_v2/partition.py`)

| commit | mode | files changed | ins/del | what it does |
|---|---|---|---|---|
| `fdf6f97` | `run_random` | partition.py +44, fl_v7.py +16, run_v7_fl_arch_sweep.py +4, test +159, confound-doc +92 | +311 / −4 | assigns whole `(run_id, slice_id)` groups to N clients by greedy least-loaded over a seed-shuffled group order → bs-coherence broken, **sequences intact** |
| `e14c295` | `run_dirichlet` | partition.py +48, fl_v7.py +14, factorial-launcher +46, test +112, confound-doc +42 | +255 / −9 | per-slice `Dir(α)` over whole `(run_id, slice_id)` groups → Dirichlet-skewed allocation, **sequences intact** |

Both dispatch through `_partition()` in `training/fl_v7.py`; both are CLI-exposed via
`experiments/run_v7_fl_arch_sweep.py --partition-mode {run_random,run_dirichlet}`.
Production order verified (fl_v7.py): `split = ood_split_by_tr(...)` → `client_dfs = _partition(split.train)`
→ per-client `build_run_sequences(...)` — i.e. **partition-then-window** (no contiguity guard), which
is exactly what row-level partitioning corrupts.

Fragmentation/audit analysis scripts (committed under `scripts/prea1/`):
`measure_contiguity.py` (§4), `partition_audit.py` (§2).

### 1.2 Unit tests — `15 passed`

```
pytest tests/test_v7_run_random_partition.py tests/test_v7_run_dirichlet_partition.py --no-cov
=> 15 passed   (8 run_random + 7 run_dirichlet)
```

Invariants asserted: (a) no `(run_id, slice_id)` group is split across clients; (b) within each
client every group's `step_idx` is contiguous (windows valid); (c) all N clients non-empty &
reasonable sizes; (d) determinism under fixed seed; (e) `run_dirichlet` skew increases as α↓; (f)
cell-name builders round-trip (105 cross-val names). Full suite still green
(`pytest --no-cov`, 605+).

---

## 2. Partition audit table (5 modes, real `split.train`)

Source: `scripts/prea1/partition_audit.py` → `artifacts/prea1/partition_audit.json`
(`coloran_raw_unified.parquet`, OOD train split = `tr ∈ 0..21`, 14.5M rows, seq_len=5, N=7, seed=0).
**Fragmentation score = fraction of per-client sliding windows whose 5 rows have contiguous
`step_idx`** (1.0 = no fragmentation; →0 = every window spans gaps).

### 2.1 Per-mode summary

| mode | total windows | fragmentation score | bs-coherence | sequences |
|---|---|---|---|---|
| natural (iid by-BS) | 14,515,074 | **1.0000** | preserved (1 bs/client) | intact |
| row-random (`random_split`) | 14,338,218 | **0.0004** | broken | **corrupted** |
| row-Dirichlet (α=1.0) | 14,345,955 | **0.0043** | broken | **corrupted** |
| `run_random` | 14,515,074 | **1.0000** | broken | intact |
| `run_dirichlet` (α=1.0) | 14,515,074 | **1.0000** | broken | intact |

Key contrast: `run_random` and row-Dirichlet **both fully break bs-coherence** (per-client bs
entropy ≈ 2.80 bits = log₂7, every client sees all 7 bs), yet `run_random` keeps fragmentation
score 1.0 and row-Dirichlet collapses to 0.004. The ONLY structural difference is whether whole
runs stay together. Also note row-level modes lose windows (14.34M < 14.52M): fragments shorter
than seq_len yield no window.

### 2.2 Per-client detail (the decisive pair; full 7×5 grid in the JSON)

**row-Dirichlet (α=1.0)** — each client holds scattered rows from *all 2457 runs*:

| client | rows | windows | runs touched | distinct bs | bs entropy | contig |
|---|---|---|---|---|---|---|
| c1 | 2,091,350 | 2,061,874 | 2457 | 7 | 2.807 | 0.0008 |
| c2 | 2,793,785 | 2,764,311 | 2457 | 7 | 2.807 | 0.0076 |
| c3 | 891,028 | 861,869 | 2457 | 7 | 2.807 | 0.0001 |
| c6 | 3,526,907 | 3,497,431 | 2457 | 7 | 2.807 | 0.0087 |

**`run_random`** — each client holds ~900 *whole* runs:

| client | rows | windows | runs touched | distinct bs | bs entropy | contig |
|---|---|---|---|---|---|---|
| c0 | 2,077,838 | 2,073,602 | 915 | 7 | 2.803 | 1.000 |
| c1 | 2,078,886 | 2,074,690 | 933 | 7 | 2.804 | 1.000 |
| c2 | 2,076,893 | 2,072,681 | 905 | 7 | 2.805 | 1.000 |
| c6 | 2,077,649 | 2,073,413 | 923 | 7 | 2.805 | 1.000 |

slice (3) and tr (22) coverage is full in every client for every mode — neither is the axis.

---

## 3. Factorial: run-level (intact) vs row-level (corrupt) Dirichlet, 3 architectures

All same-environment V100-SXM2-32GB, eager (`TORCHDYNAMO_DISABLE=1`), FedAvg, 100 rounds × 50
max-steps × 5/7 participation, lr 5e-4, seq-len 5, threshold 0.10, n_clients 7. **Paired by seed**;
CI95 = 10,000-sample bootstrap on per-seed paired deltas.

| arch | prec | α | run_dir (intact) mean±std | dir (row) mean±std | paired Δ (run−dir) | CI95 | seeds |
|---|---|---|---|---|---|---|---|
| LSTM | fp16 | 0.1 | 0.91395 ± 0.00111 | 0.83685 ± 0.02323 | **+0.0771** | [+0.0457, +0.0969] | 0,1,2 |
| LSTM | fp16 | 0.5 | 0.91577 ± 0.00025 | 0.79035 ± 0.03470 | **+0.1254** | [+0.0767, +0.1498] | 0,1,2 |
| LSTM | fp16 | 1.0 | 0.91589 ± 0.00097 | 0.75480 ± 0.00526 | **+0.1611** | [+0.1531, +0.1669] | 0,1,2 |
| Mamba | fp32 | 0.1 | 0.9156 | 0.8599 | **+0.0557** | [+0.0401, +0.0639] | 0,1,2 |
| Mamba | fp32 | 1.0 | 0.9168 | 0.7500 | **+0.1667** | [+0.1603, +0.1783] | 0,1,2 |
| Spiking-SSM | fp32 | 0.1 | 0.8393 | 0.6729 | **+0.1664** | [+0.1590, +0.1773] | 0,1,2 |
| Spiking-SSM | fp32 | 1.0 | 0.8625 | 0.6681 | **+0.1945** | [+0.1902, +0.2011] | 0,1,2 |

All 7 paired CIs exclude 0. The intact (`run_dir`) column is flat at the per-arch natural-by-BS
level (LSTM 0.916, Mamba 0.917, Spiking 0.853) regardless of α; the row-level column degrades by
fragmentation severity. `random_split` (LSTM, α=0.5 cell) = 0.73966 ± 0.00296 (≈ row-Dirichlet
α=1.0; both fully fragmented).

**`run_random` gate (LSTM, fp16, 100 rounds, 5 seeds):** `run_random` mean = **0.91583**
(s0 0.91660 / s1 0.91531 / s2 0.91533 / s3 0.91557 / s42 0.91634) vs same-env natural-by-BS 0.91572;
paired Δ(natural − run_random) CI95 **[−0.0003, +0.0001]** (straddles 0). Breaking bs-coherence with
sequences intact costs ~0. *Audit caveat:* a separate `prea1_run_random_smoke` cell exists at 8
rounds (0.86285, undertrained) and must be excluded; a naive seed-dedup that picks it up biases the
mean to ~0.905 — exclude all cells with `num_rounds < 100`.

Artifact paths (V100 `~/fl-oran-tmc`): LSTM factorial `artifacts/prea1_factorial/v7_lstm_fedavg_{rundir,dirichlet,randsplit}_*`;
multi-arch `artifacts/prea1_factorial_multiarch/v7_{mamba,spiking_expand2}_fedavg_{rundir,dirichlet}_*`
(`grid.log: ALL_MULTIARCH_DONE 15:11:54`); gate `artifacts/prea1_run_random/v7_lstm_fedavg_runrandom_n7_s*`.

---

## 4. Fragmentation ↔ AUC: ρ = 1 (definition, code, values, plot)

**Definition.** For a partition, within each client's `(run_id, slice_id)` group sort `step_idx`;
window *i* covers rows *i..i+4*; it is *contiguous* iff `step_idx[i+4] − step_idx[i] == seq_len−1 (=4)`.
The window-contiguity fraction = (#contiguous windows)/(#windows) over all clients. Code:
`scripts/prea1/measure_contiguity.py` → `artifacts/prea1/fragmentation/contiguity_vs_alpha.json`.

**Raw values** (row-level Dirichlet on `split.train`, N=7, seed=0) and the matched LSTM×FedAvg
test AUC (paper §6.2 trace, committed):

| partition | contiguity fraction | LSTM AUC |
|---|---|---|
| natural-by-BS | 1.0000 | 0.9159 |
| Dirichlet α=0.05 | 0.8425 | 0.8605 |
| Dirichlet α=0.10 | 0.2287 | 0.8361 |
| Dirichlet α=0.50 | 0.1524 | 0.7794 |
| Dirichlet α=1.00 | 0.0043 | 0.7571 |
| Dirichlet α=5.00 | 0.0013 | 0.7475 |

**Spearman ρ = 1.0000** (rank correlation over the 6 points; computed in
`artifacts/prea1/fragmentation/run.log` and re-verified). Both are strictly monotone in the same
order, so contiguity perfectly rank-orders AUC. Plot:
`artifacts/prea1/fragmentation/fragmentation_vs_alpha.{png,pdf}` — (a) contiguity & AUC vs α
(both fall monotonically), (b) contiguity-vs-AUC scatter on a single increasing curve.

Interpretation: the inverted-α "heterogeneity" axis is, mechanistically, a window-fragmentation
axis. Concentrated α gives a few clients dense coverage of each run (more contiguous windows);
uniform α scatters every run thinly (≈0 contiguous windows).

---

## 5. Commands, configs, hashes, environment, reproduction

**Environment (4060-dev, this machine):** Python 3.14.4, PyTorch 2.11.0+cu128, CUDA 12.8,
RTX 4060 Ti 16 GiB, driver 595.71.05, Ubuntu kernel 7.0.0-15.
**Training env (V100 cluster):** 4× Tesla V100-SXM2-32GB (sm_70), eager only (no Triton →
`TORCHDYNAMO_DISABLE=1`); fp16 for LSTM, fp32 (`--mixed-precision off`) for Mamba/Spiking (selective
scan / surrogate grads NaN in fp16). V100 fp16-eager reproduces 4080 bf16-compiled AUC to ~1e-4.

**Code hashes:** `run_random` `fdf6f97`, `run_dirichlet` `e14c295`. Paper reframe branch
`paper-seq-integrity-rewrite` `3ea8e84..0bafbe8`. Pre-registration ADR-003/PREREG-A1 `bdd4237`.

**Reproduce the analyses (local, memory-caged):**
```bash
source .venv/bin/activate
pytest tests/test_v7_run_random_partition.py tests/test_v7_run_dirichlet_partition.py --no-cov   # 15 passed
systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
  .venv/bin/python scripts/prea1/partition_audit.py        # -> partition_audit.json (§2)
systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
  .venv/bin/python scripts/prea1/measure_contiguity.py     # -> contiguity_vs_alpha.json (§4)
```

**Reproduce a factorial cell (V100, per run_factorial_grid*.sh):**
```bash
TORCHDYNAMO_DISABLE=1 .venv/bin/python experiments/run_v7_fl_arch_sweep.py \
  --arch lstm --partition-mode run_dirichlet --alpha 1.0 --seed 0 \
  --algorithm fedavg --n-clients 7 --num-rounds 100 --clients-per-round 5 \
  --max-steps-per-round 50 --batch-size 64 --lr 5e-4 --seq-len 5 --threshold 0.10 \
  --pos-weight-split train --mixed-precision fp16 --device cuda --output-dir artifacts/prea1_factorial
# Mamba/Spiking: --arch {mamba,spiking_expand2} --mixed-precision off
```
Launchers: `scripts/prea1/run_factorial_grid.sh` (LSTM, 21 cells),
`scripts/prea1/run_factorial_grid_multiarch.sh` (Mamba+Spiking, 24 cells).

---

## 6. JSAC draft rewrite — diff summary (`main..paper-seq-integrity-rewrite`)

Files: `paper/main.tex` (+75/−…), `paper/supplementary.tex`, `docs/PAPER_DRAFT.md`. main.tex
rebuilds clean (26 pp, 0 undefined refs); supplementary 3 pp. `main.tex` is canonical (carries the
5-arch Path-D); see memory `paper-canonical-file-is-main-tex`.

### 6.1 CLAIMS DELETED / WITHDRAWN
- "natural BS grouping **uniformly outperforms** … is the paper's **headline finding**" (§6.2 intro).
- "the standard assumption 'lower α ⇒ harder' is **empirically false** on this dataset" (§6.2).
- "bs grouping is **necessary** but not sufficient" + "the strong reading of (i) … (ii) is supported" (§7.1.5).
- "the empirical finding (… inverted α …) is **dataset-structural**" (supp App A.1).
- Threshold-sweep prediction "**inversion strengthens as the threshold tightens**" (supp C.4) — explicitly *withdrawn*.
- "mechanism is **partially open** … highest-priority follow-up" (§8 L7, §9).

### 6.2 CLAIMS REFRAMED → ARTIFACT
- Title → "Mind the Windows: Row-Level Non-IID Partitioning Inflates Heterogeneity Effects …".
- Abstract finding-1 → "an apparent 'inverted heterogeneity' … **is a sequence-integrity artifact**".
- §1 contribution-1 → "**A sequence-integrity pitfall in federated time-series benchmarking** (primary contribution)".
- §1 intro mechanism, §6 mechanism-preview, §7.1 intro+candidates → sequence integrity (3 candidates eliminated by run-level controls).
- §6.2 conclusion → α-curve = **fragmentation severity**, now MEASURED (ρ=1).
- §7.1.2 (per-bs KL) → real measurement, recontextualized as *not* the mechanism.
- §7.1.5 (per-bs Dirichlet) → within-BS row-scatter corrupts sequences → **supports** sequence integrity.
- §8 L7 → RESOLVED (artifact); + new **L17** (scope: benchmark-construction-specific).
- §9 conclusion → principal finding is methodological.
- 5-arch (abstract / §6.8) → "confirms the advantage" → "**same apparent advantage**, as expected of an architecture-agnostic artifact".

### 6.3 NEW MATERIAL
- §7.1 `sec:run-level`: run_random + run_dirichlet factorial table (3 archs).
- §7.x `sec:checklist`: 4-item partitioning checklist for FL-time-series benchmarks.
- §6.2 fragmentation measurement (contiguity 1.00→0.001, ρ=1).

### 6.4 PRESERVED (arch / energy / baseline contributions — unaffected)
- FedAdam saturates server-side headroom (§6.3 / §7.2); Mamba×SCAFFOLD catastrophic interaction (§6.4 / §7.3).
- Architecture leverage ≫ algorithm leverage on commodity-GPU energy (§6.5 / §7.4); NVML per-cell energy.
- 5-arch Path-D ranking (xLSTM/Mamba-3 robustness), FedSWA/FedSCAM/FedGMT sam-family (§7.5/§7.6).
- FedBN-reduces-to-FedAvg proof + 30-cell confirmation; LOTO cluster-bootstrap (§8 L15); tr-embedding no-tr ablation.
- All within-natural-partition comparisons are unaffected — only cross-partition comparisons were confounded.

---

## 7. Generality status (honest scope)

PROVEN (this dataset, 3 archs): row-level partitioning produces a severe sequence-integrity
artifact; the natural-by-BS advantage and inverted-α are that artifact. HYPOTHESIS (not proven):
that this affects other published FL-time-series benchmarks — the lit-check found row-level Dirichlet
is the de-facto default and is transplanted to time-series tasks, but per-paper susceptibility was
*not* audited. To raise generality: replicate on Twinning (commercial-traffic) or another RAN
time-series corpus with the same run-level controls + contiguity check.
