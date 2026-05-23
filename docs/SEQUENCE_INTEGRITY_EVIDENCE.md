# Sequence-Integrity Confound — Auditable Evidence Bundle (v2)

**Main line (locked 2026-05-22):** *row-level client partitioning can create sequence-integrity
artifacts in federated RAN time-series benchmarks.*
- For **ColO-RAN**: this is a **proven diagnosis** — the apparent "inverted heterogeneity"
  (natural-by-BS beating every Dirichlet α) is a sequence-integrity artifact, not a structural
  property of RAN telemetry.
- For **other FL time-series papers**: this is a **hypothesis only** — raising it to a claim
  requires replication on Twinning or another RAN time-series dataset (§8).

**Superseded / paused:** Paper A "RAN cell-conditional structure" thesis is falsified (do not
re-attempt). Paper B (offline xApp), Twinning, TimeRAN, xApp recommender are NOT active.
**Gate:** JSAC is not submitted until this bundle is accepted; the paused programs do not start.

Branch `paper-seq-integrity-rewrite` (PR [#29](https://github.com/thc1006/fl-oran-tmc/pull/29)).
Environment, hashes, rerun in §6.

**Leak-defense — no-BLER ablation (2026-05-24).** `ul_bler` is both a V3_CONTINUOUS feature and the
SLA target source (`y = 1[ul_bler_{t+1} > 0.10]`). An adversarial review flagged this; a drop-BLER
ablation (`--drop-continuous dl_bler,ul_bler`, LSTM, 5 seeds, V100; `scripts/prea1/run_nobler_ablation.sh`
+ `aggregate_nobler.py`) **CONFIRMS** the inverted-α gap is channel-state sequence-integrity, not a
BLER-rate confound: removing both BLER channels from the model input leaves the run-level≫row-level gap
unchanged — α=1.0: **+0.162** (CI95 [+0.158,+0.165]) vs with-BLER +0.161; α=0.1: **+0.079** (CI95
[+0.060,+0.095]) vs +0.077 — while absolute AUC drops only ~0.005. Nuance (the earlier over-claim,
corrected): the *last-value* BLER predictor = 0.5133 ≈ chance, but the *5-step rolling-mean* BLER =
0.6258 carries a **run-level-rate** signal; that signal is partition-invariant (scattered ≈ consecutive
samples estimate the same run rate) and so cancels in the gap — the ablation settles it empirically.
Reflected in main.tex §6.7 + §8 L18. 25/25 cells verified (correct `drop_continuous`, no NaN, seeds 0-4).

---

## 1. Code diffs — `run_random`, `run_dirichlet`, fragmentation audit

| commit | mode / artifact | partition.py | other files | what it does |
|---|---|---|---|---|
| `fdf6f97` | `run_random` | **+44** | fl_v7.py +16 (dispatch + `runrandom` naming), run_v7_fl_arch_sweep.py +4 (CLI choice), test +159, confound-doc +92 | assign each **whole `(run_id, slice_id)` group** to one of N clients by **greedy least-loaded-by-rows** over a seed-shuffled group order → groups never split; bs-coherence broken; **sequences intact** |
| `e14c295` | `run_dirichlet` | **+48** | fl_v7.py +14 (`rundir_a<tag>` naming), run_v7 +2, factorial launcher +46, test +112, confound-doc +42 | per-slice `Dir(α)` over **whole `(run_id, slice_id)` groups** → Dirichlet-skewed allocation; **sequences intact** |
| (this bundle) | fragmentation + audit | — | `scripts/prea1/measure_contiguity.py`, `scripts/prea1/partition_audit.py` | no-training contiguity / structure audits (§3, §5) |

Full hunks: `artifacts/prea1/diff_run_random_partition.patch`, `diff_run_dirichlet_partition.patch`.
Production order (`fl_v7.py`): `ood_split_by_tr` → `_partition(split.train)` → per-client
`build_run_sequences` = **partition-then-window with no contiguity guard** (the corruptible step).

---

## 2. Partition integrity tests — `21 passed`

```
pytest tests/test_v7_run_random_partition.py tests/test_v7_run_dirichlet_partition.py \
       tests/test_v7_partition_metadata.py --no-cov
=> 21 passed   (8 run_random + 7 run_dirichlet + 6 metadata)
```

Three invariants the user asked for, each asserted by a test:

| invariant | where | assertion |
|---|---|---|
| **(i) `(run_id, slice_id)` not split across clients** | `test_v7_run_{random,dirichlet}_partition.py` | every group's rows live in exactly one client shard |
| **(ii) valid windows not fragmented** | same | within each client, every group's `step_idx` is contiguous (no gappy windows) |
| **(iii) partition kwargs land in artifact metadata, not leaked to `algo_kwargs`** | `test_v7_partition_metadata.py` (NEW) | `asdict(V7Config)` (= summary.json `config`) carries `partition_mode`/`alpha`/`n_clients`; none leak into `algo_kwargs` (the strip step; the 60-cell-crash lesson) |

Empirical confirmation on a real cell: `prea1_factorial/v7_lstm_fedavg_rundir.../summary.json`
has `config.partition_mode="run_dirichlet"`, `config.alpha=0.1`, `config.algo_kwargs={}`.

---

## 3. Partition audit table (5 modes, client-level)

Source: `scripts/prea1/partition_audit.py` → `artifacts/prea1/partition_audit.json`
(OOD train split `tr∈0..21`, 14.5M rows, seq_len=5, N=7, seed=0). **Fragmentation score = fraction
of per-client sliding windows whose 5 rows have contiguous `step_idx`.**

### 3.1 Per-mode aggregate

| mode | total windows | fragmentation score | bs-coherence (per-client) | sequences |
|---|---|---|---|---|
| natural (iid by-BS) | 14,515,074 | **1.0000** | **1 bs/client** | intact |
| row-random (`random_split`) | 14,338,218 | **0.0004** | all 7 bs | **corrupted** |
| row-Dirichlet (α=1.0) | 14,345,955 | **0.0043** | all 7 bs | **corrupted** |
| `run_random` | 14,515,074 | **1.0000** | all 7 bs | intact |
| `run_dirichlet` (α=1.0) | 14,515,074 | **1.0000** | all 7 bs | intact |

### 3.2 Client-level detail with distributions (client 0; full 7×5 grid + per-value dists in JSON)

| mode | num_rows | num_sequences | num_runs | bs distribution | slice distribution | tr distinct | frag |
|---|---|---|---|---|---|---|---|
| natural | 2,075,530 | (∑≈2.07M) | many | **{bs1: 2.08M}** (1 bs) | {692k, 693k, 689k} | 22 | 1.000 |
| row-Dirichlet α1.0 | 2,091,350 | 2,061,874 | **2457** (all) | {bs1..7: 293k–305k} | {724k, 451k, 916k} skew | 22 | 0.0008 |
| run_random | 2,077,838 | 2,073,602 | **915** (whole) | {bs1..7: 262k–325k} | {713k, 692k, 673k} | 22 | 1.000 |

**The decisive comparison (rows 2 vs 3):** row-Dirichlet and `run_random` have *identical* structure
on every axis the literature calls "heterogeneity" — both spread all 7 bs across each client (bs
entropy ≈ 2.80 bits = log₂7), both cover all 3 slices and 22 tr. They differ on **one** axis:
row-Dirichlet scatters rows from all 2457 runs (frag 0.0008) while `run_random` holds ~915 whole
runs (frag 1.000). That single axis — sequence integrity — flips the AUC (§4). Natural is the only
mode that also preserves bs-coherence (1 bs/client), but §4 shows that is *not* what matters.

---

## 4. Result table — 3 archs × 5 modes (AUC, std, seeds, paired Δ/CI, paths)

All same-environment V100-SXM2-32GB, eager (`TORCHDYNAMO_DISABLE=1`), FedAvg, 100 rounds × 50
max-steps × 5/7 participation, lr 5e-4, seq-len 5, threshold 0.10, n=7. LSTM fp16; Mamba/Spiking
fp32 (`--mixed-precision off`; selective-scan / surrogate-grad NaN in fp16). Paired CI95 = 10k
bootstrap on per-seed paired deltas.

| arch | mode | α | AUC mean ± std | seeds | paired Δ vs row-counterpart | CI95 | artifact dir |
|---|---|---|---|---|---|---|---|
| **LSTM** | natural | – | 0.91572 (≈, E2) | 5 | — | — | `prea1_e2/` |
| LSTM | row-random | – | 0.73966 ± 0.00296 | 3 | — | — | `prea1_factorial/…randsplit` |
| LSTM | row-Dirichlet | 0.1 | 0.83685 ± 0.02323 | 0,1,2 | (run_dir−dir) **+0.0771** | [+0.0457,+0.0969] | `prea1_factorial/…dirichlet` |
| LSTM | row-Dirichlet | 1.0 | 0.75480 ± 0.00526 | 0,1,2 | (run_dir−dir) **+0.1611** | [+0.1531,+0.1669] | `prea1_factorial/` |
| LSTM | run_random | – | 0.91583 | 0,1,2,3,42 | (natural−run_random) **−0.0001** | [−0.0003,+0.0001] | `prea1_run_random/` |
| LSTM | run_dirichlet | 0.1 | 0.91395 ± 0.00111 | 0,1,2 | — | — | `prea1_factorial/…rundir` |
| LSTM | run_dirichlet | 0.5 | 0.91577 ± 0.00025 | 0,1,2 | — | — | `prea1_factorial/` |
| LSTM | run_dirichlet | 1.0 | 0.91589 ± 0.00097 | 0,1,2 | — | — | `prea1_factorial/` |
| **Mamba** | natural | – | 0.9165 (Phase-5, *cross-env*) | 10 | — | — | `v7_stage2_full/` |
| Mamba | row-Dirichlet | 0.1 | 0.8599 | 0,1,2 | (run_dir−dir) **+0.0557** | [+0.0401,+0.0639] | `prea1_factorial_multiarch/` |
| Mamba | row-Dirichlet | 1.0 | 0.7500 | 0,1,2 | (run_dir−dir) **+0.1667** | [+0.1603,+0.1783] | `prea1_factorial_multiarch/` |
| Mamba | run_dirichlet | 0.1 | 0.9156 | 0,1,2 | — | — | `prea1_factorial_multiarch/` |
| Mamba | run_dirichlet | 1.0 | 0.9168 | 0,1,2 | — | — | `prea1_factorial_multiarch/` |
| Mamba | row-random / run_random / same-env natural | – | **NOT RUN** | – | — | — | *gap — see §8* |
| **Spiking-SSM** | natural | – | 0.8529 (Phase-5, *cross-env*) | 10 | — | — | `v7_stage2_full/` |
| Spiking | row-Dirichlet | 0.1 | 0.6729 | 0,1,2 | (run_dir−dir) **+0.1664** | [+0.1590,+0.1773] | `prea1_factorial_multiarch/` |
| Spiking | row-Dirichlet | 1.0 | 0.6681 | 0,1,2 | (run_dir−dir) **+0.1945** | [+0.1902,+0.2011] | `prea1_factorial_multiarch/` |
| Spiking | run_dirichlet | 0.1 | 0.8393 | 0,1,2 | — | — | `prea1_factorial_multiarch/` |
| Spiking | run_dirichlet | 1.0 | 0.8625 | 0,1,2 | — | — | `prea1_factorial_multiarch/` |
| Spiking | row-random / run_random / same-env natural | – | **NOT RUN** | – | — | — | *gap — see §8* |

**Reading.** Intact (run_dir / run_random / natural) ≈ per-arch ceiling and flat across α; corrupt
(row-Dir / row-random) collapse by fragmentation severity. All 7 `run_dir−dir` paired CIs exclude 0;
the LSTM `run_random` gate straddles 0 (breaking bs-coherence with sequences intact costs ~0).

**Honest gaps (do not over-read):** (a) for Mamba/Spiking, "run_dir ≈ natural" uses the Phase-5
natural (4080/bf16) — *cross-environment*; only LSTM has a same-env natural+run_random gate.
(b) Mamba/Spiking row-random and run_random were not run. (c) factorial n=3 (gate n=5). These are
the PR #29 adversarial gaps; §8 lists the ~cheap V100 cells that close them. *Audit caveat:* an
8-round `prea1_run_random_smoke` cell (0.86285) exists — exclude `num_rounds<100` (a naive
seed-dedup that includes it biases the gate mean to ~0.905).

---

## 5. Fragmentation ρ — definition, code, raw values, plot

**Definition.** For a partition, within each client's `(run_id, slice_id)` group sort `step_idx`;
window *i* covers rows *i..i+4*; it is *contiguous* iff `step_idx[i+4] − step_idx[i] == seq_len−1 (=4)`.
Window-contiguity fraction = (#contiguous)/(#windows) over all clients. Code:
`scripts/prea1/measure_contiguity.py` → `artifacts/prea1/fragmentation/contiguity_vs_alpha.json`.

| partition | contiguity fraction | LSTM×FedAvg AUC (§6.2 trace) |
|---|---|---|
| natural-by-BS | 1.0000 | 0.9159 |
| Dirichlet α=0.05 | 0.8425 | 0.8605 |
| Dirichlet α=0.10 | 0.2287 | 0.8361 |
| Dirichlet α=0.50 | 0.1524 | 0.7794 |
| Dirichlet α=1.00 | 0.0043 | 0.7571 |
| Dirichlet α=5.00 | 0.0013 | 0.7475 |

**Spearman ρ = 1.0000** over these 6 points (both strictly monotone in the same order). Plot:
`artifacts/prea1/fragmentation/fragmentation_vs_alpha.{png,pdf}`.

**Causal vs correlational (important).** ρ=1 is *corroboration*, not proof: α drives **both**
contiguity and AUC, so the correlation is a common-cause. The **causal** lever is the run-level
control (§4): `run_dirichlet` varies α/heterogeneity with contiguity held at 1.0 → AUC stays flat
→ α/heterogeneity is **not** the cause; fragmentation (the only thing differing between run_dir and
dir) is. Why corrupted AUC is 0.74 (not 0.50): a fragmented window keeps its 5 rows' static
`(run, slice)` context (marginal signal) but loses temporal dynamics — so the model retains the
static-signal floor and loses the ~0.16 temporal component.

---

## 6. Commands, config, hashes, environment, rerun

**Local (4060-dev):** Python 3.14.4, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 4060 Ti 16 GiB, driver
595.71.05, kernel 7.0.0-15. **Training (V100 cluster):** 4× Tesla V100-SXM2-32GB (sm_70), eager
only; LSTM fp16, Mamba/Spiking fp32. **Hashes:** run_random `fdf6f97`, run_dirichlet `e14c295`,
ADR-003/PREREG-A1 `bdd4237`, reframe branch `3ea8e84..b8bfd23`.

```bash
# analyses (local, memory-caged — heavy local jobs MUST be caged, see memory local-box-resource-cap)
source .venv/bin/activate
pytest tests/test_v7_run_random_partition.py tests/test_v7_run_dirichlet_partition.py \
       tests/test_v7_partition_metadata.py --no-cov           # 21 passed
systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
  .venv/bin/python scripts/prea1/partition_audit.py           # -> partition_audit.json (§3)
systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 -- \
  .venv/bin/python scripts/prea1/measure_contiguity.py        # -> contiguity_vs_alpha.json (§5)

# one factorial cell (V100; launchers scripts/prea1/run_factorial_grid{,_multiarch}.sh)
TORCHDYNAMO_DISABLE=1 .venv/bin/python experiments/run_v7_fl_arch_sweep.py \
  --arch lstm --partition-mode run_dirichlet --alpha 1.0 --seed 0 --algorithm fedavg \
  --n-clients 7 --num-rounds 100 --clients-per-round 5 --max-steps-per-round 50 \
  --batch-size 64 --lr 5e-4 --seq-len 5 --threshold 0.10 --pos-weight-split train \
  --mixed-precision fp16 --device cuda --output-dir artifacts/prea1_factorial
# Mamba/Spiking: --arch {mamba,spiking_expand2} --mixed-precision off
```

---

## 7. JSAC manuscript impact report

`main.tex` is canonical (carries 5-arch Path-D); rebuilds clean 26 pp; supplementary 3 pp.

### 7.1 DELETE (claims that no longer hold)
- §6.2 "headline finding" + "lower α ⇒ harder is **empirically false**".
- §7.1.5 "bs grouping is **necessary** … (ii) is supported".
- supp App A.1 "the empirical finding is **dataset-structural**".
- supp C.4 prediction "**inversion strengthens as the threshold tightens**".
- §8 L7 / §9 "mechanism is **partially open**".

### 7.2 CHANGE → artifact framing (done on branch)
- Title; abstract finding-1; §1 contribution-1 (now primary methodological contribution).
- §6.2 → α-curve = *measured* fragmentation severity (ρ=1); §6/§7.1 mechanism = sequence integrity.
- §7.1 + new `sec:run-level` (factorial table) + `sec:checklist`; §7.1.2 KL recontextualized; §7.1.5 → supports sequence integrity; §8 L7 RESOLVED + new L17; §9 methodological.
- 5-arch (abstract/§6.8) "confirms advantage" → "same apparent advantage (architecture-agnostic artifact)".

### 7.3 KEEP but RE-STATE (results valid; remove inverted-α dependency in their framing)
| result | status | required re-statement |
|---|---|---|
| FedAdam saturates server-side headroom (§6.3/§7.2) | valid | re-state on the **natural (sequence-intact)** partition; drop any "small because heterogeneity is benign here" gloss tied to the inversion |
| Mamba × SCAFFOLD catastrophic interaction (§6.4/§7.3) | valid (at α∈{0.1,0.5}) | note it co-occurs with the **row-level** Dirichlet partition; re-verify it persists under **run-level** Dirichlet before claiming it is a pure algo×arch effect (open check) |
| Architecture ≫ algorithm leverage on energy (§6.5/§7.4) | valid (energy is partition-independent) | re-state without leaning on the inverted-α as the "heterogeneity axis"; energy/latency claims unaffected |
| 5-arch Path-D ranking (§6.8) | valid (absolute AUC) | re-state as ranking under SAM-family algos; the cross-partition "advantage persists" wording → "same apparent (artifact) advantage" |
| FedBN ≡ FedAvg proof + LOTO cluster-bootstrap (§8) | valid | unaffected; LOTO σ_tr applies to absolute AUC as before |
| tr-embedding no-tr ablation | valid | unaffected (it concerns the embedding, not the partition) |

### 7.4 UNAFFECTED
All **within-natural-partition** comparisons (arch vs arch, algo vs algo at natural-by-BS) — only
**cross-partition** comparisons were confounded.

---

## 8. Honest gaps + scope (what would raise rigor / generality)

**To close the PR #29 adversarial gaps (cheap, ~strengthens the proven ColO-RAN claim):**
- same-env (V100) natural-by-BS for **Mamba/Spiking** → makes "run_dir ≈ natural" same-env for all 3 archs (currently cross-env for 2/3).
- Mamba/Spiking **run_random** + **row-random** cells → completes the 5-mode × 3-arch table.
- bump factorial seeds 3 → ≥5.

**To raise generality (ColO-RAN proven → general claim):** the *mechanism* (row-partition fragments
windows) is general by construction — provable analytically / by a no-training contiguity audit on
any (entity, time) corpus. The *impact* (fragmentation degrades AUC) is dataset-dependent and needs
**Twinning or another RAN time-series dataset** to replicate (pre-register CONFIRM/REFUTE first).
Until then: ColO-RAN = proven diagnosis; broad-benchmark effect = hypothesis.
