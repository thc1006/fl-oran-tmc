# Sequence Integrity Matters: Row-Level Client Partitioning Artifacts in Federated RAN Time-Series Benchmarks

**Draft v0.1 (2026-05-24).** Markdown source-of-truth for the standalone methods paper. Target
venue: a benchmarking / ML-systems / networking-methods track (TBD). Evidence:
`docs/RESULTS_DELTASEQ_LAW.md`; theory: `docs/SEQUENCE_INTEGRITY_THEORY.md`; pre-registration:
`docs/PREREG-A2-deltaseq-law.md`; bundle: `docs/SEQUENCE_INTEGRITY_EVIDENCE.md`. Numbers in this
draft are quoted from those committed artifacts.

> Note: Section 2 (Related Work) and the at-risk-benchmark table in Section 6 are STUBS pending the
> large-scale literature survey (task #88).

---

## Abstract

Federated-learning (FL) benchmarks routinely synthesise client heterogeneity by partitioning a
pooled dataset across clients with a Dirichlet distribution over *rows*. For tabular or image data
this is harmless. For **time series** — where each client subsequently builds sliding windows — it
is not: row-level partitioning scatters a sequence's consecutive timesteps across clients, so the
per-client windows are no longer real trajectories. We show that this *sequence-integrity* violation
can manufacture a striking but spurious finding — that *more* client heterogeneity *improves*
accuracy — in a federated O-RAN slice-SLA benchmark (ColO-RAN), and we dismantle it. The fragmentation
itself is universal (a deterministic consequence of the partition-then-window recipe, confirmed on two
real RAN datasets), but its impact on accuracy is **task-conditional**: it degrades AUC substantially
only when the prediction target is *sequence-essential* (a small order-invariant residual remains for
window-aggregate targets, Section 4). We introduce a cheap, partition-free diagnostic, **Δ_traj**
— the AUC an intact sequence model gains from temporal *order* over an order-shuffled baseline — and
show the fragmentation AUC gap is a monotone function of it (Spearman 0.95 across 11 ColO-RAN targets;
Spearman 1.00 / Pearson 0.99 in a controlled synthetic study). The artifact is architecture-invariant for sequence models
(LSTM, GRU) and absent for a no-sequence model, confirming it is sequence-specific. We give a corrected
partitioning protocol (partition by entity/run; synthesise heterogeneity with *run-level* Dirichlet)
and recommend Δ_traj as a pre-submission screen. On a second real dataset (Open RAN Commercial Traffic
Twinning) the mechanism replicates but the AUC impact does not — exactly as the diagnostic predicts for
its persistent targets — serving as a negative control.

## 1. Introduction

Client heterogeneity (non-IID data) is the defining difficulty of FL, and benchmarks emulate it by
splitting a pooled dataset with a Dirichlet(α) distribution: small α → skewed, large α → near-IID.
This recipe is near-universal and, for i.i.d.-sample data (images, tabular rows), statistically sound.

Time-series FL inherits the recipe but adds a step: each client turns its rows into fixed-length
sliding windows for a sequence model. Here the recipe quietly breaks. If rows are assigned to clients
independently (Dirichlet/random over rows), a run's consecutive timesteps are *scattered* across
clients; when a client then windows its rows (sorted by time), each "window" is a set of
non-consecutive snapshots, and the label attached to it belongs to a temporally mismatched step. The
sequence model is trained on temporally incoherent (window, label) pairs.

We encountered this concretely. In a federated O-RAN slice-SLA prediction benchmark on the
ColO-RAN/Colosseum testbed, partitioning by Dirichlet over rows produced an *inverted* heterogeneity
curve: AUC rose as α fell (more skew → better accuracy), the opposite of the FL norm. The tempting
reading — that RAN telemetry has special cell-conditional structure that standard FL heterogeneity
cannot exploit — does not survive controls. A **run-level** Dirichlet (whole runs assigned with the
same skew) recovers the natural-partition accuracy and erases the inversion; breaking entity
coherence while keeping runs intact costs ~0 AUC. The apparent inversion is a **sequence-integrity
artifact of how the non-IID partition is constructed**, not a property of the data.

**Contributions.**
1. We identify and characterise the **sequence-integrity pitfall** of row-level partitioning in
   time-series FL benchmarks, and a simple **fragmentation audit metric** to detect it.
2. We show the artifact's AUC impact is **task-conditional**, and introduce **Δ_seq / Δ_traj**,
   capacity-matched and partition-free diagnostics that *predict* the fragmentation gap
   (Spearman 0.92 / 0.95 on ColO-RAN; 0.99 synthetic).
3. We give the **mechanism** (a target's predictability splits into a partition-invariant *run-rate*
   and a partition-vulnerable *trajectory*; autocorrelation of the target source is a pre-diagnostic)
   and a careful theory (the diagnostic is monotone, **not** a bound — the natural `gap ≤ Δ_seq`
   conjecture is falsified by a synthetic counterexample).
4. We establish **architecture-invariance** (LSTM, GRU) and a **sanity control** (a no-sequence model
   shows ~0 gap, isolating a small residual window-content component for aggregate targets).
5. We give a **corrected protocol** (entity/run-level partitioning; run-level Dirichlet for synthetic
   heterogeneity; Δ_traj pre-screen) and a **negative control** on a second real RAN dataset.

## 2. Related Work

Four threads bound our contribution; we are careful to claim only what is new.

**(a) The Dirichlet partitioning convention.** Simulating non-IID clients by a Dirichlet
distribution over *labels/rows* is the de-facto FL-benchmark standard — codified by NIID-Bench
[Li et al., ICDE 2022] (label/feature/quantity skew, default α=0.5) and shipped as the default
`DirichletPartitioner` in FL frameworks (Flower Datasets, FedML). It is statistically sound for
**i.i.d.-sample** data (images, tabular rows). Our point is that time-series FL *inherits* this
recipe and then adds a per-client windowing step, where it silently breaks. The "more heterogeneity
helps" reading we dismantle is especially suspect because the **established FL norm is the opposite**:
recent large-scale assessments find heterogeneity *harms* accuracy, with sharp drops past high skew
[A Thorough Assessment of the Non-IID Data Impact, arXiv:2503.17070, 2025] — so an *inverted* curve
should itself be treated as a red flag for an artifact.

**(b) Windowing leakage in time series.** A known pitfall is generating sliding windows *before*
the train/test split, leaking future context into training [e.g., "Hidden Leaks in Time Series
Forecasting," arXiv:2512.06932]. This is **a different failure mode**: it concerns *temporal
causality across the train/test boundary* in **centralized** pipelines (it inflates scores), and
that literature does not discuss federated client partitioning. Our pitfall is *within-training*,
*federated-specific*, and *degrades* (not inflates) — it is about per-client **window integrity
across clients**, orthogonal to train/test leakage.

**(b′) Temporal-order ablation (our Δ_traj method has precedent).** Shuffling timesteps to probe a
model's reliance on temporal order is an established technique — temporal-order-verification pretext
tasks [Shuffle and Learn, Misra et al.] and segment-shuffle representation learning [Segment, Shuffle,
and Stitch, NeurIPS 2024, arXiv:2405.20082], which notes that non-adjacent timesteps can carry strong
dependencies. We do **not** claim the shuffle ablation as novel; we reuse it for a *new purpose* —
predicting and screening for the federated fragmentation gap — and we show (Sections 5.4, 7) that the
shuffle destroys *order* whereas row-level fragmentation destroys *consecutiveness*, so Δ_traj
over-predicts for order-but-gap-robust targets. (Relatedly, attention's permutation-invariance, a
known limitation for temporal-order modeling, is why our no-positional-encoding Transformer is
order-blind, Section 5.3.)

**(c) Temporal fragmentation noted as a deployment challenge.** Some FL works observe that local
time-series can be "fragmented" and that this degrades feature extraction (e.g., unsupervised
federated anomaly-detection methods that explicitly target "degraded detection performance caused
by temporal fragmentation in distributed environments"). We build on this but differ in three ways:
we show the fragmentation is **manufactured by the row-level partitioning recipe as a
benchmark-construction artifact** (not an intrinsic deployment constraint); we show it is
**task-conditional** with a **predictive diagnostic** (Δ_traj); and we demonstrate it can produce a
**qualitatively false finding** (an inverted heterogeneity curve a researcher would misread as
scientific structure). FedSL [arXiv:2011.03180] instead studies *legitimately* sequentially-split
sequences (consecutive segments naturally residing on different clients) — a different setting from
artificial benchmark fragmentation.

**(d) FL time-series / RAN benchmarks.** FL traffic/throughput forecasting on cellular data
[e.g., 5G base-station forecasting; "Benchmarking Federated Learning for Throughput Prediction,"
arXiv:2508.08479], O-RAN toolchains [OpenRAN Gym, arXiv:2202.10318; ColO-RAN, arXiv:2112.09559], FL
HAR (WISDM/PAMAP2/USC-HAD), and FL smart-meter load forecasting are the application landscape. Their
susceptibility hinges entirely on the *partition-then-window order* (Section 6); entity-partitioned
benchmarks (by base station / subject / meter / UE) are safe, row-Dirichlet ones are at risk.

## 3. The Sequence-Integrity Pitfall

**Setup.** A pooled multivariate series is split into clients, then each client builds length-`L`
sliding windows within its rows (sorted by step); a window predicts a next-step label. *Intact*
partitions (natural entity, or run-level Dirichlet) keep whole `(run, slice)` groups together →
genuine trajectories. *Row-level* partitions scatter a group's rows → fragmented windows.

**Audit metric.** The fragmentation score = fraction of per-client windows whose `step_idx` are
contiguous. It is `1.0` for intact and `~0` for row-level on both datasets we study:

| dataset | intact | random_split (row) | dirichlet (row) |
|---|---|---|---|
| ColO-RAN | 1.000 | ~0 | ~0 |
| Twinning | 1.0000 | 0.0002 | 0.0031 |

The audit is **necessary but not sufficient** for an AUC impact — a window can be fragmented yet the
task unharmed (Section 4). Fragmentation alters three things: within-window **order**, *which* steps
**populate** the window, and the window-to-label **alignment**.

## 4. The Δ_seq / Δ_traj Diagnostic and Mechanism

**Diagnostics (capacity-matched, partition-free).**
- **Δ_seq** = AUC(seq LSTM) − AUC(same LSTM on a length-1 window) — the value of the multi-step
  window over a single instantaneous read. Conflates order with order-free multi-step averaging.
- **Δ_traj** = AUC(seq LSTM) − AUC(same LSTM on order-shuffled windows) — isolates the *order /
  trajectory* value (the partition-vulnerable part), because shuffling destroys order while keeping
  the multiset / run-rate. The cleaner predictor.

**Gap decomposition.** The fragmentation gap = AUC(intact) − AUC(row) splits into (i) a **trajectory
component** (only sequence models access it; ∝ Δ_traj; dominant for white-noise targets) and (ii) a
small **window-content component** (any model, only for *aggregate* targets, because fragmentation
changes which steps are in the window). A no-sequence mean-pool MLP isolates (ii): `0.000` for a
point target, a real `0.025` for a 5-step-smoothed target (≪ the LSTM's 0.082).

**Mechanism: run-rate vs trajectory.** A next-step target's predictability splits into a
slowly-varying per-run **run-rate** (estimable from any sample → order-free → partition-invariant)
and a short-horizon **trajectory** (needs consecutive ordered steps → partition-vulnerable). The
target source's lag-1 autocorrelation is a pre-diagnostic: persistent sources (tx_brate 0.98,
dl_buffer 0.999, dl_mcs 0.90) are run-rate-dominated (Δ_traj ≈ 0, gap ≈ 0); the white-noise BLER
source (0.02), predictable only via the multivariate channel-state trajectory, has Δ_traj 0.22 and
gap 0.15.

**A diagnostic, not a bound (honesty).** The gap is monotone in Δ_traj but we do **not** claim a
closed-form bound. The conjecture `gap ≤ Δ_seq` is *falsified*: with a pure-trajectory synthetic
target (no run-rate fallback), fragmented training is actively mis-led below the single-step ceiling
(`gap > Δ_seq`), whereas real BLER (with a run-rate fallback) gives `gap < Δ_seq`. We claim a
monotone empirical diagnostic with a mechanism, validated across synthetic + two real datasets +
architectures.

## 5. Experiments

**5.1 Synthetic causal control.** Multivariate AR runs; a tunable knob λ interpolates a target between
instantaneous and 4-step-trajectory dependence; pos-rate held ~0.5, all else fixed. The gap is monotone
in Δ_seq (Pearson **0.986**) and the `gap ≤ Δ_seq` bound is falsified at high λ. This isolates
sequence-essentiality as the causal driver.

**5.2 ColO-RAN Δ_seq law (main result).** 11 targets spanning Δ_seq ∈ [0.01, 0.23], 5 seeds, OOD-by-tr
split. The fragmentation gap is monotone in Δ_seq (Pearson 0.943, Spearman 0.918, OLS slope CI95
[0.706, 1.113] excludes 0) and cleaner in Δ_traj (Pearson 0.974, Spearman 0.945). BLER targets:
gap 0.08–0.16 (5-seed paired-bootstrap CIs exclude 0); CQI/MCS/buffer/throughput targets: gap ≈ 0.
The point target `bler_th10` reproduces the established ColO-RAN gap (intact 0.908 / row 0.757,
gap +0.152, consistent with the original ~0.91/~0.75). Δ_traj pulls the lone deviator (`brate_med`,
persistent) onto the line. [Figure: `artifacts/prea1/twinning/deltaseq_law.pdf` — gap vs Δ_seq; a
gap-vs-Δ_traj panel is pending.]

**5.3 Architecture invariance + sanity.** A no-sequence mean-pool MLP shows gap ≈ 0 even for
high-Δ_seq BLER (max 0.025 for the aggregate target; order below the LSTM's 0.16) → the gap is
sequence-specific. LSTM and GRU both learn the BLER trajectory (intact ≈ 0.90) and show the gap
tracking Δ_seq. Our small Transformer has no positional encoding, so self-attention + mean-pooling is
permutation-invariant — it is **order-blind by construction** (intact bler 0.69 ≈ MLP 0.66) and thus
acts as a *second* no-sequence sanity (gap ≈ 0), not an order-using sequence model; a
positional-encoded Transformer is untested (attention's permutation-invariance impeding temporal-order
modeling is a known limitation).

**5.4 Twinning negative control.** On Open RAN Commercial Traffic Twinning (entity = UE; real Madrid
LTE traffic twinned via Colosseum), the fragmentation mechanism replicates (audit intact 1.0 vs row
~0, 6858 (run,UE) groups, 31M rows) but the AUC impact does not (next-step CQI/MCS gap ≈ 0) — exactly
as the diagnostic predicts for Twinning's persistent targets. The mechanism is universal; the impact
is task-conditional. A targeted hunt for a *within-Twinning* sequence-essential positive (channel
drop-event targets) reinforces this and uncovers a diagnostic caveat: such targets are Δ_traj-positive
(0.07–0.13) yet *still* show no fragmentation gap (+0.002), because row-level windowing keeps each
window sorted-ascending (only gappy) — so it **preserves order and destroys only consecutiveness**
(the row AUC tracks the ordered seqC, not the shuffled baseline), and a drop-event needs order but
tolerates gaps (Section 7). Even Twinning's sequence-essential targets are fragmentation-robust.

## 6. A Correct Partitioning Protocol

1. **Partition by natural entity / run** (each holds its intact stream) — the deployment-correct
   choice and free of the artifact.
2. **To synthesise heterogeneity, use run-level Dirichlet** (assign whole runs with a skewed
   distribution), never row-level Dirichlet. This preserves window integrity while still inducing
   non-IID client distributions.
3. **Pre-screen with Δ_traj** (cheap, no partitioning), then **confirm with the run-level-vs-row-level
   control**: a Δ_traj ≈ 0 benchmark is safe; a large Δ_traj *flags* a benchmark to investigate, but
   Δ_traj can over-predict (it destroys the last-step anchor that END-aligned fragmentation preserves —
   Section 7), so the partition control is the decisive test, not Δ_traj alone.

**Which benchmarks are at risk.** Susceptibility is determined by the *partition-then-window order*,
not the domain. We classify the recipes (per-paper audit requires their code; many papers do not
state the order explicitly, so "ambiguous" entries are *potentially* at risk):

| partition recipe | window integrity | at-risk? | representative usage |
|---|---|---|---|
| Partition **raw rows** by Dirichlet/random, **then** window per client | fragmented | **YES** (if Δ_traj large) | the NIID-Bench/Flower Dirichlet recipe applied to a raw TS; ColO-RAN `fl_v7` (this work) |
| **Window first**, then distribute **whole windows** by Dirichlet | intact | No (sequence-integrity safe) | sample-level FL HAR where a "sample" is a pre-cut window |
| Partition by **entity** (subject / base station / meter / UE) | intact | No | FL HAR by subject (WISDM/PAMAP2/USC-HAD); 5G traffic by base station; smart-meter by meter; ColO-RAN natural-by-BS |
| Partition by **temporal segment** (e.g., calendar quarter) | intact within segment | Mostly no | FL financial / seasonal forecasting |
| **Legitimately** sequentially-split sequences across clients | cross-client (handled) | Different problem | FedSL |

The dangerous case is specifically **partition-then-window on raw rows** — the literal transplant of
the image/tabular Dirichlet convention onto a time series. Because the AUC impact is task-conditional
(only sequence-essential targets, large Δ_traj), a benchmark can sit in the at-risk row yet show no
symptom on an order-free target — which is exactly why the pitfall is easy to miss. **Recommendation:
report the partition-then-window order explicitly, and report Δ_traj for the benchmark's target.**

## 7. Limitations and Threats to Validity

- The Δ_seq/Δ_traj law is an **empirical** monotone diagnostic, not a proven bound (the synthetic
  super-linearity vs real sub-linearity shows the constant is data-dependent).
- **Δ_traj over-predicts for last-step-anchored targets** (Twinning hunt): drop-event targets had
  Δ_traj 0.07–0.13 but fragmentation gap ≈ 0. The shuffle destroys within-window *order* (Δ_traj = the
  order-value), whereas row-level windowing preserves order (windows stay sorted-ascending, only gappy)
  and destroys only *consecutiveness* — the gap is the smaller consecutiveness-value, and indeed the
  row AUC (0.69/0.81) ≈ the ordered seqC (0.70/0.82) ≫ the shuffled (0.63/0.69). So **Δ_traj ≥ gap**
  empirically (it conflates order with consecutiveness); it is a cheap partition-free **screen**, not a
  perfect predictor, and the run-level-vs-row-level partition control is the ground truth.
- Architecture coverage is LSTM + GRU (order-using, both confirm the law) + a mean-pool MLP and a
  no-positional-encoding Transformer (both permutation-invariant → order-blind → gap ≈ 0, two
  consistent no-sequence sanities). A *positional-encoded* Transformer is untested; we do not claim
  the law over order-using attention models, only over the two recurrent models tested.
- The Twinning AUC-impact null is a 1-seed smoke (mechanism audit + Δ_seq prediction carry it; 5-seed
  CI is straightforward hardening). ColO-RAN gaps are 5-seed paired bootstrap (n=5; CIs tight).
- Partition client counts differ by mode (iid = 7 BS; run/row Dirichlet = 8) — immaterial to the
  global-test gap but disclosed.
- `ul_sinr` was dropped pre-results (degenerate: median at a mass point → undefined AUC; documented
  in PREREG-A2, not a result-based exclusion).
- Per-paper susceptibility of specific prior benchmarks depends on their exact partition-then-window
  order, which the survey (Section 2/6) classifies but does not re-run.

## 8. Conclusion

Row-level Dirichlet partitioning — a near-universal FL-benchmark convention — can manufacture
spurious findings in time-series FL by fragmenting per-client windows. The pitfall is real but
*task-conditional*: it bites sequence-essential tasks (large Δ_traj) and spares order-free ones,
which is why it has gone unnoticed. We provide the audit metric, the Δ_traj diagnostic, the
mechanism, and a corrected protocol. We urge time-series FL benchmarks to partition by entity/run
(or run-level Dirichlet) and to report Δ_traj.

## Reproducibility

Pre-registration `docs/PREREG-A2-deltaseq-law.md`; results + commands `docs/RESULTS_DELTASEQ_LAW.md`;
theory `docs/SEQUENCE_INTEGRITY_THEORY.md`; scripts under `scripts/prea1/` (synthetic, ColO-RAN sweep,
Δ_traj addendum, arch-invariance, aggregator, Twinning audit + AUC-impact; `window_cache` with a
bit-exact test). All runs: local RTX 4060 Ti (memory-caged) + the V100 ColO-RAN sweep.
