# Theory backbone — why row-level partitioning manufactures a task-conditional artifact

Rigorous-but-bounded account for the standalone *"Sequence Integrity Matters"* paper. We state
a **mechanism**, a **decomposition**, and two **empirical diagnostics** — and we are explicit about
what is *not* proven. Evidence: `docs/RESULTS_DELTASEQ_LAW.md` (synthetic + ColO-RAN + arch + Twinning).

## Setup

A pooled multivariate time series is split into FL clients, then each client builds length-`L`
sliding windows within its rows (sorted by step). A window `[x_{t-L+1..t}]` predicts a next-step
label `y_{t+1}`.

- **Intact partition** (by natural entity / run, or run-level Dirichlet): whole `(run, slice)`
  groups go to one client → windows are genuine consecutive-timestep trajectories.
- **Row-level partition** (Dirichlet / random_split over rows): a group's rows scatter across
  clients → each client windows a *subset* of a run's rows → windows are non-consecutive, and the
  attached label belongs to a temporally mismatched step.

The **fragmentation audit metric** (fraction of per-client windows with contiguous `step_idx`) is
`1.0` for intact and `~0` for row-level — a *universal* property of the partition-then-window
recipe, confirmed on ColO-RAN and Twinning. The audit is necessary but **not** sufficient for an
AUC impact: a window can be fragmented yet the task unharmed. That is the whole point below.

## What fragmentation destroys (three channels)

Relative to an intact window, a fragmented one differs in:
1. **order** — the within-window temporal ordering / trajectory;
2. **content** — *which* timesteps populate the window (consecutive vs scattered samples of the run);
3. **alignment** — the window-to-label temporal correspondence (label is a mismatched later step).

## Decomposition of the AUC gap

We empirically separate the gap `= AUC(intact) − AUC(row)` into two components, isolated by the
model class:

- **Trajectory component** (channel 1, order). The value an intact sequence model extracts from the
  temporal *ordering*. Measured capacity-matched by **Δ_traj = AUC(seq LSTM) − AUC(shuffle-within-window
  LSTM)** (the shuffle destroys order while preserving the multiset / run-rate; same LSTM, so this is
  not a model-capacity gap). Only sequence-modelling architectures access it. It is the **dominant**
  component for white-noise targets (next-step BLER: Δ_traj 0.22, gap 0.15).

- **Window-content component** (channels 2–3). For targets that are themselves *window aggregates*
  (e.g. a 5-step smoothed BLER), the label depends on which steps are in the window, so fragmentation
  produces a small gap **even for an order-invariant model**. Measured by the **mean-pool MLP gap**:
  `0.000` for point targets (bler_th10), but a real, seed-consistent `0.025` for smoothed BLER
  (per-seed `0.0252 / 0.0248 / 0.0242`). It is small and absent for point targets.

So for a **point** target the gap is ~purely the trajectory component (MLP gap ≈ 0, LSTM gap 0.16 =
order); for an **aggregate** target there is an additional small order-invariant residual
(MLP 0.025 ≪ LSTM 0.082). The claim "the gap is sequence-specific" holds *for point targets* and is
*dominated* by the trajectory component otherwise — stated precisely, not overclaimed.

## Mechanism: run-rate (invariant) vs trajectory (vulnerable)

A next-step target's predictability splits into:
- a **run-rate** term — a slowly-varying per-run level (e.g. a run's characteristic throughput or
  channel quality), estimable from *any* sample of the run's steps → **order-free, partition-invariant**:
  a fragmented window's scattered steps estimate it just as well as consecutive ones; and
- a **trajectory** term — short-horizon dynamics that require *consecutive ordered* steps →
  **partition-vulnerable**.

The split is read off the target source's lag-1 autocorrelation (within `(run, slice)`):

| source | autocorr | predictability is… | Δ_traj | gap |
|---|---|---|---|---|
| tx_brate_dl | 0.984 | run-rate (persistent) | ~0 | ~0 |
| dl_buffer | 0.999 | run-rate | ~0 | ~0 |
| dl_mcs | 0.902 | run-rate | ~0 | ~0 |
| dl_cqi | 0.553 | mostly run-rate | ~0 | ~0 |
| **ul_bler** | **0.022** | **trajectory** (white-noise level; predicted via the multivariate channel-state trajectory) | **0.22** | **0.15** |

Autocorr of the source is thus a crude *pre*-diagnostic; Δ_traj is the precise one.

## Diagnostics, not a bound (the honest core)

- **Δ_seq = AUC(seq) − AUC(single-step)**, capacity-matched (same LSTM on a length-1 window). It
  conflates the trajectory value with the order-free *multi-step averaging* value (which fragmentation
  does **not** destroy), so it over-states vulnerability for persistent targets (brate: Δ_seq 0.11 but
  gap ~0). It is the *pre-registered* diagnostic and already strong (Spearman 0.92).
- **Δ_traj** removes the averaging confound (shuffle keeps the multiset) → cleaner (Spearman 0.95),
  and pulls brate onto the line (Δ_traj 0.01).

**Empirical law (validated, monotone):** the fragmentation gap is a *monotone increasing* function of
Δ_traj (ColO-RAN Pearson 0.974 / Spearman 0.945; synthetic Pearson 0.986). It is **not** a proven
bound. The conjecture `gap ≤ Δ_seq` is **falsified** by the synthetic control: with a pure-trajectory
target (no run-rate fallback), fragmented training is actively *mis-led* by temporally-incoherent
(window, label) pairs and drops *below* the single-step ceiling, so `gap > Δ_seq` (super-linear). Real
BLER, which retains a run-rate fallback, instead gives `gap < Δ_seq` (sub-linear). The relationship is
therefore monotone but not a fixed functional form.

## Conceptual limiting argument (carefully scoped)

- If **Δ_traj = 0** (the target carries no order-dependent signal beyond the order-free aggregate),
  an intact sequence model gains nothing from order, the order-free signal survives fragmentation, and
  the gap collapses to the (small) window-content residual ⇒ **Δ_traj → 0 ⟹ gap ≈ 0**.
- If **Δ_traj > 0**, the intact model exploits ordering the fragmented model cannot reconstruct ⇒
  **gap > 0**, increasing with Δ_traj.

We deliberately do **not** assert a tight bound: the synthetic super-linearity and the real-data
sub-linearity show the constant depends on the run-rate fallback and on training dynamics, which we do
not model formally. This is a *diagnostic with a mechanism*, not a theorem.

## What is NOT claimed

- No closed-form bound on the gap; the law is empirical (synthetic + 2 real datasets + 3 architectures).
- Δ_traj predicts *whether* a benchmark is at risk (large Δ_traj ⇒ row-level partitioning manufactures
  an artifact), not the exact magnitude (which also depends on data volume and the run-rate fallback).
- Architecture coverage is LSTM + GRU (both confirm) + a mean-pool MLP sanity; a Transformer underfit
  in our small-data FL regime and is excluded as uninformative, not as counter-evidence.
- The Twinning AUC-impact null is a 1-seed smoke (the mechanism audit + the Δ_seq prediction carry it;
  a 5-seed CI is straightforward future hardening). ColO-RAN gaps use 5-seed paired bootstrap (n=5 is
  the budget; CIs are correspondingly tight). Partition client counts differ by mode (iid = 7 BS;
  run/row Dirichlet = 8) — immaterial to the global-test gap but noted.

## Implication for benchmark design

1. **Partition by entity / run** (intact). To synthesise heterogeneity, use **run-level Dirichlet**
   (whole runs assigned with a skewed distribution), never row-level Dirichlet.
2. **Pre-screen with Δ_traj** (cheap, no partitioning needed): a benchmark whose target has large
   Δ_traj will have its conclusions distorted by row-level partitioning; one with Δ_traj ≈ 0 is safe.
   The pitfall is real but *task-conditional* — which is precisely why it has gone unnoticed.
