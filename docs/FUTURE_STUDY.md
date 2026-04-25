# Stage 1 — Future Study Issues

Items deliberately deferred from the Stage 1 short paper but flagged for
follow-up work. Each entry documents the deferral rationale and the
minimum experimental scope required to address it.

## ISSUE-1 — Cross-dataset generalization to Milan/Trentino cellular telemetry

**Status**: deferred from Stage 1 (out of scope for short paper).
**Acknowledged in**: `docs/PAPER_V6_STAGE1.md` §7 Limitations.
**Filed**: 2026-04-26 from S1-W4 cross-examination round.

### Why deferred

ColO-RAN (this Stage 1 paper) is a Colosseum-based RAN simulator with 7
gNBs, 3 slices, 4 schedulers, 28 traffic configs. HiSTM (arxiv 2508.09184,
the closest Mamba-on-cellular precedent) uses the Milan + Trentino
cellular traffic datasets — real-world telemetry from Italian operators.
The two datasets differ in:

* **Schema**: ColO-RAN per-(bs_id, slice_id, scheduler, tr) sequences with
  17 continuous + 4 categorical features (incl. ul_bler, dl_bler, mcs,
  cqi, prb_util, tx/rx pkts). Milan/Trentino uses cell-level traffic
  counts (incoming/outgoing call/sms/internet) at 10-min granularity.
* **Label**: ColO-RAN here = `ul_bler_{t+1} > 0.10` binary classification.
  Milan/Trentino is typically traffic-volume regression (next-step
  internet traffic).
* **Granularity**: ColO-RAN ~250 ms time slots within each `tr`.
  Milan/Trentino 10-min cell-level snapshots over 2 months.
* **Scale**: ColO-RAN ~18M rows after preprocessing; Milan ~6M cells × N
  time slots (different denominator).
* **Distribution shift**: ColO-RAN is simulator-generated under
  controlled `tr` schedules; Milan/Trentino is real-world with seasonal
  + weekday effects.

Re-running our LSTM / Mamba / Spiking benchmark on Milan/Trentino would
require:

1. **Dataset acquisition**: Milan/Trentino is publicly hosted but the
   variant HiSTM uses requires their preprocessing scripts.
2. **Pipeline rewrite**: new `data_v3/` package mirroring `data_v2/` but
   for the regression task, with appropriate loss (MSE/MAE not BCE),
   target builder (`add_regression_target` exists, would need adaptation),
   and evaluation (RMSE, R² instead of AUC, F1).
3. **Schema adaptation**: ColO-RAN-fitted `FeatureSchema` won't transfer.
   Categorical features (bs_id, slice_id, ...) don't exist in the
   regression dataset; continuous features are different (traffic counts
   vs RAN telemetry).
4. **Re-run the entire matrix**: 3 archs × 10 seeds × budget choices ×
   matched-25k / matched-50k / matched-100k = ~30-90 cells per setup.

Estimated effort: 1-3 weeks engineering + 5-10 hours of GPU sweep.

This is properly the scope of **a separate paper** ("Cross-dataset
benchmark of energy-aware architectures for cellular telemetry") rather
than an extension to Stage 1.

### Minimum follow-up plan when ready

* `data_v3/` package mirroring `data_v2/` (new loader, new sequences
  builder, new target builder for regression).
* New `experiments/run_v8_milan_arch_sweep.py` runner.
* Reuse `models/{forecaster_v2, mamba_forecaster, spiking_forecaster}.py`
  unchanged (architecture is dataset-agnostic given the encoder/head
  abstraction).
* Reuse `evaluation/energy_metrics.py` unchanged.
* Compare three claims to ColO-RAN findings:
  1. Does Mamba ≈ LSTM hold on real telemetry?
  2. Does Spiking-SSM applicability boundary persist?
  3. Does the Horowitz-coefficient energy ratio transfer?

### Reviewer-facing framing in Stage 1 paper

§7 limitations entry (already present): "ColO-RAN simulator: while the
dataset is widely used as a cellular benchmark, it is not real-network
telemetry. Future work on real-time RAN traces would strengthen the
claim."

A specific pointer to this issue file is added below the existing line.

---

## ISSUE-2 — Real-time RAN traces from production O-RAN deployments

(Reserved for the next round of follow-up; current scope of issue tracker.)

---

Append new entries as `## ISSUE-N — Title` with the same Status/Why
deferred/Minimum follow-up plan structure.
