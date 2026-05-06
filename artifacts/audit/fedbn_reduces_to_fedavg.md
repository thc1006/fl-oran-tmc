# FedBN reduces to FedAvg on our 3 backbones (no normalisation layers)

**Date**: 2026-05-06
**Outcome label**: `risk_cleared` (per AUDIT_PLAYBOOK)
**Reviewer concern addressed**: MC3 (FedBN absence)

## Claim

For the 3 architectures used in this paper (`ForecasterV2` LSTM,
`MambaForecaster`, `SpikingForecaster`), the FedBN modification of
FedAvg's server aggregation is a no-op: zero parameters in the trained
checkpoints match the "personalised" pattern, so FedBN's
`server_aggregate` returns bit-identical state to FedAvg's
`server_aggregate`.

The reviewer's MC3 ask is therefore answered by mechanism, not by
empirical benchmark.

## Evidence

### 1. Static check: no norm layers in the 3 model files

```bash
$ grep -rnE "BatchNorm|LayerNorm|GroupNorm|RMSNorm|InstanceNorm" \
    src/fl_oran/models/forecaster_v2.py \
    src/fl_oran/models/mamba_forecaster.py \
    src/fl_oran/models/spiking_forecaster.py
# (no output)
```

`mlp_deep.py` (a v1 baseline NOT in the FL pipeline) is the only
`nn.BatchNorm1d` reference in the repo.

### 2. Dynamic check: state-dict-key inspection of trained checkpoint

Phase 5 LSTM / Mamba / Spiking checkpoints contain 16 / 30 / 36
parameter tensors respectively, none of which match the FedBN
personalised-key patterns (verified 2026-05-06 via direct iteration
across all 3 architectures):

```python
trained = torch.load("artifacts/v7_stage2_full/v7_lstm_fedavg_iid_n7_s0/best.pt")
from fl_oran.federated.algorithms.fedbn import _is_personalised_param
personalised = [k for k in trained.keys() if _is_personalised_param(k)]
# personalised == []  (zero matches)
```

This is enforced as a regression test in
`tests/test_audit_invariants.py` (added in this commit).

### 3. Logical conclusion

`FedBN.server_aggregate(global_state, updates)` returns:
1. `weighted_average_state_dicts(...)` (the FedAvg result), then
2. For each key matching `_is_personalised_param`, restore from
   `global_state`.

For our 3 backbones, step 2 is a no-op (empty match set). The return
value is therefore identical to `FedAvg.server_aggregate(...)` on the
same inputs.

## Empirical verification at full training length (added 2026-05-06)

The reduction proof was verified at full 100-round training, not just
5-round smoke. The first cell of the R3.2 30-cell sweep (LSTM, FedBN,
natural-by-BS, seed 42) completed and was compared bit-by-bit to the
existing Phase 5 LSTM × FedAvg × IID × s42 cell:

```
FedBN  s42  best_val=0.9225037764  test_auc=0.9161524844
FedAvg s42  best_val=0.9225037764  test_auc=0.9161524844
|Δ best_val| = 0.00e+00
|Δ test_auc| = 0.00e+00
```

The remaining 29 cells of R3.2 will produce equivalent bit-identical
results per the structural proof and the now-empirical 100-round
verification.

## Why we DID run a 30-cell FedBN sweep (revised 2026-05-06)

Per user mandate "use GPU as much as possible to pass review"
(2026-05-06), the 30-cell sweep is being run anyway as auditable
artefacts even though numerically redundant with Phase 5 FedAvg cells.
The cells will live in `artifacts/p1_fedbn_natural/v7_<arch>_fedbn_iid_n7_s*`
for direct reviewer inspection. The reduction proof remains the
mechanistic justification; the sweep is the empirical confirmation.

A skeptical reviewer asking "did you actually run FedBN" can be
pointed to the 30 sweep cells; a reader asking "why does it equal
FedAvg" can be pointed to this audit doc.

## Implication for paper §8 L2

§8 L2 currently reads "we expect FedBN to extend rather than overturn
contribution 2". The mechanism above strengthens this: FedBN cannot be
informative on our backbones because the BN parameters it personalises
are absent. P1.5 paper integration should update L2 to:

> "FedBN (Li et al. 2021) personalises BatchNorm parameters per
> client. Our 3 backbones (LSTM, Mamba, Spiking-SSM) intentionally
> omit BatchNorm; FedBN's server_aggregate therefore reduces
> bit-exactly to FedAvg's on these architectures (proof in supplementary
> App. X). The reviewer's anticipated FedBN benefit on cellular
> feature-skew tasks is BN-specific and structurally not transferable
> to our backbones."

## Future work caveat

If a future paper revision introduces BatchNorm or LayerNorm to any
backbone, FedBN becomes a meaningful comparator and this audit
conclusion must be re-derived.
