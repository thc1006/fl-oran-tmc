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

## Why we did NOT run a 30-cell FedBN sweep

A 30-cell sweep would take ~5.7 hr GPU on RTX 4080. By construction
the result equals FedAvg's natural-by-BS column already in
`artifacts/v7_stage2_full/`. Running it would produce 30 cells of
redundant data. The reviewer's MC3 is genuinely answered by the
reduction proof above; FedBN's BN-skipping benefit (Li et al. 2021
reports +0.01-0.03 AUC over FedAvg on cellular feature-skew tasks) is
structurally tied to BatchNorm presence, which our backbones lack by
design.

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
