# P0-A0.1 audit: ul_bler in 17 continuous features

**Date**: 2026-05-05
**Outcome label**: `risk_confirmed`

## Evidence chain

`src/fl_oran/data_v2/features.py`:
- L31: continuous feature canonical list includes `"dl_bler", "ul_bler"` (among 17 names)
- L60: feature-projection helper also lists `"dl_bler", "ul_bler"` in the continuous block
- L89-91: target derivation:
  ```python
  next_ul_bler = df.groupby(key, observed=True)["ul_bler"].shift(-1) \
      if "ul_bler" in df.columns else pd.Series(0.0, index=df.index)
  df["y_sla_violation_next"] = (next_ul_bler > SLA_BLER_THRESHOLD).astype("float32")
  ```

`src/fl_oran/data_raw/merge.py`:
- L114-118: `ul_bler` is derived from `rx_errors_ul_pct` upstream column.

## Implication

`ul_bler[t]` is in the input feature vector; target is `1[ul_bler[t+1] > 0.10]`.
This is the canonical autoregressive forecasting setup.

**Reviewer's MC2 / MR-Minor concern about naive baselines is valid**: a
last-BLER persistence baseline `predict_t+1 = 1[ul_bler[t] > 0.10]` is a
non-trivial baseline that the FL methods must beat by a meaningful margin to
justify the modelling complexity.

## Action

P1.1 (last-BLER persistence + logistic-regression baselines) proceeds as
designed. The baseline computation reads the existing test parquet via
`build_run_sequences` + extracts the `ul_bler` column at offset `seq_len-1`
(the last seen time step) for the persistence prediction.

No P1 redesign required.
