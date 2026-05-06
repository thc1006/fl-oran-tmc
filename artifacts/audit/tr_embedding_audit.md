# P0-A0.2 audit: tr embedding gradient flow for unseen test tr

**Date**: 2026-05-05
**Outcome label**: `risk_confirmed`

## Evidence chain

1. **Vocab size**: `src/fl_oran/training/centralized_v3.py:41`
   ```python
   V3_CAT_SIZES = {"bs_id": 8, "slice_id": 4, "sched": 4, "tr": 29}
   ```

2. **Embedding instantiation** (identical pattern in all 3 model files):
   - `forecaster_v2.py:54`
   - `mamba_forecaster.py:183`
   - `spiking_forecaster.py:238`

   ```python
   col: nn.Embedding(schema.categorical_sizes[col] + 1, cat_embed_dim)
   ```

   For `tr`: `nn.Embedding(29 + 1, k) = nn.Embedding(30, k)` → **30 embedding rows**.

3. **Train/test tr ranges** (`data_v2/split.py:34`, `trainer_v2.py:50`,
   `centralized_v3.py:60`, plus per-cell `summary.json` confirms):
   ```python
   train_tr = list(range(22))   # {0..21}, 22 configs
   val_tr   = [22, 23, 24]      # 3 configs
   test_tr  = [25, 26, 27]      # 3 configs
   ```

4. **Gradient flow** (PyTorch `nn.Embedding` semantics): only embedding rows
   indexed during `forward()` receive gradient during `backward()`.
   Training never indexes rows 22-29 → those 8 rows remain at random
   initialisation (PyTorch default: `N(0, 1)` per `nn.Embedding.reset_parameters`).

## Implication

At test time (tr ∈ {25, 26, 27}), the model receives **random-init embedding
vectors** for the tr feature. This is a real bug.

### Bug-finding sub-question: does it explain natural-by-BS dominance?

**Hypothesis**: the random tr embedding hurts Dirichlet-partition cells more
than natural-by-BS cells, because Dirichlet-partition models rely more on
slice/tr mixture features (each client sees skewed slice mix → tr
disambiguation matters more), while natural-by-BS models can lean on bs_id
features which are properly trained.

If this hypothesis holds, **part of the C1 (natural-by-BS dominance) finding
may be a tr-embedding-bug artefact** rather than a structural property of
heterogeneity.

## Empirical verification (added 2026-05-05 post-Phase-0 code review)

**Decisive test**: load `artifacts/v7_stage2_full/v7_lstm_fedavg_iid_n7_s0/best.pt`,
construct a fresh `ForecasterV2` with `seed_everything(0, deterministic=True)`,
and compute per-row L2 delta `||trained - fresh||`. Rows that received gradient
during training will have delta > 0; rows that never received gradient will be
bit-identical to fresh init (delta = 0 modulo float32 round-off).

| Row range | Category | Mean L2 delta | Interpretation |
|---|---|---|---|
| 0-21 | TRAIN tr | **3.0e-01** | Real training drift |
| 22-24 | VAL tr | 4.0e-08 | Float32 round-off only |
| 25-27 | TEST tr | 7.5e-08 | Float32 round-off only |
| 28-29 | UNUSED (+1 padding) | 3.0e-08 | Float32 round-off only |

**6+ orders of magnitude separation** between trained and untrained rows.
The mechanism prediction holds exactly: PyTorch `nn.Embedding` only updates
indexed rows, so rows 22-29 (never indexed during train) are byte-identical
to seed-0 init values.

Methodology lesson for future P0 audits: per-row L2 norm alone is insufficient
to distinguish trained from untrained embeddings (both can fall in the
N(0,1)-magnitude band). Direct comparison with fresh seed-deterministic init
is the decisive test.

## Action

P1.2 sanity-check experiment proceeds as designed:
- Re-train 1 LSTM cell × {natural-by-BS, Dirichlet α=0.05} × {normal, frozen
  test_tr (set to mean of trained rows or zero)} × 3 seeds = 12 cells.
- Compare AUC deltas. If natural-by-BS dominance shrinks substantially with
  the tr fix, the C1 finding has tr-embedding confound and §3.5 + §7.1
  require revision.

GPU budget: ~0.6 hr on RTX 4080.

## Permanent fix recommendation (post-P1.2)

Regardless of P1.2 outcome, the embedding bug should be fixed in the artefact:
either (a) restrict `nn.Embedding(num_embeddings)` to actual `tr` count
encountered + 1 padding row, OR (b) use a hash-bucket encoder that
generalises to unseen tr values, OR (c) replace `tr` with the actual RBG
allocation numeric vector (interpretable + extrapolates to new traffic
configs).
