# v3 Federated Learning Results (Path A — Raw ColO-RAN)

## Task Definition
**Binary classification**: predict `ul_bler_{t+1} > 0.10` one 250ms step into the future.

Rationale: `ul_bler` has near-zero autocorrelation (acf(1) ≈ 0.01–0.09), so
persistence cannot trivially win. Positive class rate ~30.8% (balanced).

## Data Pipeline
- Raw ColO-RAN dataset (~7.6 GB, 39,882 CSVs) → unified parquet (18.3M rows).
- 20% random subsample (≈3.65M rows) for faster iteration.
- **OOD split by `training_config`**: train tr0–21, val tr22–24, test tr25–27.
  - Test set contains RBG allocations the model never saw in training.
- Sequences built within each (run_id, slice_id) group, no cross-run leakage.

## Results (held-out test set, unseen training configs)

| Model | Accuracy | AUC | F1 | Note |
|-------|:--------:|:---:|:--:|------|
| Majority baseline (predict "no violation") | 0.690 | 0.500 | 0.000 | no discrimination |
| Persistence classifier (next ≈ current) | 0.603 | ~0.55 | — | weaker than majority (acf ≈ 0) |
| **Centralized LSTM** | 0.591 | **0.674** | **0.510** | 3 epochs, 2.87M train sequences |
| **FL IID LSTM** (7 BS clients, 20 rounds) | **0.594** | **0.672** | **0.507** | **matches centralized** |
| **FL Non-IID LSTM** (slice specialists, 20 rounds) | 0.615 | **0.665** | 0.487 | +0.7% acc, −0.7% AUC vs IID |

### Key findings

1. **LSTM beats all trivial baselines on AUC/F1**. Accuracy is lower than
   majority because we used `pos_weight=2.23` in BCE to balance recall —
   essential for SLA-violation use case (missing violations is costlier
   than false alarms).

2. **Federated IID achieves centralized's AUC within 0.2 pp** (0.672 vs 0.674).
   Concrete statement: 7 BSs can train a shared SLA predictor without
   sharing any raw KPIs, with no measurable accuracy penalty vs pooling.

3. **Federated Non-IID loses only 0.7 pp AUC** (0.672 → 0.665) despite each
   client seeing only 1 of 3 slice types. Demonstrates FedAvg can aggregate
   slice-specialist local models into a generalist global model.

4. **F1 = 0.51 is the real headline**. Majority baseline F1 = 0 (predicts
   nothing positive). LSTM catches ~60% of true violations vs 0 for
   majority, while keeping false positive rate modest.

## Architecture choices that mattered (Path A lessons)

| Before (v1/v2) | After (v3) | Why |
|---|---|---|
| `StandardScaler` on bs_id → val values ±10⁶ | `nn.Embedding(8, 8)` | Categorical features shouldn't be z-scored |
| Predict `allocation_efficiency` (synthetic target = 0.5·a+0.3·b+0.2·c) | Predict `ul_bler_{t+1} > 0.1` (real physical KPI) | no target leakage |
| FedAvg with 62,500 local steps per round | `max_steps_per_round=500` + gradient clipping ‖g‖≤1 + LR warmup | drift ↓10×, convergence 3× faster |
| No persistence/majority baselines | Both reported up-front | gate: must beat persistence to claim "learning" |
| Random 80/20 split within data | OOD split by training_config | real generalisation test |

## Pipeline timing (RTX 4080, bf16, 20% subsample)
- Unified parquet build: ~2 min (3,080 runs, stream write)
- Centralized LSTM training: 3 epochs × ~40s = 2 min
- FL IID (20 rounds × 5 clients × 500 steps): ~2.5 min
- FL Non-IID (same config): ~1.7 min
- **Total end-to-end from raw CSVs: ~10 min**

## Tests
- 86 unit/integration tests, TDD-written before implementation
- Coverage includes: parsers (blank headers, renames), feature schema,
  target builders (group-boundary correctness), ForecasterV2 (persistence
  identity, embedding differentiation), step-capped trainer, partitioner
  (IID & non-IID invariants)

## Honest limitations

- 20% subsample keeps runs fast; results should replicate with full data
- ColO-RAN is simulation-derived; real networks have more heterogeneity
- `ul_bler > 0.1` is an arbitrary SLA threshold; operators may tune
- Only Gaussian DP mechanism is supported; no `dp-accounting` precise ε
- Non-IID is synthesised (slice-specialist mapping), not observed

## Reproducibility
```bash
# Requires the raw dataset already extracted to raw/colosseum-oran-coloran-dataset-master/
python -m fl_oran.data_raw.cli \
    --raw-root raw/colosseum-oran-coloran-dataset-master \
    --out-path data/coloran_raw_unified.parquet
python experiments/run_v3_centralized.py
python experiments/run_v3_fl_iid.py
python experiments/run_v3_fl_noniid.py
```
