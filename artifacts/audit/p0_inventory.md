# P0-A0.3 audit: test fixtures, checkpoints, RNG seeds

**Date**: 2026-05-05
**Outcome label**: `risk_cleared`

## Cell directories

| Sweep | Path | Cell count | Per-cell artefacts |
|---|---|---|---|
| Phase 5 (Stage 2 full) | `artifacts/v7_stage2_full/` | 900 | `best.pt`, `history.csv`, `summary.json` |
| Phase 6 (per-BS Dirichlet) | `artifacts/v7_phase6_per_bs_dirichlet/` | 60 | same |
| §7.1.1 random_split V100 | `artifacts/v7_ablation_random_split/` | 15 | same |
| Misc smokes | `artifacts/v7_phase{2,3,3a,3e,_sweep}/` | varies | same |

## summary.json schema (per cell)

```yaml
config:
  name, arch, arch_kwargs, algorithm, algo_kwargs,
  partition_mode (iid|dirichlet|random_split|per_bs_dirichlet),
  alpha, n_clients, num_rounds (=100), clients_per_round (=5),
  max_steps_per_round (=50), batch_size (=64),
  lr, lr_warmup_rounds, grad_clip,
  unified_parquet, sample_ratio, threshold (=0.1),
  seq_len (=5), train_tr (=range(22)), val_tr, test_tr,
  seed
metrics:
  best_val_auc, best_val_round, test_auc, test_f1, ...
energy:
  training_model_attributable_mJ, idle_baseline_mW, ...
```

## Aggregated artefacts

- `artifacts/v7_stage2_full/aggregated_phase5.json` — 90 group means + 270
  paired-bootstrap distributions (the canonical paper-results JSON).
- `artifacts/figures/{algo_ranking,interaction_heatmap,pareto}.{png,pdf,svg}`
  — paper figures (post-S11 vector format).
- `artifacts/figures/results_table.csv` — per-cell aggregated table.

## RNG seeds

- Per-cell training seed: `summary.json::config::seed` ∈ {0, 1, 2, ..., 9, 42}
  (10 seeds per group; some sweeps include seed 42 as 11th for backup).
- Bootstrap base seeds: 2026 (algorithm-pair sweep, 180 pairs), 2027
  (architecture-pair sweep, 90 pairs).
- Per-pair offset: BLAKE2b hash of pair identifier added to base seed
  (decorrelates 270 paired-bootstrap streams; `scripts/aggregate_v7_results.py`).

## Test data loader

- Parquet: `/home/thc1006/dev/colosseum-oran-federated-slicing/data/coloran_raw_unified.parquet`
  (~18M rows; symlink target).
- Loader: `src/fl_oran/data_v2/sequences.py::build_run_sequences()`.
- OOD split: `src/fl_oran/data_v2/split.py::ood_split_by_tr()` — train tr
  ∈ {0..21}, val tr ∈ {22..24}, test tr ∈ {25..27}.

## Phase 1 fixture readiness

| P1 task | Required fixture | Available? |
|---|---|---|
| P1.1 (naive baselines) | test parquet + ul_bler column | ✅ |
| P1.2 (tr embedding) | best.pt for at least 1 LSTM × IID × seed | ✅ (e.g. `v7_lstm_fedavg_iid_n7_s0/best.pt`) |
| P1.3 (FedBN) | new spec YAML + new algo class | ❌ to create |
| P1.4 (language) | PAPER_DRAFT.md + main.tex | ✅ |

No fixture-blocking issues. P1 can proceed once P0-A0.1 + P0-A0.2 outcomes
are folded into P1.1 + P1.2 designs (both `risk_confirmed`, no design
changes needed).
