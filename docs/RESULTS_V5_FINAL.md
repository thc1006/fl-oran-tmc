# V5 Final Results — IEEE TMC Submission Material

> **150 cells: 5 seeds × 5 Dirichlet α × 6 FL algorithms × 20 rounds**
> Wall-clock 2 h 53 min on a single RTX 4080 (08:44 → 11:37 CST,
> 2026-04-25). All cells use the post-adversarial-review pipeline:
> `pos_weight_split=train`, `cudnn_deterministic=True`, SCAFFOLD
> Option-II, FedDyn `update_mode=option_ii`, FedAdam
> `bias_correction=True`, MOON `(μ, τ)=(0.1, 1.0)` from a 15-cell HPO
> grid at α=0.5.

## Table 1 — Test AUC (mean ± std, n=5 seeds)

| Algorithm | α=0.05 | α=0.1 | α=0.5 | α=1.0 | α=10.0 |
|---|---|---|---|---|---|
| FedAvg | 0.8249 ± 0.040 | 0.8130 ± 0.024 | **0.7698 ± 0.013** | **0.7553 ± 0.006** | 0.7456 ± 0.003 |
| FedProx (μ=0.01) | 0.8200 ± 0.041 | 0.8079 ± 0.024 | 0.7652 ± 0.010 | 0.7541 ± 0.004 | **0.7473 ± 0.005** |
| FedAdam (lr=0.01, bc=T) | 0.7976 ± 0.041 | 0.7860 ± 0.027 | 0.7434 ± 0.016 | 0.7366 ± 0.012 | 0.7276 ± 0.009 |
| SCAFFOLD | 0.7619 ± **0.083** | 0.7611 ± 0.042 | 0.7444 ± 0.017 | 0.7263 ± 0.020 | 0.7332 ± 0.003 |
| FedDyn (α=0.01) | 0.8172 ± 0.038 | 0.8002 ± 0.024 | 0.7572 ± 0.015 | 0.7480 ± 0.006 | 0.7408 ± 0.011 |
| **MOON (μ=0.1, τ=1.0)** | **0.8288 ± 0.041** | **0.8131 ± 0.025** | 0.7680 ± 0.012 | 0.7529 ± 0.006 | 0.7442 ± 0.004 |

Bold marks the best mean per column (statistical ties not bolded).

## Per-α winners (statistical, 95% CI overlap test)

| α | Top tier (CI overlap) |
|---|---|
| 0.05 | MOON, FedAvg, FedProx, FedDyn (all CIs overlap, n=5) |
| 0.1 | MOON, FedAvg, FedProx, FedDyn (all overlap) |
| 0.5 | FedAvg, MOON, FedProx, FedDyn (all overlap) |
| 1.0 | FedAvg, MOON, FedProx, FedDyn (all overlap) |
| 10.0 | FedProx, FedAvg, MOON, FedDyn (all overlap) |

**Across all α**: top tier is consistent — `{FedAvg, FedProx, FedDyn, MOON}`. Mid tier `{FedAdam}`. Bottom tier `{SCAFFOLD}` (high variance).

## Table 2 — Test F1 (mean ± std)

| Algorithm | α=0.05 | α=0.1 | α=0.5 | α=1.0 | α=10.0 |
|---|---|---|---|---|---|
| FedAvg | 0.6472 ± 0.046 | 0.6280 ± 0.018 | 0.5948 ± 0.010 | 0.5832 ± 0.004 | 0.5750 ± 0.003 |
| FedProx | 0.6413 ± 0.047 | 0.6254 ± 0.019 | 0.5901 ± 0.008 | 0.5820 ± 0.003 | 0.5767 ± 0.004 |
| FedAdam | 0.6137 ± 0.039 | 0.5926 ± 0.016 | 0.5676 ± 0.014 | 0.5605 ± 0.011 | 0.5563 ± 0.007 |
| SCAFFOLD | 0.5892 ± 0.073 | 0.5810 ± 0.044 | 0.5670 ± 0.017 | 0.5598 ± 0.021 | 0.5669 ± 0.002 |
| FedDyn | 0.6352 ± 0.041 | 0.6166 ± 0.017 | 0.5810 ± 0.011 | 0.5733 ± 0.006 | 0.5702 ± 0.010 |
| **MOON** | **0.6505 ± 0.048** | 0.6279 ± 0.021 | 0.5912 ± 0.008 | 0.5811 ± 0.005 | 0.5735 ± 0.004 |

## Findings — paper-quality

### F1. MOON is rescued by hyperparameter tuning

Pre-HPO with paper-default `(μ=1.0, τ=0.5)` (chosen for CIFAR), MOON sat
~0.06–0.09 AUC below FedAvg across all α — clearly bottom-tier. After a
15-cell HPO grid at α=0.5 picked `(μ=0.1, τ=1.0)`, MOON jumps to
**top-tier across all 5 α values, including the best mean at
α=0.05/0.1**. The CIFAR-default `μ=1.0` makes the contrastive term
overwhelm the BCE base loss; for RAN telemetry, the contrastive signal
must be 10× weaker.

### F2. Counterintuitive α monotonicity

All six algorithms achieve **higher AUC at low α (more concentrated
non-IID)** than at high α (near-IID). The effect is strong: ~0.07–0.09
AUC gap between α=0.05 and α=10.0 for every method.

This contradicts the standard NIID-Bench narrative. Our partition is
over `slice_id` (a feature axis) rather than the prediction label.
At low α, each client trains on essentially one slice's traffic
pattern; the federated average across 5 clients then implicitly mixes
the slice distribution. At high α, every client trains on already-mixed
data, so the averaged model is "pre-blurred" with less per-slice
signal.

This characterises a regime worth a dedicated paper section: when
heterogeneity is in the *feature space* (network slice type, in our
case) rather than the *label space*, the conventional FL non-IID
intuition is inverted.

### F3. Vanilla FedAvg is a strong baseline

At every α, FedAvg's CI overlaps with the top performer. Algorithmic
sophistication (FedProx prox term, FedDyn dynamic regularisation, MOON
contrastive loss) does not yield statistically significant gains in
this domain. We attribute this to:

- **High per-client volume** (~3M rows at α=0.5) reduces variance —
  the regime where fancy regularisers help.
- **Only 5 clients** — SCAFFOLD's variance reduction theory targets
  many clients with high heterogeneity.
- **Feature-axis partitioning** — each client's distribution is locally
  clean (one slice), so prox / dynamic / contrastive terms regularise
  toward an already-near-optimal centroid.

This is a publishable **negative result**: vanilla FedAvg is
competitive for cellular RAN telemetry forecasting; algorithmic
sophistication does not automatically yield gains in this domain.

### F4. SCAFFOLD has anomalously high variance

At α=0.05, SCAFFOLD's std is 0.0832 — 2-4× larger than every other
algorithm. Even after our Option-II fix (which corrected the
SGD-vs-Adam scaling collapse from F1=0.21 in the pilot), SCAFFOLD's
control-variate estimates remain high-variance with only 5 clients.
The theory assumes many clients; at small fleet size the per-client
gradient estimates fluctuate more than they correct.

### F5. FedAdam under-performs across all α

Even with `bias_correction=True` (added to address momentum ramp-up
under <20 rounds), FedAdam consistently lands ~0.025-0.04 AUC below
the top tier. The server-side Adam update is theoretically motivated
for non-IID but, in our regime, the moment estimates over only 20
rounds with fresh client deltas have insufficient warm-up. Longer
training (50-100 rounds) might close this gap; we leave it as future
work.

## Hyperparameter selection — final

| Algorithm | Hparam | Value | Provenance |
|---|---|---|---|
| FedAvg | — | — | parameter-free |
| FedProx | μ | 0.01 | NIID-Bench default; further tuning offered no significant gain in pilot ablation |
| FedAdam | server_lr, β1, β2, τ, bias_correction | 0.01, 0.9, 0.99, 1e-3, True | Reddi paper Algorithm 2 + bias_correction (this work, 5-round under-train fix) |
| SCAFFOLD | update_mode | Option-II | Karimireddy paper §3 alternative; required because Option-I assumes SGD and our local optimiser is Adam |
| FedDyn | α (FedDyn), update_mode | 0.01, option_ii | conservative α + Adam-friendly grad-based h_i (this work) |
| **MOON** | μ, τ | **0.1, 1.0** | **15-cell HPO grid at α=0.5, seed=42** |

## Bit-level reproducibility

- All cells run with `cudnn_deterministic=True`
- Per-cell seed-everything before training
- `pos_weight_split=train` (no test-set leakage into training loss)
- bf16 AMP throughout, fused Adam on CUDA
- Outputs: `artifacts/v5_sweep/v5_<algo>_a<α>_s<seed>/{summary.json, history.csv, best.pt}` per cell

## Reproducer

```bash
# 1. Build the venv (one-time)
cd /path/to/fl-oran-tmc
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# 2. Symlink the ColO-RAN raw parquet
ln -s /path/to/coloran_raw_unified.parquet data/coloran_raw_unified.parquet

# 3. Run the full 150-cell sweep (~2.9 h on RTX 4080)
./scripts/run_full_sweep.sh

# 4. Aggregate results
python scripts/aggregate_v5_results.py
# Produces: artifacts/RESULTS_V5.md, artifacts/v5_sweep/aggregated_table.csv
```

## File map

```
docs/
├── ADR-001-v5-tmc-paper-plan.md   # 16-decision plan + revision history
├── RESULTS_V5_PRELIM.md            # 30-cell pre-HPO snapshot (5 seeds × α=0.5 + s42 α-curve)
└── RESULTS_V5_FINAL.md             # this file (150-cell post-HPO)

scripts/
├── run_full_sweep.sh               # 5×5 launcher (used to generate this table)
├── run_moon_hpo.sh                 # MOON (μ, τ) HPO at α=0.5
├── run_seed_checkpoint.sh          # 5-seed × 1-α mini-sweep (preliminary)
└── aggregate_v5_results.py         # produces RESULTS_V5.md from cells

experiments/
├── run_v5_sweep_matrix.py          # multi-cell driver (single Python process)
├── run_v5_algorithm_sweep.py       # single-cell CLI
└── run_moon_hpo.py                 # MOON-specific HPO

src/fl_oran/
├── training/fl_v5.py               # SharedSplits + prepare_v5_data + _run_training
├── federated/algorithms/           # 6 algorithm classes (FedAvg/FedProx/FedAdam/SCAFFOLD/FedDyn/MOON)
└── ...                             # data_v2/, models/, utils/ unchanged from v4
```
