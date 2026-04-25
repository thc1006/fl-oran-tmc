# V5 Preliminary Results — Pre-HPO

> **Snapshot before MOON hyperparameter tuning + full 5×5 sweep.**
> Generated 2026-04-25 ~08:21 CST after the 5-seed × α=0.5 checkpoint
> sweep (22 min wall-clock, 24 cells s43-s46 + 6 reused cells s42).

## Configuration

- 6 algorithms × 5 seeds × 1 alpha (α=0.5) × 20 rounds = 30 cells
- ColO-RAN unified parquet, 18.35M rows
- OOD split by `tr`: train 0-21, val 22-24, test 25-27
- Dirichlet partition over `slice_id`, n_clients=5
- Batch size 256, lr 5e-4, warmup 3 rounds, grad_clip 1.0
- bf16 AMP, torch.compile(reduce-overhead), fused Adam, cudnn deterministic
- pos_weight derived from train (not test — fixed v4 leakage)
- SCAFFOLD Option-II grad update; FedDyn update_mode=option_ii
- FedAdam bias_correction=True

## Headline table — 5-seed × α=0.5 (n=5)

| Algorithm | Test AUC | Test Acc | Test F1 |
|---|---|---|---|
| **FedAvg** | **0.7698 ± 0.0127** | 0.6590 ± 0.0115 | 0.5948 ± 0.0098 |
| FedProx (μ=0.01) | 0.7652 ± 0.0104 | 0.6560 ± 0.0060 | 0.5901 ± 0.0075 |
| FedDyn (α=0.01, option-ii) | 0.7572 ± 0.0146 | 0.6307 ± 0.0171 | 0.5810 ± 0.0106 |
| SCAFFOLD (option-ii) | 0.7444 ± 0.0170 | 0.6467 ± 0.0211 | 0.5670 ± 0.0169 |
| FedAdam (lr=0.01, bc=T) | 0.7434 ± 0.0164 | 0.6166 ± 0.0240 | 0.5676 ± 0.0142 |
| MOON (μ=1.0, τ=0.5) | 0.6824 ± 0.0151 | 0.4611 ± 0.0401 | 0.5190 ± 0.0057 |

### Statistical clusters (95% CI, n=5)

- **Top tier**: FedAvg / FedProx / FedDyn — CI overlap, statistically tied
- **Mid tier**: SCAFFOLD / FedAdam — CI tied with each other; ~1.5σ below top
- **Bottom tier**: MOON — fully separated from others; ~6σ below FedAvg
  → **MOON requires hyperparameter tuning before paper submission**

## Preliminary α curve — seed=42 only (n=1, no error bars)

| Algorithm | α=0.05 | α=0.1 | α=0.5 | α=1.0 | α=10.0 |
|---|---|---|---|---|---|
| FedAvg | 0.8132 | 0.8014 | 0.7652 | 0.7478 | 0.7493 |
| FedProx | 0.8138 | 0.7977 | 0.7624 | 0.7509 | 0.7545 |
| FedAdam | 0.7786 | 0.7897 | 0.7480 | 0.7297 | 0.7329 |
| SCAFFOLD | 0.7651 | 0.7857 | 0.7359 | 0.7195 | 0.7294 |
| **FedDyn** | **0.8159** | 0.7931 | 0.7587 | 0.7478 | 0.7557 |
| MOON | 0.7380 | 0.7335 | 0.6906 | 0.7080 | 0.6819 |

### Counterintuitive observation

All algorithms achieve **higher test AUC at low α (more concentrated
non-IID)**. Hypothesis: Dirichlet partitioning over `slice_id` (a feature
column) at low α makes each client specialise on a single slice, and the
federated average then implicitly mixes the slice distribution. At high α
each client trains on already-mixed data with less per-slice signal,
yielding a more diluted averaged model.

This contrasts with the standard NIID-Bench pattern (Dirichlet over
labels), where low α typically hurts FedAvg most. The discrepancy is
**worth a paper section** — it characterises a regime where the conventional
"more non-IID is harder" intuition does not apply.

## Why FedAvg ties top tier (paper framing)

In our setup, FedAvg is statistically indistinguishable from FedProx and
FedDyn at α=0.5. Possible reasons:

1. **Large per-client volume** (~3M rows/client at α=0.5) reduces the
   variance reduction benefits SCAFFOLD/FedDyn target.
2. **Only 5 clients** — SCAFFOLD's variance reduction theory assumes many
   clients with high heterogeneity; at n_clients=5 the regularisers
   contribute less than they cost.
3. **Slice-axis partition** (vs label-axis) — at low α each client's
   distribution is locally clean (one slice), so prox/dynamic terms
   regularise toward an average that's already near-optimal.

This is a publishable negative result: vanilla FedAvg is a strong
baseline for cellular RAN telemetry forecasting; algorithmic
sophistication does not automatically yield gains in this domain.

## Pre-HPO hyperparameters (current defaults, possibly suboptimal)

| Algorithm | Hparam | Value | Status |
|---|---|---|---|
| FedAvg | — | — | no hparam |
| FedProx | μ | 0.01 | NIID-Bench default |
| FedAdam | server_lr, β1, β2, τ | 0.01, 0.9, 0.99, 1e-3 | bias_correction=True |
| SCAFFOLD | — | — | hparam-free (Option-II) |
| FedDyn | α (FedDyn), update_mode | 0.01, option_ii | conservative; option-ii avoids Adam-scale blow-up |
| MOON | μ, τ | 1.0, 0.5 | **CIFAR defaults — likely wrong for RAN** |

**Next**: MOON HPO at α=0.5 (M5-1) → tuned (μ*, τ*) → full 5×5 sweep (M5-3).
