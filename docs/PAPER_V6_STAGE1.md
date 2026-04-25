# Energy-Aware Architectures for O-RAN Slice SLA Forecasting

## (Stage 1 short paper draft — placeholders to be replaced after S1-W2 sweep + S1-W3 aggregation)

**Target venue (priority)**: IEEE IoTJ → IEEE TNSM → IEEE Globecom 2026 (deadline ~July) → IEEE WCNC 2027 short.

**Authors**: thc1006 et al.

---

## Abstract (placeholder, ~200 words)

The Open Radio Access Network (O-RAN) deploys always-on xApp inference loops
that forecast slice service-level agreement (SLA) violations from cellular
telemetry. We benchmark three temporal backbones — long short-term memory
(LSTM), Mamba-S6 selective state-space model (SSM), and a spiking SSM with
leaky integrate-and-fire (LIF) output neurons — on the public ColO-RAN
dataset under identical encoder, classifier head, parameter budget, OOD
split, optimisation budget, and bf16 precision.

We measure both accuracy (test AUC, F1) and a Horowitz-2014 45 nm CMOS
energy estimate (4.6 pJ per dense MAC, 0.9 pJ per spike accumulate).

**[RESULTS_PLACEHOLDER — fill from `docs/RESULTS_V6_STAGE1.md` after S1-W3]**

We find that **[ARCH_SUMMARY]**, with paired-bootstrap CI95 of
**[CI_SPIKING_VS_LSTM]** for Spiking vs LSTM and **[CI_MAMBA_VS_LSTM]**
for Mamba vs LSTM (n=10 seeds). The Spiking architecture's energy ratio
to LSTM is **[RATIO_SPIKING_VS_LSTM]**, suggesting **[INTERPRETATION]**.

**Reproducibility**: full source under Apache-2.0 at
[github.com/thc1006/fl-oran-tmc](https://github.com/thc1006/fl-oran-tmc),
exact commit `[COMMIT_HASH]`, hardware: single RTX 4080.

---

## 1. Introduction

Cellular RAN xApps run forecasting models continuously to anticipate SLA
violations and trigger pre-emptive resource reallocation. Modern xApp
deployments observe several thousand cells per Open-RAN deployment, so
even a single-digit reduction in inference energy compounds rapidly.

While prior work has focused on **algorithmic** non-IID adaptation in
federated learning (e.g. FL-DRAM[ref] and SliceFed[ref] in 2026, FedBN
on cellular[ref] in 2025), the **architecture** of the per-client model
has been treated as fixed at LSTM. We investigate the orthogonal
question: **does the choice of temporal backbone matter for SLA
forecasting on cellular telemetry, both in accuracy and in energy?**

Our contribution is the first systematic Spiking-SSM benchmark on the
public ColO-RAN dataset, with a parameter-matched comparison against
LSTM and Mamba-S6 baselines and a transparent Horowitz-coefficient
energy estimate. The Spiking-SSM × ColO-RAN combination has zero hits
in arxiv / IEEE Xplore / NeurIPS proceedings searches as of April 2026.

### 1.1 Related work

* **HiSTM** (arxiv 2508.09184): Hierarchical Spatiotemporal Mamba for
  Milan/Trentino cellular *traffic* prediction (centralized, regression,
  no spiking).
* **SpikySpace** (arxiv 2601.02411, Jan 2026): Spiking SSM for general
  time-series forecasting; not RAN, not slice SLA.
* **SpikingMamba** (TMLR Jan 2026, arxiv 2510.04595): LLM via knowledge
  distillation; reports a 4.78% accuracy gap vs Mamba — we set our
  accuracy threshold at 3% gap to be more aggressive than literature SOTA.
* **arxiv 2508.08479** (2025-08): FedAvg/FedProx/FedBN with LSTM /
  CNN / Transformer on 5G live-streaming throughput — finds FedBN-LSTM
  beats FedAvg by 11.7%. No spiking.
* **FL-DRAM** (Springer 2026-03): hierarchical FL with PerFedRL slice-aware
  adaptation. No architectural variation.
* **SliceFed** (arxiv 2603.11390, 2026-03): federated MARL on gNB agents
  for 6G spectrum slicing. No architectural variation.

We complement rather than compete with these works: the architectural
choice is orthogonal to the FL-algorithm choice, so our findings can be
combined with any of the above.

---

## 2. Background

### 2.1 ColO-RAN dataset and slice SLA forecasting

**[Brief description of the 7 base stations × 28 traffic configurations,
slice / scheduler axes, the binary `ul_bler_{t+1} > 0.10` task, and
the previously documented `allocation_efficiency = 0.5 * throughput_eff
+ 0.3 * qos + 0.2 * prb_util` target leakage we eliminate (v4 audit).]**

### 2.2 State-space models in brief

**[Two paragraphs: structured SSMs (S4/S5/S6), continuous-time recurrence,
discretisation under zero-order hold; selective-SSM input dependence
(Gu and Dao 2024 §3.5); pure-PyTorch sequential-scan implementation
without the Triton kernel.]**

### 2.3 Spiking neurons and surrogate gradients

**[Two paragraphs: LIF dynamics, threshold-and-reset; atan surrogate
(Wei et al. 2018, snntorch); difficulty of training surrogate-gradient
models with Adam at large LR — motivates our preregistered lr=1e-4 and
1250-step warmup.]**

---

## 3. Task definition

### 3.1 Target leakage audit (reused from v4)

**[Re-export of the v4 leakage audit table demonstrating that
`allocation_efficiency` is a deterministic linear combination of three
other features, hence unsuitable as a regression target. Cite v4 paper /
ColO-RAN documentation.]**

### 3.2 Out-of-distribution split

The ColO-RAN traffic-config index `tr` ranges over 0..27. We use a
canonical OOD split: train on tr 0-21, validate on tr 22-24, test on
tr 25-27. This split is identical to v4 and v5 to ensure cross-paper
comparability of accuracy numbers.

### 3.3 Preregistered hyperparameters (no in-paper HPO)

| Hyperparameter | LSTM | Mamba | Spiking |
|---|---|---|---|
| Optimiser | Adam | Adam | Adam |
| Learning rate | 5e-4 | 5e-4 | 1e-4 |
| LR warmup steps (linear) | 750 | 750 | 1250 |
| Weight decay | 0.0 | 0.0 | 0.0 |
| Dropout | 0.1 | 0.1 | 0.0 |
| Total gradient steps | 5000 | 5000 | 5000 |
| Batch size | 64 | 64 | 64 |
| Mixed precision | bf16 | bf16 | bf16 |
| `cudnn_deterministic` | True | True | True |
| Sequence length | 5 | 5 | 5 |
| Hidden width | 64 → 32 | d_model=64, expand=1, 2 blocks | d_model=80, 2 blocks |
| Backbone state dim | (LSTM internal) | d_state=16 | d_state=16, T_inner=1 |
| Trainable parameters | **[N_LSTM]** | **[N_MAMBA]** | **[N_SPIKING]** |

Parameter counts within ±10% of LSTM baseline by design (per ADR §D-20
and verified in `tests/test_v6_param_count.py`). The Spiking learning
rate, dropout=0, and 0 weight-decay choices are preregistered based on
spiking-SSM literature (SpikingSSMs AAAI 2025; Yin et al. ICCV 2023);
no HPO inside Stage 1.

---

## 4. Three-architecture methodology

All three classes share an identical encoder (`nn.Embedding` per
categorical feature followed by concatenation with continuous features
that have been standardised by a leak-free `ContinuousScaler` fitted on
the train split only) and an identical classifier head
(`Linear(32 → 64) → ReLU → Linear(64 → 1)`). Only the temporal
backbone differs.

### 4.1 LSTM baseline (`ForecasterV2`)

Two-layer stacked nn.LSTM (input_dim → 64 → 32). Unchanged since v3.

### 4.2 Mamba-S6 (`MambaForecaster`)

Pure-PyTorch implementation of the selective state-space block from
Gu and Dao 2024 §3.5: `Linear(input_dim → 64)` → 2× `MambaS6Block`
(`d_model=64, d_state=16, d_conv=4, expand=1`) → `Linear(64 → 32)`.
Each `MambaS6Block` performs (a) input-dependent projection of `dt`,
`B`, `C`, (b) a depthwise causal 1-D conv on the gating branch with
SiLU, (c) a sequential discretised SSM scan, (d) gating by SiLU(z)
and final down-projection. We deliberately implement the scan in
pure PyTorch to remove the `mamba-ssm` Triton-kernel dependency: our
target reproducibility artifact requires only PyTorch + CUDA runtime,
not a system CUDA-dev toolkit.

### 4.3 Spiking SSM (`SpikingForecaster`)

`Linear(input_dim → 80)` → 2× `SpikingSSMBlock`
(`d_model=80, d_state=16, lif_threshold=1.0, lif_beta=0.9, atan_alpha=2.0`)
→ `Linear(80 → 32)`. Each `SpikingSSMBlock` performs (a) `in_proj`
linear projection, (b) diagonal SSM recurrence with learnable B, C,
D and log-parameterised A initialised at `-[1, 2, ..., d_state]`
per channel, (c) a per-channel `snntorch.Leaky` LIF neuron emitting
binary spikes via the atan surrogate gradient, (d) `out_proj` linear
consuming the binary spike train. The classifier head receives the
time-major last-step activation as in the LSTM and Mamba paths.

`d_model=80` (vs 64 for the other two) is preregistered to satisfy the
±10% parameter-count parity constraint.

---

## 5. Energy estimation protocol

We do not deploy on neuromorphic hardware. The energy numbers reported
here are **theoretical** under the Horowitz 2014 45 nm CMOS coefficients
(`pJ_per_MAC_FP32 = 4.6`, `pJ_per_AC_FP32 = 0.9`). They serve as an
**upper bound** on the energy a deployed system might achieve.

For each architecture we measure, **per single inference**:

* `flops` = MAC count over the entire forward pass, computed via
  `fvcore.nn.FlopCountAnalysis` plus a hand-counted contribution for
  `nn.LSTM` modules (which fvcore does not trace into the C++ kernel
  for). Without the LSTM correction, the LSTM energy estimate is ~80×
  too low and the energy comparison becomes meaningless.
* `sops` = synaptic operations = `Σ_block (spike_count_block ×
  out_proj.out_features)`, summed over `SpikingSSMBlock` instances
  only. Zero for LSTM and Mamba models.
* `total_energy_pJ = flops × 4.6 + sops × 0.9`.

Limitation: the fvcore FLOPs term double-counts `out_proj` operations
in `SpikingForecaster`, since their input is binary and the operation
is truly accumulate-only. The reported `total_energy_pJ` is therefore
an **upper bound** on Spiking energy. We additionally report
`backbone_only_energy_ratio` so reviewers can audit the dense vs
spike contribution split.

---

## 6. Results

**[Section to be filled from `docs/RESULTS_V6_STAGE1.md` after S1-W3.]**

### 6.1 Accuracy

**[Table: 3 archs × n_seeds with mean ± std AUC, F1, accuracy.]**

### 6.2 Pairwise paired-bootstrap CI95

**[Table: delta_auc(Spiking, LSTM), delta_auc(Mamba, LSTM),
delta_auc(Spiking, Mamba) with CI95 and Wilcoxon p as supplementary.]**

### 6.3 Energy

**[Table: per-arch FLOPs, SOPs, total energy_pJ, energy_ratio
relative to LSTM. Pareto plot accuracy vs energy_ratio.]**

### 6.4 Per-layer spike rate (SpikingForecaster)

**[Heatmap: average spike rate per channel per timestep.]**

---

## 7. Limitations

* **No neuromorphic hardware**: energy is theoretical upper bound, not
  measured. Deployed Spiking-SSM energy on Loihi-2 / Truenorth would
  realise the savings only after additional engineering.
* **Centralized only**: this short paper does not extend to a federated
  setting. The Stage 2 follow-up (conditional on the GO/NO-GO outcome
  recorded in our ADR) integrates the chosen primary backbone with
  the existing 7-algorithm FL registry on the same dataset.
* **`T_inner=1`** (one LIF integration per sequence position) trades
  off accuracy for compute. Recovery HPO with `T_inner=5` is part of
  S1-W3 if the C1 accuracy criterion is not met at `T_inner=1`.
* **ColO-RAN simulator**: while the dataset is widely used as a
  cellular benchmark, it is not real-network telemetry. Future work
  on Colosseum-collected real-time traces would strengthen the claim.

---

## 8. Conclusion

**[Two-paragraph summary tied to the actual GO/NO-GO outcome.]**

---

## Reproducibility

Source code: <https://github.com/thc1006/fl-oran-tmc> (private during review,
Apache-2.0 on acceptance). Commit hash and `requirements.lock` in the
supplementary. Hardware: single NVIDIA RTX 4080 (16 GiB VRAM), Ubuntu 24.04,
Python 3.12.3, PyTorch 2.10 + CUDA 12.8.

Run the full sweep with:

```bash
python experiments/run_v6_arch_sweep.py \
  --arch lstm,mamba,spiking \
  --seeds 42,0,1,2,3,7,11,13,17,23 \
  --total-steps 5000
python scripts/aggregate_v6_results.py
```

Total wall-clock on the listed hardware: **[ELAPSED]** (measured in
`artifacts/logs/v6_arch_sweep.log`).
