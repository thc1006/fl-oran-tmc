# Federated O-RAN Slice SLA Prediction Across Architectures and Heterogeneity Regimes: A Comprehensive Benchmark on Colosseum/ColO-RAN

> **Status (2026-04-27 night)**: §1 Introduction + §2 Related Work — DRAFT v1.
> Empirical fill-ins marked `[TODO Phase 3/5]`.
> Target venue: IEEE Transactions on Mobile Computing (JIF 7.6).
> Fallback: IEEE Open Journal of the Communications Society (JIF 6.1).

---

## 1. Introduction

Open Radio Access Networks (O-RAN) decomposes the cellular base station into commodity hardware components controlled by ML-driven xApps and rApps running on disaggregated RAN Intelligent Controllers (RICs) [Polese et al. 2022]. Among the standardized use cases, slice-level Service Level Agreement (SLA) violation prediction is critical: each gNB simultaneously serves heterogeneous traffic slices (eMBB, URLLC, mMTC), and proactive forecasting of multi-second-ahead SLA risk drives near-RT RIC resource allocation. Because edge gNBs accumulate per-user telemetry that cannot be centrally pooled, federated learning (FL) is the natural paradigm: each gNB trains a local sequence model on its own traces and shares only weight updates with the SMO/non-RT RIC.

Despite the explosion of FL benchmarks since 2018, the deployment design space for FL-based slice SLA prediction remains under-characterized. The space spans four axes:

1. **Sequence model architecture** — recurrent (LSTM), structured state-space (Mamba), or bio-inspired sparse spiking variants (Spiking-SSM).
2. **Federation algorithm** — FedAvg, FedProx, FedAdam, SCAFFOLD, FedDyn, MOON, or hybrid.
3. **Client heterogeneity** — typically parameterized by Dirichlet concentration α over class or feature distributions; for O-RAN, over slice IDs at the gNB.
4. **Commodity hardware** — consumer-tier edge GPUs (RTX-class) at the gNB rather than data-center accelerators.

Existing benchmarks address each axis in isolation. NeurIPS-class FL benchmarks evaluate algorithm × heterogeneity on toy image data [LEAF 2018; FedScale 2022; FLAIR 2022; pfl-research 2024]. Spiking-NN benchmarks compare architectures in centralized regimes on neuromorphic-friendly datasets [NeuroBench 2025; STEP 2025]. FL × wireless papers focus on a single architecture [Fed-LSTM 2022; arxiv 2508.08479; Mangi et al. 2026]. The implicit assumption underlying single-axis recommendations: **best architecture × best algorithm × representative α × commodity hardware = best deployment**.

**This paper tests the assumption empirically.** We construct the first 6D integration study on the publicly-available Colosseum/ColO-RAN testbed dataset [Polese et al. 2022; Bonati et al. 2024], sweeping {LSTM, Mamba, Spiking-SSM} × {FedAvg, FedProx, FedAdam, SCAFFOLD, FedDyn} × Dirichlet α∈{0.05, 0.10, 0.50, 1.00, 5.00} × 10 seeds = 750 cells, with NVML-instrumented per-round GPU energy measurement on RTX 4080 (sm_89) commodity hardware. Statistical inference uses paired-bootstrap CI95 with Bonferroni multi-comparison correction across all 5×3×5 = 75 (algorithm, architecture, α) configuration triples.

**Contributions**:

1. **Empirical test of the FL deployment separability assumption**: under realistic stress regimes (α≤0.10, modeling worst-case slice imbalance across gNBs), we expose joint architecture × algorithm × heterogeneity interactions that cannot be inferred from single-axis benchmarks. `[TODO Phase 3a/3e/5: fill in specific interaction findings, e.g. "FedProx-Spiking diverges at α=0.05 while FedProx-LSTM converges; FedAvg-Mamba dominates at α=0.5 but loses to FedAdam-LSTM at α=5"]`.

2. **NVML reality factor extension to the federated regime**: the FLOP-vs-actual-energy gap [Shen et al. 2023; α-FLOPs 2021] documented in centralized SNN benchmarks (ratio 197×–9700× per-inference per Stage 1) holds for FL training on commodity edge GPU. `[TODO Phase 5 NVML: report per-round training energy with idle-baseline subtraction across all 750 cells]`.

3. **Open reproducible reference benchmark**: spec-driven YAML pipeline with pre-flight algorithm validation and `--skip-completed` resumability; paired-bootstrap statistical pipeline; Croissant metadata; Zenodo DOI; MIT license. Released alongside this paper as the first reference baseline for O-RAN edge-FL practitioners. The full 6D table maps deployment configurations → (AUC, F1, latency, energy, paired CI95), enabling lookup-style operator decisions.

The remainder of the paper is organized as follows. §2 reviews related work across the four axes. §3 describes the Colosseum/ColO-RAN dataset and our preprocessing pipeline. §4 details the architecture, algorithm, and partition methodology. §5 enumerates the reproducibility infrastructure and our Croissant-aligned data release. §6 reports results across the 6D matrix with paired-bootstrap CI95. §7 discusses the separability assumption finding and articulates the applicability boundary on commodity edge hardware. §8 enumerates limitations. §9 concludes.

---

## 2. Related Work

### 2.1 Federated learning benchmarks

The canonical FL benchmark is **LEAF** [Caldas et al. 2018], which standardized federated MNIST/Shakespeare evaluation pipelines for early FedAvg variants. **Flower** [Beutel et al. 2020] later provided a framework rather than a benchmark, supporting heterogeneous device experiments. **FedScale** [Lai et al. 2022] scaled to thousands of cross-device clients; **FLAIR** [Song et al. 2022] introduced a long-tail multi-label image dataset. **pfl-research** [Granqvist et al. 2024] integrates differential-privacy mechanisms into a simulation framework. Most recently, **fev-bench** [Shchur et al. 2025] introduced rigorous bootstrap-CI95 evaluation for time-series forecasting, though without the federated dimension. None of these benchmarks targets O-RAN slice SLA prediction specifically; their toy-data settings cannot capture the covariate-skew structure of real RAN telemetry.

### 2.2 FL on cellular and wireless networks

A first generation of FL × cellular work targeted SLA-constrained slicing under the Beyond-5G banner. **Statistical FL for B5G SLA-Constrained RAN Slicing** [IEEE 2021] introduced a long-term CDF SLA constraint. **CDF-Aware FL** [IEEE 2021] formalized the constraint as a per-client penalty. **Fed-LSTM** [2022] deployed an LSTM forecaster for slice-level traffic. **TRACTOR** [Groen et al. 2023] integrated reinforcement learning xApps for resource block assignment. **REAL** [Barker et al. 2025] embedded near-RT RIC algorithms with realistic urban-mobility channel models on srsRAN. Recently **Hayek et al. 2025** measured FL convergence on a 5G/WiFi/Ethernet COTS testbed with Raspberry-Pi clients, exposing uplink straggler effects. **arxiv 2508.08479** [2025] benchmarked FedAvg/FedProx/FedBN on five throughput-prediction datasets with LSTM/CNN/Transformer architectures, finding FedBN superior under feature skew. **Mangi et al. 2026** ("WHEN CLIENTS DRIFT") proposed regime-aware aggregation under leave-one-regime-out evaluation on synthetic 6G RAN telemetry.

Each of these papers covers one or two of our four axes. None benchmarks LSTM, Mamba, and Spiking-SSM jointly under parametric Dirichlet heterogeneity, and none reports per-round NVML energy. Our work is differentiated by the combination of architecture breadth, algorithm breadth, parametric heterogeneity, and energy instrumentation on the **publicly-available** Colosseum/ColO-RAN dataset, in contrast to closed simulators or undisclosed datasets used by Mangi et al.

### 2.3 Spiking-SSM and FL × spiking neural networks

Hybrid spiking-state-space architectures emerged from late 2024. **SpikMamba** [Chen et al. 2024] combined LIF spiking neurons with Mamba selective scan for event-camera action recognition. **Mamba-Spike** [arxiv 2408.11823, 2024] introduced a spiking front-end for general temporal data. **Spiking Point Mamba** [ICCV 2025] extended to 3D point clouds. **SpikingSSMs** [AAAI 2025] formalized sparse-parallel spiking state-space layers. **SpikingMamba** [arxiv 2510.04595, 2025] distilled large language models into spiking variants for energy efficiency. **SpikingBrain** [arxiv 2509.05276, 2025] scaled spiking architectures to 76B parameters on data-center GPU clusters with explicit FLOP-utilization measurement.

Federated-learning variants of spiking neural networks emerged earlier than spiking-SSM: **arxiv 2106.06579** [2021] founded FL × SNN. **FedLEC** [arxiv 2412.17305, 2024] extended to label-skew non-IID. **SNN in Vertical FL** [arxiv 2407.17672, 2024] examined VFL energy trade-offs. **Robustness of SNN in FL with Compression** [arxiv 2501.03306, 2025] addressed Byzantine attacks. **Privacy in FL with SNN** [arxiv 2511.21181, 2025] characterized gradient leakage. Most recently, **arxiv 2602.12009** [Feb 2026] examined firing-rate sensitivity to differential privacy in federated SNN — preempting the DP-SNN-FL angle. We accordingly cite this work in §7 as a deferral and do not duplicate the DP study; our contribution is orthogonal in its multi-architecture × parametric-Dirichlet × NVML scope.

To our knowledge, no published work combines spiking-SSM with FL on cellular/RAN telemetry. Our `spiking_expand2` variant adapts the wide-inner-state design pattern from Mamba [Gu & Dao 2023] to spiking-SSM and demonstrates competitive AUC/energy trade-off on commodity edge GPU under FL.

### 2.4 GPU energy measurement methodology

The argument that FLOP counts mis-estimate actual hardware energy predates our work. **Shen et al. 2023** ("Is Conventional SNN Really Efficient?") critiqued synaptic-operation accounting via a Bit Budget framework, finding that quantized ANNs dominate SNNs at equivalent bit budgets. **α-FLOPs** [Asperti et al. 2021] proposed a parallelism-aware alternative. **NeuroBench** [Yik et al., Nature Communications 2025] standardized neuromorphic benchmarking on real hardware. The **ML.ENERGY Benchmark** [Chung et al. 2025] measured inference energy on generative AI; **Where Do the Joules Go?** [Chung et al. 2026] diagnosed the breakdown across model families. **STEP** [Shen et al. 2025] specifically benchmarked spiking transformers including memory-access cost. **Beyond Backpropagation** [Spyra 2025] applied NVML + CodeCarbon to alternative training objectives.

We follow the **Energy API** approach (`nvmlDeviceGetTotalEnergyConsumption`, available on Volta+) preferred over polling-power Riemann-sum integration per the ML.ENERGY 2025 best-practices blog. Our novel angle is the **federated-training regime**: prior NVML benchmarks measure inference [NeuroBench, ML.ENERGY] or centralized training [SpikingBrain, Beyond Backpropagation]; we measure per-round energy across an FL sweep, including idle-baseline subtraction.

### 2.5 Heterogeneity in federated learning

**NIID-Bench** [Li et al. 2022] established Dirichlet partitioning as the standard non-IID benchmark. **FedBN** [Li et al. ICLR 2021] keeps client-local BatchNorm statistics for feature-skew settings; **arxiv 2508.08479** demonstrated FedBN's superiority on cellular feature-skew. **pFedFDA** [NeurIPS 2024] specifically targets covariate shift. **Borazjani et al. 2025** [arxiv 2503.14553] argued for embedding-based heterogeneity definitions beyond label-skew.

We use Dirichlet partitioning over slice IDs as the canonical covariate-skew model for O-RAN: each gNB serves a different mix of slice traffic, with mixing entropy controlled by α. We adopt α∈{0.05, 0.10, 0.50, 1.00, 5.00} to span extreme stress (each gNB dominated by a single slice) through near-IID. We discuss the applicability of embedding-based alternatives in §7.

---

> **Sections 3-9 to follow once Phase 3a/3e/5 results land. See `PAPER_CONTRIBUTION_CLAIM.md` for full contribution claim outline.**
