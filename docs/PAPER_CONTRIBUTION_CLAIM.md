# Paper Contribution Claim — IEEE TMC submission

**Provisional title**: *Federated O-RAN Slice SLA Prediction Across Architectures and Heterogeneity Regimes: A Comprehensive Benchmark on Colosseum/ColO-RAN*

**Target venue**: IEEE Transactions on Mobile Computing (JIF 7.6).
**Fallback venue**: IEEE Open Journal of the Communications Society (JIF 6.1).
**Target submission**: Q3 2026 (8-12 weeks from 2026-04-27).
**Status**: 2026-04-27 night — venue + narrative locked; awaiting user authorization for Phase 5 Stage 2 FULL sweep.

## The Separability Assumption Hypothesis (paper's central scientific question)

Existing federated learning benchmarks for cellular networks treat the deployment design space as four separable axes:

1. **Architecture** — pick best (LSTM/Mamba/Spiking-SSM) via centralized benchmark.
2. **Algorithm** — pick best (FedAvg/FedProx/...) via NIID-Bench-style FL benchmark.
3. **Heterogeneity** — pick representative α for non-IID via NIID-Bench convention.
4. **Hardware** — pick deployment GPU via SNN-vs-neuromorphic comparison.

The implicit assumption: **best architecture × best algorithm × representative α × commodity hardware = best deployment**.

**This paper tests that assumption empirically.**

## Central scientific contribution

We provide the first 6D empirical study testing whether the four axes are separable, on the publicly-available Colosseum/ColO-RAN open testbed dataset. Our results establish whether prior single-axis benchmarks support joint deployment inference, or whether emergent joint interactions invalidate axis-by-axis recommendations.

[FILL IN AFTER PHASE 3a/3c/3e/5 RESULTS]

If separability holds → engineering value: existing single-axis benchmarks remain usable for joint inference.
If separability fails → scientific contribution: expose joint interactions that single-axis benchmarks cannot reveal.

## Six dimensions of integration

1. **Architecture set**: LSTM (`ForecasterV2`), Mamba (`MambaForecaster`), Spiking-SSM (`spiking_expand2` per ADR D-21 final outcome).
2. **Algorithm set**: FedAvg, FedProx (`mu=0.01`), FedAdam (`server_lr=0.01`), SCAFFOLD, FedDyn (`alpha=0.01`). MOON deferred per ADR D-22 (encode_fn × spiking is paper-level open question).
3. **Heterogeneity**: parametric Dirichlet α∈{0.05, 0.1, 0.5, 1.0, 5.0} (5-grid over `slice_id`); extreme stress α∈{0.01} as supplementary failure-envelope evidence (Phase 3e).
4. **Dataset**: Colosseum/ColO-RAN 18M-row real RAN telemetry traces. Public testbed (NSF + ONR funded). Dataset on GitHub: `wineslab/colosseum-oran-coloran-dataset`.
5. **Energy/Latency measurement**: NVML per-cell instrumentation on RTX 4080 (commodity edge GPU). Cross-GPU robustness smoke on T4 + A100 (Phase 3c).
6. **Reproducibility**: Spec-driven YAML loader with pre-flight + skip-completed; paired-bootstrap CI95 with Bonferroni multi-comparison correction; cudnn_deterministic + bf16 + fixed-seed; full open-source release with Croissant metadata + Zenodo DOI.

## Engineering deliverable

A 6D decision lookup table for O-RAN edge-FL practitioners. Given (arch, algo, α, hardware) → expected AUC + latency + energy + paired CI95. Released as open dataset on Zenodo with permanent DOI; reproducibility infrastructure released on GitHub under MIT license.

## What we cite (must-cite shortlist; full §relatedWork in paper)

### FL benchmark precedent (~7 citations)
LEAF (Caldas 2018), Flower (Beutel 2020), FedScale (Lai 2022), FLAIR (Song 2022), pfl-research (Granqvist 2024), fev-bench (Shchur 2025), arXiv 2508.08479 FL throughput (2025).

### FL × SLA/RAN/wireless (~7 citations)
Polese et al. 2022 (ColO-RAN, IEEE TMC), Polese et al. 2024 (Colosseum digital twin, IEEE OJ-COMS), Statistical FL for B5G SLA (IEEE 2021), CDF-Aware FL (IEEE 2021), Fed-LSTM (2022), TRACTOR (2023), **Mangi et al. 2026** (regime-aware FL-SLA — closest competitor, narrower scope).

### FL × SNN (~6 citations)
arXiv 2106.06579 (founding), FedLEC 2024, Privacy-FL-SNN 2025, Robustness-FL-SNN 2025, **arXiv 2602.12009 (Feb 2026, DP×SNN×FL)** — Phase 3b deferral citation.

### SNN GPU energy methodology (~6 citations)
**Shen et al. 2023** (Bit Budget — direct prior art for reality factor), NeuroBench 2025, SpikingBrain 2025, Prosperity 2025, STEP 2025, ML.ENERGY 2025, Where Do the Joules Go? 2026.

### Mamba × FL × cellular (~3 citations)
Habib et al. 2025/2026 (Mamba in 6G IDN, no FL); SpikMamba 2024, SpikingMamba 2025 (no FL).

### Heterogeneity in FL (~4 citations)
NIID-Bench, FedBN (Li et al. ICLR 2021), pFedFDA (NeurIPS 2024), embedding-skew (Borazjani et al. 2025).

### 6G/wireless benchmark genre (~2 citations)
6G-Bench (Debbah 2026), ORAN-Bench-13K (Gajjar 2024).

**Total: ~35 must-cite. Full paper §relatedWork: 50-80 citations expected.**

## What we do NOT claim

- ❌ Novel FL aggregation algorithm (we use 5 standard algos; MOON deferred per D-22).
- ❌ Novel architecture (LSTM/Mamba/Spiking-SSM are existing; `spiking_expand2` is Mamba's expand=2 trick applied to spiking-SSM, not new design).
- ❌ Novel discovery that FLOP-as-energy proxy is wrong (Shen 2023, ML.ENERGY 2025 already documented).
- ❌ DP × FL × SNN methodology (preempted by arXiv 2602.12009; cite as §discussion deferral).
- ❌ Novel non-IID definition (we use standard Dirichlet; cite Borazjani 2025 as advanced alternative).

## What we DO claim

- ✅ **First 6D integration benchmark on Colosseum/ColO-RAN open testbed**.
- ✅ **Empirical test of FL deployment separability assumption** with parametric α and 5 algorithms.
- ✅ NVML reality factor extension from centralized [Shen 2023] to **FL setting**.
- ✅ Architecture-variant boundary documentation: `spiking_expand2` vs vanilla on commodity edge GPU.
- ✅ **Open-source reproducible reference benchmark** with Croissant metadata + Zenodo DOI + spec-driven pipeline + pre-flight + paired-bootstrap CI95 infrastructure.

## Limitations (be upfront in §limitations)

1. **Single primary hardware**: RTX 4080 16GB (sm_89, BF16, cudnn_deterministic). Cross-GPU smoke on T4 + A100 confirms AUC robustness; latency/EDP numbers RTX 4080-specific.
2. **Single dataset**: ColO-RAN only. Cross-dataset generalization not tested.
3. **No MOON algorithm**: deferred per ADR D-22 (encode_fn × spiking is paper-level open question).
4. **No DP-SGD empirical study**: preempted by arXiv 2602.12009 (Feb 2026); cited as §discussion deferral.
5. **7 clients fixed**: matches ColO-RAN 7-gNB testbed; scaling beyond 7 not tested.
6. **Sequence length fixed at seq_len=5**: M5 baseline; longer-sequence forecasting not in scope.

## Update history

- **2026-04-27 night**: Initial creation. Venue locked = IEEE TMC primary / OJ-COMS fallback after 3-round deep lit review. Separability narrative adopted. Phase 3 plan FINAL (drop 3b + 3d, add 3e, revise 3a; new Phase 5 Stage 2 FULL).
