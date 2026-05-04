# Supplementary Material

Cross-architecture federated-learning benchmark on Colosseum/ColO-RAN: per-section appendix material moved out of the main paper for length compliance with the IEEE TMC / TNSM / MobiSys main-body conventions. Each section heading carries the originating main-paper section in parentheses.

---

## Appendix A. Mechanism analysis details (extends §7.1)

### A.1 Applicability boundary (extends §7.1.3)

The empirical finding (natural-by-BS dominance and inverted α monotonicity) is dataset-structural, not algorithm-mechanistic. We therefore expect it to replicate whenever a FL deployment satisfies (a) clients are partitioned along an axis on which the dataset has measurable distributional heterogeneity (here: bs_id), (b) the alternative partition strategy redistributes rows along an orthogonal axis that *does not* correspond to dataset structure (here: slice_id, which is statistically uniform across bs_ids in this corpus). Conversely, datasets where the natural client partition has uniform marginals on every axis would be expected to behave classically (lower α ⇒ harder). We conjecture similar inversion will appear in other near-RT RIC forecasting workloads (uplink throughput-violation prediction, latency-budget breach prediction) where per-gNB telemetry distributions differ measurably; quantitative verification on additional corpora is open future work.

### A.2 Hardware drift caveat for the §7.1.1 ablation (extends §7.1.4)

The §7.1.1 random_split ablation cells were run on 4× Tesla V100-SXM2-32GB (sm_70, driver 535.161, CUDA 12.1) while the Phase 5 baseline (natural-by-BS and Dirichlet partitions) was run on RTX 4080 (sm_89). The two GPU generations have different bf16 implementations (V100 emulates bf16 via fp16 tensor cores with possible fp32 fallback; Lovelace has native bf16 tensor cores), and `cudnn_deterministic=True` pins kernel choice within a hardware target but does not impose bit-equivalence across hardware. We bound the resulting AUC drift by comparing V100 random_split AUC to 4080 Dirichlet α=5.00 AUC (both are partition strategies that destroy bs grouping, so they should yield equivalent AUC under hardware-independent training): the residual delta is at most 0.0072 (LSTM), 0.0043 (Mamba), 0.0021 (Spiking-SSM), all within the corresponding seed σ. Hardware drift is therefore bounded above by ~0.007 AUC, more than an order of magnitude smaller than the ~0.18 mechanism signal (ratio ≈ 25×). Strict elimination of hardware confound would require re-running Phase 5's IID column on V100 (≈10 GPU-hours on the 4× cluster); we did not undertake this because the bounded-drift argument is sufficient to interpret §7.1.1's ablation conclusion.
