# Supplementary Material

Cross-architecture federated-learning benchmark on Colosseum/ColO-RAN: per-section appendix material moved out of the main paper for length compliance with the IEEE TMC / TNSM / MobiSys main-body conventions. Each section heading carries the originating main-paper section in parentheses.

---

## Appendix A. Mechanism analysis details (extends §7.1)

### A.1 Applicability boundary (extends §7.1.3)

The empirical finding (natural-by-BS dominance and inverted α monotonicity) is dataset-structural, not algorithm-mechanistic. We therefore expect it to replicate whenever a FL deployment satisfies (a) clients are partitioned along an axis on which the dataset has measurable distributional heterogeneity (here: bs_id), (b) the alternative partition strategy redistributes rows along an orthogonal axis that *does not* correspond to dataset structure (here: slice_id, which is statistically uniform across bs_ids in this corpus). Conversely, datasets where the natural client partition has uniform marginals on every axis would be expected to behave classically (lower α ⇒ harder). We conjecture similar inversion will appear in other near-RT RIC forecasting workloads (uplink throughput-violation prediction, latency-budget breach prediction) where per-gNB telemetry distributions differ measurably; quantitative verification on additional corpora is open future work.

### A.2 Hardware drift caveat for the §7.1.1 ablation (extends §7.1.4)

The §7.1.1 random_split ablation cells were run on 4× Tesla V100-SXM2-32GB (sm_70, driver 535.161, CUDA 12.1) while the Phase 5 baseline (natural-by-BS and Dirichlet partitions) was run on RTX 4080 (sm_89). The two GPU generations have different bf16 implementations (V100 emulates bf16 via fp16 tensor cores with possible fp32 fallback; Lovelace has native bf16 tensor cores), and `cudnn_deterministic=True` pins kernel choice within a hardware target but does not impose bit-equivalence across hardware. We bound the resulting AUC drift by comparing V100 random_split AUC to 4080 Dirichlet α=5.00 AUC (both are partition strategies that destroy bs grouping, so they should yield equivalent AUC under hardware-independent training): the residual delta is at most 0.0072 (LSTM), 0.0043 (Mamba), 0.0021 (Spiking-SSM), all within the corresponding seed σ. Hardware drift is therefore bounded above by ~0.007 AUC, more than an order of magnitude smaller than the ~0.18 mechanism signal (ratio ≈ 25×). Strict elimination of hardware confound would require re-running Phase 5's IID column on V100 (≈10 GPU-hours on the 4× cluster); we did not undertake this because the bounded-drift argument is sufficient to interpret §7.1.1's ablation conclusion.

---

## Appendix C. Extended limitations and threats to validity (extends §8)

Each subsection below holds the full prose of an §8 limitation whose main-paper bullet was condensed to a 1-2 sentence headline. The §1 contribution(s) each item threatens are stated in the main-paper §8 bullet headers.

### C.1 L9 full discussion: pos_weight_split=train choice (extends §8 L9)

We compute the BCE positive-class weight from the train partition's positive rate (per ADR-001 D-12), in line with standard practice that prevents test-set positive-rate from leaking into the loss. Two alternative computations (val-derived pos_weight; held-out-fold pos_weight) would also be valid and might shift absolute AUC numbers slightly, but per §7.1 our V100 random_split ablation confirms the inverted-α mechanism direction is robust to the partition-axis shuffle, and the BCE loss reweighting is identical (globally pooled) across all clients regardless of which split the rate is computed from — so the *direction* of all our findings is invariant to this choice.

### C.2 L10 full discussion: bf16 mixed-precision training versus fp32 (extends §8 L10)

All Phase 5 cells use bf16 mixed precision per the Stage 1 inheritance contract for memory and throughput on RTX 4080. The numerical-precision gap between bf16 and fp32 is bounded for our model architectures by the lack of overflow-prone operations (no exponential losses, no large-fan-in matmul without LayerNorm); we do not report a fp32 verification cell and treat bf16 vs fp32 cross-check as a reproducibility-supplementary item for the camera-ready archive. Readers reproducing on hardware without native bf16 (Pascal sm_60 and earlier) should expect O(0.001) AUC drift.

### C.3 L11 full discussion: per-client compute budget split (extends §8 L11)

Our (round, max_steps) configuration matches Stage 1's audit-corrected 25 000 total-gradient-steps budget. An alternative split (e.g., 50 rounds × 100 max-steps, 200 rounds × 25 max-steps) would shift the round-vs-step ratio and potentially affect server-side aggregation count by 2-4×; we did not sweep this dimension, treating round count as a fixed reproducibility anchor. Whether the FedAdam-saturates-headroom finding (§6.3) survives at 4× more rounds with proportionally fewer steps per round is open.

### C.4 L12 full discussion: SLA-threshold sensitivity (extends §8 L12)

The 10 % BLER threshold is the canonical ColO-RAN SLA gate (Polese et al. 2022, §V); the empirical positive rate at this threshold on the train partition is 30.9 % (§7.1 setup-level facts). A threshold sweep over 5 % / 15 % / 20 % would test whether the inverted-α monotonicity strengthens at sparser-positive regimes (where any structural specialisation has higher loss-leverage) and weakens at higher thresholds; we did not run this sweep but predict (a) inversion strengthens as threshold tightens and (b) inversion weakens or disappears as threshold approaches the symmetric 50 % regime. This is the highest-leverage open ablation for the camera-ready revision after §7.1.1.

### C.5 L13 full discussion: bootstrap CI percentile vs BCa (extends §8 L13)

§4.5 explains the choice: at our n=10 paired seeds with approximately symmetric per-seed delta distributions, the bias-correction term in the BCa interval is bounded above by O(1/√n) ≈ 0.32 standard errors; this magnitude is smaller than our seed σ on every reported finding except §6.4's Mamba×SCAFFOLD interaction, where σ inflation by 23× makes the CI choice less consequential than the Wilcoxon p-value. Readers who prefer BCa should treat our CI95 as a slightly tighter version of the BCa interval; the qualitative conclusions are unchanged.

### C.6 L14 full discussion: FedAdam hyperparameter sensitivity (extends §8 L14)

β₁ = 0.9, β₂ = 0.99, η_server = 0.01 follow the FedAdam paper's preregistered values (Reddi et al. 2021 Table 1); β₂ = 0.99 differs from canonical Adam's 0.999. We did not run a (β₁, β₂, η_server) sensitivity sweep — the FedAdam-vs-FedAvg deltas reported in §6.3 are anchored to paper-canonical hyperparameter choices rather than to a per-task hyperparameter optimum. Whether the FedAdam-saturates-headroom finding survives at alternative β₂ (e.g., 0.999) is open; we expect modest sensitivity (≤ 0.005 AUC drift) given Adam's relative robustness to β₂ variation in offline ML literature, but this is anticipated, not measured.
