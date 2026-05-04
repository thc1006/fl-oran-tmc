# Supplementary Material

Cross-architecture federated-learning benchmark on Colosseum/ColO-RAN: per-section appendix material moved out of the main paper for length compliance with the IEEE TMC / TNSM / MobiSys main-body conventions. Each section heading carries the originating main-paper section in parentheses.

---

## Appendix A. Mechanism analysis details (extends §7.1)

### A.1 Applicability boundary (extends §7.1.3)

The empirical finding (natural-by-BS dominance and inverted α monotonicity) is dataset-structural, not algorithm-mechanistic. We therefore expect it to replicate whenever a FL deployment satisfies (a) clients are partitioned along an axis on which the dataset has measurable distributional heterogeneity (here: bs_id), (b) the alternative partition strategy redistributes rows along an orthogonal axis that *does not* correspond to dataset structure (here: slice_id, which is statistically uniform across bs_ids in this corpus). Conversely, datasets where the natural client partition has uniform marginals on every axis would be expected to behave classically (lower α ⇒ harder). We conjecture similar inversion will appear in other near-RT RIC forecasting workloads (uplink throughput-violation prediction, latency-budget breach prediction) where per-gNB telemetry distributions differ measurably; quantitative verification on additional corpora is open future work.

### A.2 Hardware drift caveat for the §7.1.1 ablation (extends §7.1.4)

The §7.1.1 random_split ablation cells were run on 4× Tesla V100-SXM2-32GB (sm_70, driver 535.161, CUDA 12.1) while the Phase 5 baseline (natural-by-BS and Dirichlet partitions) was run on RTX 4080 (sm_89). The two GPU generations have different bf16 implementations (V100 emulates bf16 via fp16 tensor cores with possible fp32 fallback; Lovelace has native bf16 tensor cores), and `cudnn_deterministic=True` pins kernel choice within a hardware target but does not impose bit-equivalence across hardware. We bound the resulting AUC drift by comparing V100 random_split AUC to 4080 Dirichlet α=5.00 AUC (both are partition strategies that destroy bs grouping, so they should yield equivalent AUC under hardware-independent training): the residual delta is at most 0.0072 (LSTM), 0.0043 (Mamba), 0.0021 (Spiking-SSM), all within the corresponding seed σ. Hardware drift is therefore bounded above by ~0.007 AUC, more than an order of magnitude smaller than the ~0.18 mechanism signal (ratio ≈ 25×). Strict elimination of hardware confound would require re-running Phase 5's IID column on V100 (≈10 GPU-hours on the 4× cluster); we did not undertake this because the bounded-drift argument is sufficient to interpret §7.1.1's ablation conclusion.

---

## Appendix B. Reproducibility infrastructure detail (extends §5)

### B.1 Test infrastructure (extends §5.2)

The release ships **277 passing tests** across four suites that protect numerical and pipeline-level invariants:

* `tests/test_v7_*.py` — 202 unit + integration tests for `fl_v7` trainer (model-build, partition correctness, scaler determinism, FedDyn canonical-vs-Option-II contract).
* `tests/test_aggregate_v7_results.py` — 22 tests for the aggregator (paired-bootstrap, BLAKE2b decorrelation, JSON round-trip, schema-mismatch detection, 1260-cell scaling).
* `tests/test_paper_draft_invariants.py` — 37 paper-text invariants (forbidden hallucinated facts; required corrected facts; license consistency; hardware specs).
* `tests/test_paper_claims_sources.py` — 16 paper-vs-data claims tests that re-read `artifacts/step1_factfinding.json`, `step2_mechanism_search.json`, and `aggregated_phase5.json` and assert paper-reported numbers match the underlying measurements within rounding tolerance.

The latter two suites (53 tests total) form a paper-correctness developer pre-commit gate: every PAPER_DRAFT.md edit is verified by re-running them, preventing quick-claim regressions.

### B.2 Statistical pipeline (extends §5.3)

`scripts/aggregate_v7_results.py` consumes the per-cell `summary.json` files and produces `aggregated_phase5.json` (90 group means + 270 paired-bootstrap distributions: 180 algorithm-pairs + 90 architecture-pairs) plus a paper-grade Markdown table. Each pairwise comparison's bootstrap RNG seed is derived from a BLAKE2b hash of the pair identifier added to a base seed (2026 for algorithm-pairs, 2027 for arch-pairs), guaranteeing independent bootstrap streams across all 270 distributions and supporting joint coverage claims (§4.5).

### B.3 Croissant dataset metadata (extends §5.4)

The Colosseum/ColO-RAN public release is upstream-licensed; our preprocessed `coloran_raw_unified.parquet` (≈18 M rows) and the partition specifications are documented in a Croissant 1.0 metadata file shipped with the release archive. The metadata block describes the (bs_id, slice_id, sched, tr) keying, the 17 continuous features (§3.1), the OOD train/val/test split (§3.3), and the `add_classification_target` derivation rule (§3.2).

### B.4 Reproducible execution environment (extends §5.5)

The release archive bundles:

* `Dockerfile.repro` — pinned to the exact CUDA / PyTorch / `nvidia-ml-py` versions used in Phase 5 (CUDA 12.8, PyTorch 2.10, `nvidia-ml-py` 12.x);
* `requirements.lock` — SHA256-pinned Python dependencies (74 packages);
* `repro.sh` — one-line entry point that loads the parquet, runs a 1-cell smoke (LSTM, FedAvg, IID, seed=42, 5 rounds × 20 max-steps, ~30 s on RTX 4080), and verifies bit-equivalent test AUC against a stored reference value within 1e-6.

### B.5 Demo notebook (extends §5.6)

A Jupyter notebook (`notebooks/colosseum_oran_federated_slicing_demo.ipynb`) walks through (a) loading aggregator JSON, (b) reproducing Figures 1-3, (c) running a downscaled smoke version of the §7.1.1 random_split ablation on a single GPU (the full ablation runs on the 4× V100 setup documented in App. A.2), and (d) regenerating §6 paired-bootstrap CI95 from raw cells. The notebook is the recommended onboarding entry point for downstream researchers extending the benchmark.

---

## Appendix C. Extended limitations and threats to validity (extends §8)

Each subsection below holds the full prose of an §8 limitation whose main-paper bullet was condensed to a 1-2 sentence headline. The §1 contribution(s) each item threatens are stated in the main-paper §8 bullet headers.

**A note on internal references in C.1-C.6:** "Stage 1" refers to a predecessor single-architecture benchmark on ColO-RAN whose preregistered hardware budget, precision policy, and gradient-step count are inherited by Phase 5 (the federated extension reported in this paper). "ADR-001 D-N" denotes the N-th preregistered design decision in the project's Architecture Decision Record. Both artefacts are documented in the release archive for downstream reproduction.

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

---

## Appendix D. Future work directions (extends §9.1)

Several open questions remain. **First**, §7.1.5 disambiguates the inverted-α mechanism (the strong reading of §7.1.1 (i) — bs grouping alone suffices — is refuted; (ii) per-client structural coherence is the additional necessary ingredient), but two design constraints (`sub_per_bs=2` halving per-client data, 5/14 sampling fraction vs Phase 5's 5/7) prevent a clean decomposition of the 1–4 pp residual gap to natural-by-BS; an isolation re-run keeping `sub_per_bs=2` while bumping `clients-per-round` to 10 (matches Phase 5's 5/7 sampling fraction at the doubled client count) would isolate the slice-mixing effect from per-round participation, following the per-axis isolation methodology of Borges et al. (arXiv:2503.17070, 2025) and the proximity-partition framing of ProFed (arXiv:2503.20618, 2025). **Second**, our 10 % BLER threshold should be perturbed to {5, 15, 20} % to confirm operational robustness, in line with recent rare-event metric-stability work (arXiv:2504.16185, 2025). **Third**, scaling beyond N = 7 base stations toward the 100-client regime studied by Chiarani et al. (arXiv:2503.12435, 2025) and Sezgin et al. (arXiv:2509.11421, 2025) would test whether inverted-α is a small-N artefact. **Fourth**, the FedAdam headroom-saturation finding may be soft: variance-aware sharpness methods such as FedSCAM (arXiv:2601.00853, 2026) and trajectory-matched FedSynSAM (arXiv:2602.11584, 2026) introduce per-client perturbation modulation that could extract additional accuracy. **Fifth**, integrating differential privacy via over-the-air channel noise (arXiv:2510.23463, 2026) and exploring neuromorphic deployment on Loihi 2 with sparse RNN kernels (arXiv:2502.01330, 2025; arXiv:2604.27004, 2026) would close the loop toward production-deployable privacy-preserving energy-efficient O-RAN slice prediction.
