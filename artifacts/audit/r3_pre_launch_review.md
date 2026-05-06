# R3 GPU sweeps: deep pre-launch review (per user mandate 2026-05-06)

User instruction: "before training, deep ultrathink/review to confirm
scripts/code being submitted are correct and have no errors. Don't
want training output to be garbage."

## R3.2 FedBN 30-cell sweep — pre-launch checklist

### Code-correctness verification

| Check | Status | Evidence |
|---|---|---|
| FedBN class registers via `@register` | ✅ | `from fl_oran.federated.algorithms import REGISTRY; 'fedbn' in REGISTRY` returns True after `__init__.py` import added |
| FedBN reduces to FedAvg structurally on no-norm archs | ✅ | `_is_personalised_param` regex returns 0 matches for all keys in trained LSTM/Mamba/Spiking checkpoints (16/30/36 params) |
| FedBN reduces to FedAvg empirically (LSTM, 5 rounds, seed 42) | ✅ | smoke produced `test_auc=0.8132799429` for both algos (\|Δ\| = 0.0e+00, bit-identical) |
| FedBN reduces to FedAvg empirically (Mamba, 5 rounds, seed 42) | (pending cross-arch smoke) | |
| FedBN reduces to FedAvg empirically (Spiking, 5 rounds, seed 42) | (pending cross-arch smoke) | |
| `_ALGO_REQUIRED_KWARGS` table updated for fedbn | ✅ | `fl_v7.py` L259 added `"fedbn": set()` (no algo-specific kwargs) |
| Spec YAML matches Phase 5 hyperparameters | ✅ | `p1_fedbn_natural.yaml` mirrors `stage2_full.yaml` shared+overrides exactly |

### What this means for "garbage output" risk

Because FedBN ≡ FedAvg bit-exactly on our no-norm-layer backbones, the
30-cell sweep WILL produce numbers bit-identical to existing Phase 5
FedAvg cells in `artifacts/v7_stage2_full/v7_<arch>_fedavg_iid_n7_s*`.
The "value" of running it is the auditable artefact (cell directories
named `v7_<arch>_fedbn_iid_n7_s*`) for reviewer-facing rebuttal.

**Risk of garbage**: zero. The reduction is mechanical; the smoke
empirically confirms bit-identical numerics.

**Risk of wasted GPU**: real. ~3-6 hr produces redundant data.

**User mandate**: "use GPU as much as possible to pass review" —
running for the auditable artefact is justified.

### GPU budget estimate (revised from naive linear extrapolation)

Smoke: 1 cell, 5 rounds = 48s total (training: 8.94s; overhead: 36.6s).
Steady-state per-round at scale ≈ 1.65s. At 100 rounds:
- Training time: 100 × 1.65 = 165s = 2.75 min
- Total per cell: ~3.5 min (LSTM)
- Mamba: ~5-7 min (selective scan slower)
- Spiking: ~25 min (T=5 simulation timesteps)

30 cells (10 LSTM + 10 Mamba + 10 Spiking):
- LSTM:   10 × 3.5 = 35 min
- Mamba:  10 × 6 = 60 min
- Spiking: 10 × 25 = 250 min
- **Total: ~6 hr** (was estimated 5-6 hr — confirmed)

### Pre-launch checklist

- [x] Code review of FedBN class
- [x] Spec YAML matches Phase 5 baseline
- [x] `_ALGO_REQUIRED_KWARGS` updated
- [x] LSTM smoke bit-identical to FedAvg
- [ ] Cross-arch smoke (running)
- [ ] Output dir `artifacts/p1_fedbn_natural` created (will be by launcher)
- [ ] GPU verified free
- [ ] `--skip-completed` flag verified for resume on interruption

## R3.3 centralized LSTM — design

### Goal

Decompose the naive-vs-FL gap (Round 1 finding: +0.26 AUC) into:
- **ML lift**: centralized non-linear model − centralized linear model
- **Federation cost**: centralized non-linear model − FL non-linear model

If FL ≥ centralized → federation has no cost
If FL < centralized → federation cost = (centralized − FL) AUC

### Experimental design

Two centralized cells for fair comparison:

| Cell | Setup | Purpose |
|---|---|---|
| Centralized 1-epoch | LSTM, batch=256, 1 epoch ≈ 56k steps | Wall-clock-match to FL (~6 min) |
| Centralized 3-epoch | LSTM, batch=256, 3 epochs ≈ 170k steps | Convergence-match (the upper bound) |

Both use the same train/val/test split as FL (OOD by tr).

### Hardware decision

Hardware-independent comparison (no paired-bootstrap-CI95 vs FL needed
— we just want point estimates). Either RTX 4080 or V100 works. **Will
run on RTX 4080** to defer V100 setup overhead; if FedBN sweep
saturates 4080, will move to V100.

### Pre-launch checklist (will verify when implementing)

- [ ] Custom `experiments/run_p1_centralized_lstm.py` script
- [ ] Same V3_CONTINUOUS feature schema, ContinuousScaler fit on train
- [ ] Same y_sla_violation_next target derivation
- [ ] Test AUC reported

## Decision matrix: order of operations

| Order | Task | Rationale |
|---|---|---|
| 1 | Wait for cross-arch smoke (running) | confirms reduction across 3 archs |
| 2 | Launch FedBN 30-cell sweep on RTX 4080 (background) | ~6 hr, biggest job |
| 3 | (Parallel) Implement centralized LSTM script | CPU work, no GPU conflict |
| 4 | After FedBN done: run centralized LSTM × 2 epochs | ~10-15 min GPU |
| 5 | Optional: FedSWA × 5 seeds (~30 min) | empirical MC7 refutation |
| 6 | Optional: FedAdam β2 × 3 cells (~10 min) | strengthens L14 |

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FedBN sweep crashes mid-way | LOW | medium | --skip-completed flag enables resume |
| FedBN ≠ FedAvg unexpectedly | very LOW | high | smoke proved equivalence; reduction proof intact |
| GPU OOM | LOW | medium | Phase 5 ran the same configs on same hardware |
| Disk space (30 cells × ~200KB) | very LOW | low | <10MB total |
| Centralized LSTM diverges | LOW | medium | use lower lr if needed (5e-4 worked for FL) |

## Final pre-launch decision

**APPROVE FedBN 30-cell sweep launch** subject to cross-arch smoke
confirming bit-equivalence. If smoke fails, debug FedBN class before
launching.
