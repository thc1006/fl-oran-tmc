# R2 GPU experiment design — hardware-aware plan

**Date:** 2026-05-07
**Branch:** `reviewer-r2-fixes`
**Status:** Plan only — Phase 2 infrastructure ready, GPU launches paused for user confirmation per CLAUDE.md rule #6.

This document captures the hardware-aware design choices for the three R2 GPU experiments (C1, C3, C4) so the launcher choice is auditable and the user has a single source of truth for "which GPU runs what, why".

---

## Hardware inventory

| GPU | Arch | VRAM | TGP | BF16 | NVLink | Cards | Access |
|-----|------|------|-----|------|--------|-------|--------|
| RTX 4080 | Ada Lovelace (sm_89) | 16 GiB | 320 W | native | no | 1 | local |
| Tesla V100-SXM2 | Volta (sm_70) | 32 GiB | 300 W | emulated (slow) | yes (NVSwitch) | 4 | `ssh -p {51419,50800} leo07010@203.145.216.194` |

**Key implications:**
- **RTX 4080**: faster per-card (Phase 5 ~6 min/cell), but single GPU → no embarrassingly-parallel speedup. Native BF16 → keep `mixed_precision: bf16`.
- **V100**: ~2× slower per card on BF16-bottlenecked ops (because BF16 is software-emulated), but 4× cards + 32 GiB each → can parallelise multiple cells / oversubscribe smaller jobs. **Use `mixed_precision: fp16` on V100** (FP16 is native; loses ~0.5% AUC mid-training, recovers at eval).
- **NVLink**: matters for true multi-GPU data-parallel within one training run; for embarrassingly-parallel "one cell per card" sweeps it is irrelevant.

---

## Per-experiment GPU assignment

### C1 — Same-step centralized LSTM (addresses A1 root)

| Quantity | Value |
|----------|-------|
| N_cells | 5 (5 seeds) |
| Per-cell wall (RTX 4080, bf16) | ~6 min |
| Per-cell wall (V100, fp16) | ~12 min |
| Per-cell VRAM | ~3 GiB |
| RTX 4080 sequential wall | **~30 min** |
| V100 4-way parallel wall | ~24 min (1 batch of 4 + 1 sequential) |

**Decision: RTX 4080 sequential.** Speedup from V100 is marginal (24 vs 30 min) and incurs SSH/upload overhead. Single GPU keeps the experiment trivially reproducible.

**Spec: `experiments/specs/r2_same_step_centralized.yaml`**
- arch: lstm
- mode: centralized (`run_p1_centralized_lstm.py` with `--max-steps 25000`)
- seeds: [0, 1, 2, 3, 42] (matching Phase 5 seed coverage)
- max_steps: 25000 (= FL 100 rounds × 50 max_steps × 5 sampled clients)
- batch_size: 64
- mixed_precision: bf16
- device: cuda:0
- cudnn_deterministic: true

**Expected outcome envelope:**
- If centralized@25k AUC < 0.9159 (FL): A1 reframes to "FL outperforms centralized at same compute" — strongly positive for the paper
- If centralized@25k AUC ≈ 0.9159: A1 reframes to "FL ≈ centralized at same compute" — neutral, defensible
- If centralized@25k AUC > 0.9311 + ε: surprising; would require investigation

---

### C3 — Post-hoc per-BS fine-tune (addresses A2 FedBN-spirit gap)

| Quantity | Value |
|----------|-------|
| N_cells | 105 (7 BS × 3 archs × 5 seeds) |
| Per-cell wall (RTX 4080, bf16) | ~2 min (small fine-tune job, mostly data loading) |
| Per-cell wall (V100, fp16) | ~3 min |
| Per-cell VRAM | ~5 GiB (fits ≥4 cells/card on V100) |
| RTX 4080 sequential wall | **~3.5 hr** |
| V100 4-way oversubscribed (4 cells/card) wall | ~30-40 min (7 batches × 3 min, with stream contention) |

**Decision: V100 cluster, 4 cards × 4 concurrent cells per card.** Cuts wall time ~70 %. Worth the SSH/checkpoint upload (~3 MiB) overhead since 105 cells is significant.

**Implementation:** `scripts/v100_r2_c3_launcher.sh` reads cell list, splits into 4 GPU groups, submits each as background job with up to 4 concurrent processes per GPU (CUDA_VISIBLE_DEVICES + GPU_NUM env var). Each process writes a per-cell JSON; driver script aggregates.

**Spec: `experiments/run_r2_post_hoc_per_bs_finetune.py`**
- CLI: `--cells "lstm:bs1:s0,lstm:bs1:s1,..." --device cuda:N --finetune-steps 200 --batch-size 64 --mixed-precision fp16`
- Loads `artifacts/v7_stage2_full/v7_<arch>_fedavg_iid_n7_s<seed>/best.pt`
- Per BS: subsets train data to that BS only; fine-tunes K=200 Adam steps with same hyperparameters as Phase 5 client step
- Eval: per-BS test split; reports personalised vs global per-cell AUC + per-BS Δ

**Expected outcome envelope:**
- If mean Δ_personalised < +0.005 AUC: confirms our "feature-shift personalisation gives little" claim → **strengthens** §8 L2 against MC3
- If mean Δ_personalised > +0.01 AUC: weakens our story; need to add "personalisation does help; we choose global for deployment simplicity" framing
- If mean Δ_personalised < 0 (negative): even stronger — global model is best, personalisation harms

---

### C4 — No-`tr` ablation (addresses A3 long-version)

| Quantity | Value |
|----------|-------|
| N_cells | 10 (LSTM × FedAvg × natural-by-BS × no-tr × 10 seeds) |
| Per-cell wall (RTX 4080, bf16) | ~6 min |
| Per-cell wall (V100, fp16) | ~10 min |
| Per-cell VRAM | ~8 GiB (3 cells/card on V100) |
| RTX 4080 sequential wall | **~60 min** |
| V100 4-way (3 cells/card) wall | ~10 min (1 batch of 10 → 4+4+2 distribution × 10 min/cell ≈ 10 min wall, dominated by longest) |

**Decision: V100 if cluster available; else RTX 4080.** V100 saves ~50 min wall; RTX 4080 is the local-only fallback.

**Spec: `experiments/specs/r2_no_tr_ablation.yaml`**
- arch: lstm
- algo: fedavg
- partition: natural (mode=iid, n_clients=7)
- seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
- num_rounds: 100
- arch_overrides.lstm.drop_categorical: ["tr"]
- mixed_precision: fp16 (V100) / bf16 (RTX 4080) — spec-overridable
- device: cuda

**Model patch:** `forecaster_v2.py` / `lstm_multi.py` accept new `drop_categorical: list[str]` arg in their `__init__`. When non-empty, the categorical encoder skips lookup for the listed features and concatenates only the remaining cat embeddings + continuous features.

**Expected outcome envelope:**
- If natural-by-BS gap shrinks <10% (no-tr vs with-tr): tr is mostly cosmetic → **strengthens** §7.1.6, C1 mechanism finding robust
- If gap shrinks 10-30%: matches the §7.1.6 quantification (≤10% bug, ≥90% structural) — defensible
- If gap shrinks >50%: tr is doing real work → §7.1.6 needs revision; tells us C1 mechanism is partly tr-embedding-driven (would need cover-letter explanation)

---

## Total GPU budget (worst case all on RTX 4080)

| Experiment | RTX 4080 wall | V100 wall | Recommended |
|-----------|---------------|-----------|-------------|
| C1 | 30 min | 24 min | **RTX 4080** |
| C3 | 3.5 hr | 30-40 min | **V100** |
| C4 | 60 min | 10 min | **V100** (else RTX 4080) |

**Recommended split:**
- C1 on RTX 4080 (30 min, locally launched)
- C3 + C4 on V100 cluster, run in parallel (different cards groups), wall ~40 min
- **Total wall: ~40 min** if V100 available; **5 hr** if all on RTX 4080

---

## Precision / determinism knobs per GPU

| Knob | RTX 4080 | V100 |
|------|----------|------|
| `mixed_precision` | `bf16` (native) | `fp16` (native) — DO NOT use bf16 (emulated, slow) |
| `cudnn_deterministic` | `true` (~10% slowdown) | `true` (~20% slowdown) — keep for reproducibility |
| `torch.set_float32_matmul_precision` | `"high"` (TF32) | `"high"` (FP32 with TF32) |
| `torch.compile` | enabled (per-cell cache leak fix in `run_v7_sweep`) | enabled (same fix) |
| GradScaler | not needed (bf16) | **required** (fp16 underflow on small grads) |

---

## Rollback plan if V100 cluster unavailable

If V100 SSH (port 51419 / 50800) is unreachable on the day of launch, fall back to RTX 4080 sequential for all three:
- C1: 30 min
- C3: 3.5 hr (105 cells × ~2 min)
- C4: 60 min
- Total: ~5 hr wall, single GPU saturated

This is acceptable — not ideal, but no work blocked.

---

## Provenance

Design choices grounded in:
- `memory/v100_cluster_access.md` — V100 cluster credentials + ports
- `memory/project_v5_state.md` — Phase 5 per-cell timing on RTX 4080
- `CLAUDE.md` — torch 2.10 / CUDA 12.8 / RTX 4080 (sm_89) environment
- `CLAUDE.md` — rule #6 "Do not run training without asking"
- Phase 5 audit: per-cell VRAM ~10 GiB max (Spiking × bs=64); LSTM ~3 GiB
