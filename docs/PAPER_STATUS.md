# PAPER_DRAFT.md — Status / Changelog

> Internal draft-tracking log. NOT in submission build.
> Stripped from PAPER_DRAFT.md by T-M (#168) for desk-reject prevention.
> Update entries here when paper revisions land.

---

## 2026-05-02 — DRAFT v3 (§1 - §9 — §5 reproducibility deferred to camera-ready / supplementary archive)

v3 changes (2026-05-02) after Step 1 / Step 2 dataset measurement + V100 random_split ablation:

* Rebuilt §1 hook + §1 contributions + §6.6 + §7.1 around the *empirical observation + tested hypothesis* framing (replacing the invalidated per-client-pos_weight specialist mechanism)
* Added §7.1.1 random_split V100 ablation (15 cells)
* Added §7.1.2 per-bs continuous-feature KL measurements (channel-state features dl_cqi, dl_mcs dominate)
* Added §7.1.4 V100-vs-4080 hardware drift caveat
* Added §8 L9-L13 reviewer-objection responses (pos_weight_split, bf16, round/step split, threshold sensitivity, percentile-vs-BCa)
* Replaced 6D framing with 4-axis (the actual sweep dimensionality)
* Fixed dataset cardinality (3 slices, 3 schedulers, 17 continuous features, bs_id ∈ {1..7}, 30.9 % positive rate, all from Step 1 measurement vs prior placeholder values)
* License MIT → Apache-2.0
* RTX 4080 12 nm/120W → TSMC 4N/320W
* Introduced α_dir / α_dyn / α_LA notation disambiguation

## 2026-05-01 — DRAFT v2 (§1 + §2 only, with placeholder mechanism story now invalidated)

## 2026-04-27 — DRAFT v1

## Target venue

* IEEE Transactions on Mobile Computing (JIF 7.6) — primary
* IEEE Open Journal of the Communications Society (JIF 6.1) — fallback
