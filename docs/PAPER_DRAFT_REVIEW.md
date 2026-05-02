# PAPER_DRAFT.md — Three-persona reviewer-style review

**Reviewer:** Self-review by Claude (acting as 3 personas)
**Date:** 2026-05-02
**Draft:** PAPER_DRAFT.md v3 (288 lines)

---

## Persona 1 — TMC Editor (desk-review screening)

### Verdict
The paper has a credible 4-axis empirical contribution, but **submitted as-is would be desk-rejected** by IEEE TMC because of missing required sections, not because of contribution weakness.

### Blocking issues (must fix before submission)

* **E1 — §5 reproducibility section is a one-line stub.** Contribution 5 promises "Open reproducible reference benchmark" but §5 says only "(deferred to the camera-ready archive)". TMC editors expect §reproducibility complete on submission. **Fix:** at minimum, populate §5 with bullet structure (Croissant fields, Dockerfile recipe path, demo-notebook description) — full release artefact can wait until acceptance.
* **E3 — Status block (lines 3-7) is a draft tracking log, not paper content.** Lines saying "DRAFT v3", "v3 changes from v2 (2026-05-02): rebuilt §1 hook + ..." reveal active drafting. Reviewer would see this. **Fix:** remove for submission; keep in commit history.
* **E4 — No Abstract.** TMC requires 150-250-word abstract immediately after title. Currently goes title → status → §1.
* **E5 — Author list / affiliations / ORCID / contact missing.** Even for double-blind, TMC needs the metadata block.
* **E6 — No Keywords list.** TMC asks for 5-8 IEEE keywords below abstract.
* **E8 — Internal TODO references at line 288.** "§5 Reproducibility ... is task #146; ... is task #147" — internal task tracking that shouldn't ship.

### High-impact concerns

* **E2 — Title length.** "Federated O-RAN Slice SLA Prediction Across Architectures and Heterogeneity Regimes: A Cross-Architecture Empirical Benchmark on Colosseum/ColO-RAN" — 21 words, two layers. Trim to one of:
  * "Federated O-RAN Slice SLA Prediction: A Cross-Architecture Empirical Benchmark on Colosseum/ColO-RAN"
  * "An Empirical Benchmark of Federated Learning for O-RAN Slice SLA Prediction"
* **E7 — §1 hook second paragraph (line 14) is a 5-sentence ~600-word run-on.** Editor's first impression: wall of text. Split into 2 paragraphs (one for the inversion finding, one for the random_split ablation summary).

---

## Persona 2 — O-RAN Domain Expert Reviewer

### Verdict
Strong empirical benchmark, but several O-RAN-architectural / dataset-semantic claims need precision tightening.

### Concrete corrections needed

* **D1 — "Use case" terminology.** §1 line 12 calls "slice-level SLA violation prediction" a "standardized use case" — in O-RAN ATIS/WG2 terminology, "use case" usually refers to network-level scenarios (eMBB/URLLC/mMTC), not specific xApp implementations. Re-cast as "near-RT RIC xApp pattern" or "rApp use case" with citation to the specific O-RAN WG document.
* **D2 — eMBB/URLLC/mMTC vs slice_id 0/1/2 mapping is missing.** Paper writes "heterogeneous traffic slices (eMBB, URLLC, mMTC)" but never maps slice_id 0/1/2 to these labels. ColO-RAN release should document this; cite or footnote the mapping.
* **D3 — Aggregator placement claim.** §1 line 12 says "non-RT RIC / SMO that hosts the federation aggregator". The aggregator could be an xApp (near-RT RIC), an rApp (non-RT RIC), or an SMO function — paper picks one without citing which O-RAN WG spec. **Fix:** add 1 line citing the O-RAN WG2 / WG10 spec the design follows, or soften to "the orchestration layer (rApp or SMO function depending on deployment)".
* **D4 — Scheduler policy names missing.** §3.1 says "3 admission-control / scheduler policies" but doesn't name them (round-robin / proportional-fair / waterfilling? slot-allocation policies?). Domain reviewer wants the names.
* **D5 — Why is "natural-by-BS" the deployment-realistic partition?** In real O-RAN deployment, FL aggregation could pool clients regionally (one xApp serves N BS). Paper assumes one-client-per-BS without engaging with alternatives.
* **D6 — Are dl_cqi / dl_mcs differences "structural" or "snapshot variance"?** §7.1.2 reports per-bs CQI/MCS KL=0.475 / 0.267. But these capture momentary channel state. Over 22 train tr configurations the CQI distribution might converge across BS. Need clarify whether the per-bs KL is averaged over the 22 configs or measured per-config-per-bs.
* **D7 — Spiking-SSM as O-RAN deployment candidate is overclaimed.** §9 line 282 lists Spiking-SSM as "candidate for sparsity-aware accelerators". No actual O-RAN deployment plans for Spiking exist. Soften to "research direction" or "open question for future O-RAN hardware".
* **D8 — Future-work testbed extensions overstated.** §8 L4 says "OpenAirInterface 5G or srsRAN would let future work probe the cross-device limit". OAI 5G doesn't easily produce ColO-RAN-format telemetry; srsRAN lacks integrated FL training. Either qualify ("subject to telemetry-format engineering") or remove.

---

## Persona 3 — FL / ML Methodology Reviewer

### Verdict
Methodologically careful (paired-bootstrap, BLAKE2b decorrelation, 13-item §8) but several technical-rigour gaps remain.

### Statistical / methodology concerns

* **M2 — Bonferroni mentioned but never applied in §6 findings.** §4.5 discusses family-wise threshold (0.05/180 ≈ 2.78e-4) but no §6.X finding reports a Bonferroni-corrected p-value. **Fix:** for each main-text claim, report uncorrected and Bonferroni-adjusted p side-by-side, OR remove Bonferroni from §4.5 entirely as not actually applied.
* **M5 — Paired comparison N=5 (V100) vs N=10 (4080) seed counts.** §7.1.1 reports random_split N=5 seeds vs natural-by-BS N=10 seeds. **The Δ in the table is mean-vs-mean (unpaired), not paired-bootstrap-CI95.** This is a real protocol gap; need explicit caveat in §7.1.1: "Δ here is mean − mean (unpaired across V100 / 4080 seed sets). Strict paired comparison requires running random_split with the same 10 seeds as Phase 5; we used 5 seeds for V100-time efficiency. The 25× SNR margin makes this distinction non-blocking but should be noted."
* **M6 — Random_split per-seed std combines partition variance + init variance.** §7.1.1 reports random_split std (0.0027 LSTM) vs natural-by-BS std (0.0004 LSTM). The two std's are not on the same metric: random_split's std includes both partition shuffle variance + init RNG variance; natural-by-BS only the latter. **Add explicit caveat** that std comparison is not strictly fair (mean comparison still valid).
* **M9 — §7.1.1 confounds candidate (i) and (ii) elimination.** §6.6 lists 3 mechanism candidates; §7.1.1 random_split breaks both bs grouping AND slice grouping. Result is consistent with EITHER candidate (i) "bs-conditioned signal preservation" OR candidate (ii) "Dirichlet's per-slice row redistribution effect". Paper claims (i) confirmed; technically only (i)-or-(ii) confirmed. **Fix:** add 1 sentence acknowledging this conjunction; full disambiguation needs an additional ablation that breaks slice grouping but preserves bs grouping (e.g., per-bs Dirichlet over within-bs slices).
* **M10 — FedBN benchmark gap is the biggest reviewer attack surface.** §2.5 says "arxiv 2508.08479 demonstrated FedBN's superiority on cellular feature-skew"; §8 L2 admits "we similarly do not benchmark FedBN". For contribution 2's "algorithm-flatness" claim, FedBN is the algorithm most likely to break it (its strength is exactly feature skew, which is what we have). **Strongest mitigation:** run FedBN as a 6th algorithm (1 hr GPU on V100, similar to T-ABLATION). Alternative: tighten §8 L2 to "FedBN benchmarking is the highest-priority algorithm gap; we estimate it would extend rather than overturn contribution 2 because (a) Phase 5's IID column already shows tight ordering of all 5 algorithms within 0.005 AUC, so a 6th method's deviation from FedAdam would need to exceed ~0.01 AUC to change the ranking, (b) arxiv 2508.08479 reports FedBN's gain over FedAvg is in the 0.01-0.03 range on similar feature-skew tasks". This anticipated-result framing is acceptable as a §8 limitation.
* **M4 — "FedAdam saturates the headroom available to ANY server-side aggregation modification" is overclaim.** This generalises a 5-algorithm experiment to a hypothesis class. **Fix:** "FedAdam achieves the empirical maximum +0.006-0.016 AUC over FedAvg in our 5-algorithm baseline; we expect (without proving) that LookAhead-style overshoot cannot systematically exceed this ceiling because of the variance-estimation mechanism gap (§2.6)".
* **M3 — §6.4 Mamba×SCAFFOLD per-seed sign count missing.** Reviewer wants "9 of 10 seeds had Δ < 0" type evidence to substantiate "consistent direction" beyond just std and mean. **Fix:** report per-seed sign count.
* **M7 — FedAdam β₂=0.99 sensitivity untested.** Currently a 1-line caveat. Reviewer might ask: would β₂=0.999 (canonical Adam) change the FedAdam-saturates conclusion? **Fix:** add §8 L14 "FedAdam hyperparameter sensitivity: β₁/β₂/server_lr were preregistered to FedAdam paper values; a sensitivity sweep is candidate future work — the current FedAdam-vs-FedAvg deltas are consistent with paper-canonical hyperparameter choices on a different task."
* **M8 — §6.5 "10× energy span" claim is RTX 4080-specific (already in §8 L1).** §1 contribution 4 should add hardware-scope qualifier: "10× span on RTX 4080 (§7.4 + §8 L1 discuss hardware sensitivity)". Currently contribution 4 reads as architecture-fundamental.
* **M11 — `mode="iid"` misnomer.** §4.3 has 1-line disclosure. For reproducibility-paper standards, this needs more prominence (footnote or explicit "Note:" block) so future researchers running our code don't get confused.

### Statistical methodology — minor wording

* **M1 — §4.5 is a single 12-line wall-of-text.** Convert to a numbered list:
  1. Per-cell summary stat: mean ± std test AUC over 10 seeds
  2. Pairwise stat: paired-bootstrap CI95 (n_boot = 10 000, percentile interval; rationale: ...)
  3. Decorrelation: BLAKE2b-derived seed offset per (a, b) pair
  4. Pair-set base seeds: 2026 (algo-pairs), 2027 (arch-pairs)
  5. Secondary check: Wilcoxon signed-rank
  6. Multi-comparison: uncorrected per finding; Bonferroni thresholds noted in supplementary

---

## Cross-persona meta-issues

* **X1 — No §5 reproducibility content.** All 3 personas flag this.
* **X2 — No Abstract.** Editor + Method.
* **X3 — No author info.** Editor.
* **X4 — Section gap (§4 → §6 directly).** Editor + reproducibility expectations. Either insert §5 placeholder or renumber.
* **X5 — No References / Bibliography section.** Editor + Method. Inline `[Author Year]` is consistent but separate references list is required.
* **X6 — No formal Figure captions.** Editor + Method.
* **X7 — No formal Table numbering.** Method reviewer expects Table 1-N.
* **X8 — Future work scattered, no consolidated §9.X subsection.**

---

## Priority for fixes (post-T-L)

### P0 — desk-reject blockers (must fix before submission)
1. E3 — Remove status block (lines 3-7) for submission build
2. E4 — Add Abstract (~200 words)
3. E5 — Add author block + ORCID + email
4. E6 — Add Keywords (5-8)
5. E8 — Remove line-288 task-tracking comment
6. X1/X4 — Add §5 reproducibility (at minimum: bullet placeholder for Croissant + Dockerfile + demo notebook)
7. X3 — Author info (covered by E5)
8. X5 — References section
9. X6 — Figure captions

### P1 — strong reviewer-rebuttal triggers (should fix or convince yourself the limitation entry is enough)
10. M5 — §7.1.1 paired-vs-unpaired comparison disclosure
11. M6 — §7.1.1 std confound caveat
12. M9 — §6.6/§7.1 candidate (i)+(ii) confound disclosure
13. M10 — §8 L2 FedBN extension argument (or run FedBN benchmark)
14. M2 — Bonferroni: apply or remove
15. M4 — §6.3 / §1 contribution 2 "saturates" → "achieves empirical maximum + cannot systematically exceed"
16. M8 — §1 contribution 4 hardware-scope qualifier
17. E2 — Title trim
18. E7 — §1 second paragraph split

### P2 — polish (after P0/P1)
19. M1 — §4.5 list format
20. M3 — §6.4 per-seed sign count
21. M7 — §8 L14 FedAdam β₂ sensitivity caveat
22. M11 — `mode="iid"` misnomer footnote
23. D1-D8 — domain-precision fixes
24. X8 — Future work consolidation

---

## Recommendation

**Do not submit as-is.** The desk-reject blockers (P0 items 1-9) are simple to fix (~3-4 hr). After P0, the paper is submission-ready; P1 fixes can be done in parallel with reviewer-objection rebuttals during peer review.

**T-L #165 verdict:** P0 blockers identified. Mark T-L done; spawn P0 fixes as a new task (call it T-M SUBMISSION-PREP) before final submission build.
