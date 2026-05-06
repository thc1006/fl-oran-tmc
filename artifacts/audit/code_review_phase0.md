# Code review of Phase 0 (post-completion self-audit)

**Date**: 2026-05-05
**Reviewer**: self
**Subject**: Phase 0 audit work (#18 ul_bler, #19 tr embedding, #20 fixture inv)
        + branch operations (PR #1 merge, rebuttal-phase1 branch creation)

## Summary verdict

**Phase 0 conclusions stand**, but **methodology was weak in 1 critical place**
(A0.2 declared `risk_confirmed` based on theoretical reasoning alone; the
decisive empirical test was deferred to P1.2-GREEN). Code review forced the
empirical test to happen now and **vindicated the conclusion**: tr embedding
rows 22-29 are byte-identical to fresh seed-0 init, exactly as the bug
mechanism predicted.

## Findings

### CR-1 (HIGH severity, RESOLVED): A0.2 audit lacked decisive empirical proof

**What I claimed in A0.2**: `risk_confirmed`, based on PyTorch `nn.Embedding`
gradient semantics + train/test tr range disjointness.

**Why this was insufficient**: theoretical reasoning + empirical test on
checkpoint should be the discipline; I only did the former. A subsequent
per-row L2 norm inspection during code review was inconclusive (norms all
in N(0,1) magnitude band).

**Decisive test (run during code review)**:
```
mean ||trained - fresh_seed0_init|| per row band:
  TRAIN  rows  0-21: 3.0e-01  ← real training drift
  VAL    rows 22-24: 4.0e-08  ← float32 round-off only
  TEST   rows 25-27: 7.5e-08  ← float32 round-off only
  UNUSED rows 28-29: 3.0e-08  ← float32 round-off only
```

**6+ orders of magnitude separation**. Bug confirmed exactly as A0.2
hypothesized.

**Fix applied**:
- `artifacts/audit/tr_embedding_audit.md` amended with "Empirical
  verification" section
- `tests/test_audit_invariants.py` added with 5 regression tests, including
  decisive `test_tr_embedding_rows_22_to_29_byte_identical_to_fresh_init`
- pytest now 519 GREEN (was 514)

**Methodology lesson for future audits**: when declaring `risk_confirmed`,
attach 1 empirical sanity check, not just theoretical chain.

### CR-2 (MEDIUM severity, FIXED): A0.1 audit doc imprecise about feature count

**What I wrote**: "ul_bler IS in 17 continuous features (features.py L31, L60)".

**What's actually true**:
- `CLEAN_FEATURES` (data_v2/features.py:22-36) defines **19** entries (17 raw
  + 2 trend features `tx_brate_dl_roll3` + `tx_brate_dl_volatility`)
- `V3_CONTINUOUS` (training/centralized_v3.py:42) defines **17** entries
  (the 2 trend features are dropped before model input)
- Paper §3 claim "17 continuous features" matches `V3_CONTINUOUS`, not
  `CLEAN_FEATURES`

Not a bug, but my doc was sloppy.

**Fix applied**: `artifacts/audit/ul_bler_audit.md` amended with explicit
distinction between `CLEAN_FEATURES` (19 entries, dataframe-level filter)
and `V3_CONTINUOUS` (17 entries, model-level input). Verified count via
`test_v3_continuous_has_17_features`.

### CR-3 (METHODOLOGY DEBT): Phase 0 lacked an "empirical confirmation" rubric

The `risk_confirmed | risk_cleared | requires_redesign` outcome label
framework I added per Flaw 3 in the prior re-audit pass should be extended:
**`risk_confirmed` MUST include 1 empirical sanity check before label
assignment**. Theoretical chain alone is insufficient for irreversible
downstream actions (e.g., 5.7 hr GPU on FedBN cells).

**Action**: Update the audit playbook in
`artifacts/audit/AUDIT_PLAYBOOK.md` (to be created in next pass) with
this rule.

### CR-4 (LOW): tr → embedding index has no remap (verified)

`grep -nE "tr.*encoder|tr.*lookup|tr.*remap|tr.*to_int|map.*tr|encode.*tr|categorical_to_int"` returned empty for `data_v2/` and `training/`. tr value
is used directly as embedding index. No OOV bucket. Bug pattern confirmed
as the simple "PyTorch nn.Embedding only updates indexed rows" type.

### CR-5 (LOW, hygiene): Untracked figure files + non-deterministic PDF generation

- 21 untracked artefact files (pre-existing v7_* cell directories that
  were never tracked through any S11 work — not new debris)
- 6 modified figure files (`artifacts/figures/*.{pdf,svg}`) showing
  modified status: matplotlib's `savefig()` writes a creation timestamp
  in PDF metadata, so re-running `phase5_paper_figures.py` produces
  bit-different output despite identical inputs. **Reproducibility hole**.

**Action**: defer to camera-ready cleanup. Either pin matplotlib's
`metadata={'CreationDate': None}` in `savefig()` calls, OR document
non-determinism in `paper/README.md`.

### CR-6 (LOW, hygiene): Local branches need cleanup

- `feat/phase-5-6-implementation` is dead post-merge (commits all in main)
  but exists locally; safe to delete with `git branch -d feat/...`.
- Local `main` is 43 commits behind `origin/main`; needs `git pull` if
  switching back. Doesn't block since work is on `rebuttal-phase1`.

**Action**: optional, defer until after Phase 1 lands.

### CR-7 (LOW): No PR for rebuttal-phase1

Branch pushed but no PR opened. For visibility + future CI gating,
opening a draft PR is good practice.

**Action**: optional. Open when first P1 RED test commits land.

## Self-criticism

The most embarrassing finding is **CR-3 (methodology debt)**: I declared
`risk_confirmed` as the audit outcome without running a decisive empirical
test, even though:

1. The `feedback_audit_before_launch` memory entry explicitly mandates
   audit-before-action discipline (60-cell crash precedent)
2. The empirical test is a 5-minute Python script using existing artefacts
3. I had time during Phase 0 to run it

I rushed to declare the audit "complete" instead of completing it properly.
This code review caught it; the decisive test now exists in the regression
suite (`test_tr_embedding_rows_22_to_29_byte_identical_to_fresh_init`) so
the same shortcut can't happen for this specific finding again.

## What this means for Phase 1

**No Phase 1 task design needs to change**. The audit conclusions are
correct; my work on them was just methodologically thinner than it should
have been. P1.1-RED, P1.2-RED, P1.3-RED, P1.4-RED can all proceed as
planned.

**One additional acceptance criterion** for Phase 1 GREEN tasks: each
"experiment confirms hypothesis" finding should be backed by a regression
test in `tests/test_audit_invariants.py` (or similar) so future code
changes can't silently invalidate the finding.
