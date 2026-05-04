# Project Documentation

The single entry point for project context is `CLAUDE.md` at the repo root.

## Architecture Decision Records (ADRs)

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-v5-tmc-paper-plan.md) | v5 Pipeline Extension (Stage 1 + Stage 2 path) | Accepted (with revisions 2026-04-25) |
| [ADR-002](ADR-002-phase6-fedswa.md) | Phase 6: FedSWA Integration | **REJECTED** v3 (mechanism-based dismissal in §related-work; see ADR for rationale) |

## Current paper docs (JSAC submission target)

| File | Role |
|---|---|
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | JSAC main paper (Stage 2 federated extension) |
| [PAPER_SUPPLEMENTARY.md](PAPER_SUPPLEMENTARY.md) | Supplementary material (App. A–D) |
| [PAPER_V6_STAGE1.md](PAPER_V6_STAGE1.md) | Stage 1 short paper (in preparation; centralized 3-arch benchmark) |
| [PAPER_CONTRIBUTION_CLAIM.md](PAPER_CONTRIBUTION_CLAIM.md) | §1 contribution audit log |

## Current results docs

| File | Role |
|---|---|
| [RESULTS_V7_PHASE5.md](RESULTS_V7_PHASE5.md) | V7 Phase 5 sweep results (referenced by §6 of paper) |
| [RESULTS_V6_STAGE1.md](RESULTS_V6_STAGE1.md) | V6 Stage 1 sweep results (Stage 1 paper data) |
| [RESULTS_V6_STAGE1_ANALYSIS.md](RESULTS_V6_STAGE1_ANALYSIS.md) | V6 Stage 1 analysis (deeper interpretation) |
| [RESULTS_V5_FINAL.md](RESULTS_V5_FINAL.md) / [RESULTS_V5_PRELIM.md](RESULTS_V5_PRELIM.md) | V5 results (pre-Stage 1 baseline reference) |

## Future work + research notes

| File | Role |
|---|---|
| [FUTURE_STUDY.md](FUTURE_STUDY.md) | Long-term research roadmap |
| [FUTURE_WORK_RESEARCH.md](FUTURE_WORK_RESEARCH.md) | Specific candidate research directions |

## Archived docs (superseded; preserved for git history reference)

`docs/archive/` contains review notes / status snapshots / implementation plans that have been incorporated into current docs and are kept only for traceability. Each archived file carries a frontmatter block (`status: superseded`, `as-of:`, `see:`, `reason:`) pointing to its successor.

| Archived file | Successor / pointer |
|---|---|
| [archive/PAPER_DRAFT_REVIEW.md](archive/PAPER_DRAFT_REVIEW.md) | `docs/PAPER_DRAFT.md` (review notes incorporated) |
| [archive/PAPER_STATUS.md](archive/PAPER_STATUS.md) | `memory/paper_split_status.md` (live status) |
| [archive/T_G_references_audit.md](archive/T_G_references_audit.md) | `docs/PAPER_DRAFT.md §References` |
| [archive/T_ABLATION_RESULTS.md](archive/T_ABLATION_RESULTS.md) | `docs/PAPER_DRAFT.md §7.1.1` + supp App. A.1/A.2 |
| [archive/TOMORROW_NVML_FEDDYN_PLAN.md](archive/TOMORROW_NVML_FEDDYN_PLAN.md) | `docs/ADR-001-v5-tmc-paper-plan.md D-22` |

## Reading order for new contributors

1. `CLAUDE.md` (repo root) — build commands, conventions, hard rules
2. `docs/PAPER_DRAFT.md` — current paper (JSAC target)
3. `docs/ADR-001-v5-tmc-paper-plan.md` — design decisions backing the implementation
4. `docs/RESULTS_V7_PHASE5.md` — current benchmark headline numbers

## ADR conventions (for this project)

Following Anthropic's Claude Code guidance on living documents rather than traditional Nygard-style immutable ADRs. Rationale: small team, fast iteration, git provides hard audit trail. Updates go in-place with a Revision History table at the top. Create a new ADR only when the current one grows past ~250 lines or the scope fundamentally changes.
