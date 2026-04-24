# Project Documentation

The single entry point for project context is `CLAUDE.md` at the repo root.

## Architecture Decision Records (ADRs)

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-v5-tmc-paper-plan.md) | v5 Pipeline Extension for IEEE TMC Submission | Accepted (with revisions 2026-04-25) |

## Reading order for new contributors

1. `CLAUDE.md` (repo root) — build commands, conventions, hard rules
2. `artifacts/RESULTS_V4.md` — current benchmark results
3. `docs/ADR-001-v5-tmc-paper-plan.md` — v5 plan in detail

## ADR conventions (for this project)

Following Anthropic's Claude Code guidance on living documents rather than traditional Nygard-style immutable ADRs. Rationale: small team, fast iteration, git provides hard audit trail. Updates go in-place with a Revision History table at the top. Create a new ADR only when the current one grows past ~250 lines or the scope fundamentally changes.
