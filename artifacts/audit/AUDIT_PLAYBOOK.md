# Audit Playbook

Discipline rules for `artifacts/audit/` Phase-N audit work, derived from
the Phase 0 code-review post-mortem on 2026-05-05 (`code_review_phase0.md`,
finding CR-3) and `feedback_audit_before_launch` memory entry.

## When an audit is required

Before any of the following actions:

1. Launching ≥1 hr of GPU work
2. Adding ≥10 cells to a sweep YAML
3. Asserting a claim that will appear in paper text or commit message
4. Marking a `risk_*` outcome label on a Phase-N audit task

## Outcome label rubric

Audit tasks emit one of:

| Label | Meaning | Required evidence |
|---|---|---|
| `risk_confirmed` | The hypothesised risk is real | **Theoretical chain + 1 empirical check** (see below) |
| `risk_cleared` | The hypothesised risk does not apply | **Theoretical chain + 1 empirical check** |
| `requires_redesign` | Audit revealed a different problem | Document the new problem; halt downstream tasks |

### Empirical check requirement

The form of the empirical check depends on the claim type:

| Claim type | Acceptable empirical check |
|---|---|
| **Code-presence** ("X is in feature list") | Static check: `grep` + `len()` assertion + cite file:line |
| **Behavioural** ("rows of X never receive gradient") | **Runtime check**: load actual artefact (checkpoint, dataframe, log), inspect numerical state, demonstrate the prediction holds bit-exactly modulo float32 round-off |
| **Statistical** ("AUC of X < AUC of Y") | Run experiment with prior-belief preregistered in `experiments/preregistered/`; report measured value + CI |

**Theoretical reasoning alone never qualifies as `risk_confirmed`/`cleared`.**
This is the rule that Phase 0's A0.2 violated (declared `risk_confirmed`
without runtime check; code review forced the check, vindicated the
conclusion 6+ orders of magnitude — but the discipline gap was real).

## Required artefacts per audit

Every `risk_*` label must produce, in `artifacts/audit/`:

1. `<topic>_audit.md` with sections:
   - **Outcome label**
   - **Evidence chain** (file:line citations)
   - **Empirical verification** (the numerical demonstration; tabulate inputs and outputs)
   - **Action** (downstream task implications)

2. A regression test in `tests/test_audit_invariants.py` (or sibling) that
   re-runs the empirical check on every commit. Test must `pytest.skip` if
   external artefact (e.g., trained checkpoint) is missing, never silently
   pass. The test name must reference the audit topic so that a failure
   triggers `audit/<topic>.md` re-derivation.

3. Memory entry update (`memory/paper_split_status.md` or equivalent)
   with: outcome label + 1-line empirical-check headline.

## Anti-patterns to refuse

- **Defer-as-fix**: claiming an audit issue is "fixed" by adding "TODO:
  audit playbook" to a doc, without writing the playbook. (CR-3 violated
  this; this playbook is the genuine fix.)
- **Norm-only checks for trained-vs-untrained embedding distinction**: per-
  row L2 norm cannot distinguish trained from untrained rows because
  trained values can naturally fall in the N(0,1) magnitude band. Always
  use `||trained - fresh_seed_X_init||` direct delta.
- **Single-checkpoint regression tests for mechanism claims**: if the
  claim is mechanism-level ("PyTorch nn.Embedding only updates indexed
  rows"), parameterize the test over multiple architectures using the same
  mechanism. One-arch tests pass for the wrong reasons sometimes.

## Where this playbook is enforced

- Code review (this document is the rubric)
- `tests/test_audit_invariants.py` (the empirical checks themselves)
- Pre-launch audit task templates (P1.3-AUDIT, future P2.X-AUDIT, etc.)
