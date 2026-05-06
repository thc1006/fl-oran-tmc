"""R2 reviewer feedback — paper-language invariants.

This file pins the language-tone fixes from the second-round reviewer
review (post merge of PR #11 / commit bfa2641). Each invariant either:
  * forbids a phrase the R2 reviewer flagged as too strong / contradictory
  * requires a replacement phrase that scope-limits the original claim

Tests are RED before the GREEN edits land. After GREEN, they prevent
regression — any future edit that re-introduces the over-claim will
break CI.

Reviewer-finding mapping
------------------------
- A1   (federation cost) → ``test_no_equivalent_training_budget_phrase``
                          + ``test_centralized_reference_gap_present``
- B1   (cannot be centrally pooled) → ``test_no_cannot_be_centrally_pooled``
- B2   (inverting standard assumption) → ``test_no_inverting_standard_fl_assumption``
- B3   (FedAdam-saturated headroom bounding) →
                          ``test_no_fedadam_saturated_bounding_claim``
- B4   ("predictive ceiling empirically validated") →
                          ``test_no_validated_predictive_ceiling_claim``
                          + ``test_magnitude_survived_direction_did_not_present``
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DRAFT = REPO_ROOT / "docs" / "PAPER_DRAFT.md"
MAIN_TEX = REPO_ROOT / "paper" / "main.tex"


def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    return path.read_text(encoding="utf-8")


# ---------- A1: federation cost framing ----------


def test_no_equivalent_training_budget_phrase() -> None:
    """A1: §6.7 must NOT call centralized vs FL gap a 'federation cost at
    equivalent training budget'. Reviewer R2 #14: FL is ~0.1 epoch and
    centralized is 1 epoch, so 'equivalent budget' is factually wrong.
    The +0.0152 gap is a centralized-reference gap, not a federation cost.
    """
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "equivalent training budget" not in text, (
            f"R2-A1 forbidden phrase 'equivalent training budget' in "
            f"{path.name}: §6.7 must reframe to 'centralized-reference gap' "
            f"because FL ~0.1 epoch ≠ centralized 1 epoch."
        )


def test_centralized_reference_gap_present() -> None:
    """A1: after R2 fix, §6.7 must use 'centralized-reference gap' or an
    equivalent budget-non-equivalence acknowledgement."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    acceptable = ("centralized-reference gap", "centralized reference gap",
                  "different optimization budgets",
                  "different optimisation budgets",
                  "not be interpreted as a pure federation cost",
                  "not be interpreted as pure federation cost")
    found_md = any(p in md for p in acceptable)
    found_tex = any(p in tex for p in acceptable)
    assert found_md, (
        f"R2-A1 §6.7 markdown must use centralized-reference framing. "
        f"Acceptable phrasings: {acceptable}"
    )
    assert found_tex, (
        f"R2-A1 §6.7 LaTeX must use centralized-reference framing. "
        f"Acceptable phrasings: {acceptable}"
    )


# ---------- B1: cannot be centrally pooled ----------


def test_no_cannot_be_centrally_pooled() -> None:
    """B1: 'cannot be centrally pooled' is too absolute. FL is a candidate
    when raw telemetry pooling is undesirable / infeasible (policy /
    privacy / bandwidth / governance) — not because it's physically
    impossible. Also FL ≠ formal privacy (DLG arXiv 1906.08935)."""
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "cannot be centrally pooled" not in text, (
            f"R2-B1 forbidden phrase 'cannot be centrally pooled' in "
            f"{path.name} — soften to 'undesirable to pool centrally' or "
            f"similar; FL is a candidate paradigm, not the only option."
        )


# ---------- B2: inverting standard FL-benchmark assumption ----------


def test_no_inverting_standard_fl_assumption() -> None:
    """B2: abstract finding (1) currently overgeneralises 'inverting the
    standard FL-benchmark assumption that lower α is always harder'. Our
    finding is dataset-specific (ColO-RAN). Scope-limit it."""
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "inverting the standard FL-benchmark assumption" not in text, (
            f"R2-B2 forbidden phrase 'inverting the standard FL-benchmark "
            f"assumption' in {path.name} — scope to 'in this ColO-RAN setup, "
            f"contradicts the common Dirichlet-stress trend'."
        )


# ---------- B3: FedAdam-saturated headroom bounding sharpness-aware ----------


def test_no_fedadam_saturated_bounding_claim() -> None:
    """B3: abstract finding (2) currently says 'algorithm-design space is
    flat at FedAdam-saturated headroom (≤+0.016 AUC over FedAvg), bounding
    the room available to recent sharpness-aware methods'. The 'bounding'
    claim extends an N=1 (LSTM × natural-by-BS, FedSWA only) probe to a
    whole class. Reviewer says: scope to tested 5-algorithm baseline +
    explicitly note the FedSWA probe is limited to one cell."""
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "bounding the room available to recent sharpness-aware methods" not in text, (
            f"R2-B3 forbidden phrase 'bounding the room available to recent "
            f"sharpness-aware methods' in {path.name} — abstract must scope "
            f"to the tested 5-algorithm baseline + acknowledge limited "
            f"FedSWA probe."
        )


# ---------- B4: predictive ceiling empirically validated ----------


def test_no_validated_predictive_ceiling_claim() -> None:
    """B4: §6.3 / §7.5 currently say 'predictive ceiling is empirically
    validated' even though §7.5 also notes the directional inequality is
    REVERSED. Reviewer says this reads as hard-sell. Use 'magnitude
    prediction survived; directional ranking did not'."""
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "predictive ceiling is empirically validated" not in text, (
            f"R2-B4 forbidden phrase 'predictive ceiling is empirically "
            f"validated' in {path.name} — replace with 'magnitude survived; "
            f"direction did not' framing."
        )


def test_magnitude_survived_direction_did_not_present() -> None:
    """B4: after fix, the §6.3 / §7.5 paragraph must articulate the split
    (magnitude bound holds; direction was reversed)."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # Permissive set of acceptable phrasings
    acceptable = (
        "magnitude prediction survived",
        "magnitude survived",
        "magnitude-bound prediction survived",
        "magnitude bound survived",
    )
    found_md = any(p in md for p in acceptable)
    found_tex = any(p in tex for p in acceptable)
    assert found_md, (
        f"R2-B4 §6.3/§7.5 markdown must use 'magnitude survived' framing. "
        f"Acceptable phrasings: {acceptable}"
    )
    assert found_tex, (
        f"R2-B4 §6.3/§7.5 LaTeX must use 'magnitude survived' framing. "
        f"Acceptable phrasings: {acceptable}"
    )
