"""P1.4-RED forbidden-language invariants for paper tone-down.

These guard PAPER_DRAFT.md (and propagate to paper/main.tex) from
re-introducing over-claim language that the rebuttal commits to remove.

Tests fail until P1.4-GREEN edits land. After P1.4-GREEN, these tests
prevent regression (any future edit that re-adds the forbidden phrases
will break CI).
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


# ---------- forbidden over-claim language ----------

def test_no_strong_heterogeneity_helps_claim_in_markdown() -> None:
    """RED: PAPER_DRAFT.md must NOT contain 'structurally helpful, not
    harmful' — this language overstates what the data show. The actual
    finding is 'preserving natural BS grouping helps'; mechanism is
    structural-not-heterogeneity-as-such per §7.1.5 ablation."""
    text = _read(PAPER_DRAFT)
    assert "structurally helpful, not harmful" not in text, (
        "Forbidden phrase 'structurally helpful, not harmful' present in "
        "PAPER_DRAFT.md — replace with scope-limited phrasing per "
        "reviewer MC1 + P1.4-GREEN."
    )


def test_no_strong_heterogeneity_helps_claim_in_latex() -> None:
    text = _read(MAIN_TEX)
    assert "structurally helpful, not harmful" not in text, (
        "Forbidden phrase present in main.tex — propagate the markdown "
        "tone-down to LaTeX per P1.4-GREEN."
    )


def test_no_deployment_anti_pattern_phrase() -> None:
    """RED: 'deployment anti-pattern' language is too prescriptive for the
    evidence base; soften to 'should be justified by specific operational
    constraint'."""
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        assert "deployment anti-pattern" not in text, (
            f"Forbidden phrase 'deployment anti-pattern' in {path.name} — "
            f"reviewer MC1 + Minor#3."
        )


def test_no_dismiss_language_for_fedswa_family() -> None:
    """RED: §2.6 + §7.5 must not use 'dismiss' / 'rule out' / 'cannot
    exceed' for the FedSWA / FedSCAM / FedMoSWA family. The mechanism
    argument is predictive ('we predict |Δ|<|Δ_FedAdam|') not empirical;
    use hedged language (predict, anticipated, likely) per reviewer MC7."""
    forbidden = ("we accordingly dismiss", "cannot systematically exceed",
                 "rule out FedSWA", "rules out FedSWA")
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        for phrase in forbidden:
            assert phrase not in text, (
                f"Forbidden FedSWA-dismissal phrase {phrase!r} in {path.name}"
            )


def test_implementation_specific_caveat_in_c4() -> None:
    """RED: §1 contribution 4 (architecture-leverage claim) must contain
    an 'on this implementation' caveat. Mamba uses pure-PyTorch sequential
    scan, not Triton kernel — energy comparison is implementation-specific.
    Reviewer MC4 + Paper §4.1 + §8 L1 already partly acknowledge this."""
    text = _read(MAIN_TEX)
    # The contribution 4 paragraph must mention implementation-specificity
    # near the energy claim. We check a permissive set of acceptable
    # phrasings.
    acceptable = ("on this implementation", "implementation-specific",
                  "pure-PyTorch implementation", "this RTX~4080")
    found = any(phrase in text for phrase in acceptable)
    assert found, (
        "main.tex contribution 4 must include an implementation-specific "
        f"caveat. Acceptable phrasings: {acceptable}. Reviewer MC4 + P1.4-GREEN."
    )
