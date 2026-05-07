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
    """A1: after R2 fix, §6.7 must use a budget-non-equivalence-aware
    framing for the federation-cost discussion. Round 2 deep-review
    upgraded the canonical phrasing from 'centralized-reference gap'
    (which was ambiguous after C1) to 'matched-budget federation cost'
    (the C1 same-step result, +0.0084 AUC). Both frames are accepted
    so legacy commits still parse."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    acceptable = (
        # Round 2 deep-review canonical (post-C1)
        "matched-budget federation cost",
        "matched compute budget",
        "matched 25k-step budget",
        # Phase 3a interim
        "centralized-reference gap", "centralized reference gap",
        "different optimization budgets",
        "different optimisation budgets",
        # Phase 1 interim
        "not be interpreted as a pure federation cost",
        "not be interpreted as pure federation cost",
    )
    found_md = any(p in md for p in acceptable)
    found_tex = any(p in tex for p in acceptable)
    assert found_md, (
        f"R2-A1 §6.7 markdown must use a budget-non-equivalence-aware "
        f"framing. Acceptable phrasings: {acceptable}"
    )
    assert found_tex, (
        f"R2-A1 §6.7 LaTeX must use a budget-non-equivalence-aware "
        f"framing. Acceptable phrasings: {acceptable}"
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


# ---------- R2 Round-2 deep-review additions (2026-05-07) ----------


def test_no_stale_0_0152_behind_phrasing() -> None:
    """R2-RA: §6.7 Statistical caveat must NOT call FL '0.0152 AUC behind
    centralized at 1 epoch' as if it were the federation cost — that
    contradicts the Phase 3a reframe (+0.0084 is the matched-budget true
    cost). The 0.0152 number can still appear (in the table + as the
    historical 1-epoch reference) but not framed as the federation cost."""
    forbidden = (
        "FL at ≈0.1 epoch is only 0.0152 AUC behind",
        "FL at $\\approx 0.1$ epoch is only $0.0152$ AUC behind",
    )
    for path in (PAPER_DRAFT, MAIN_TEX):
        text = _read(path)
        for phrase in forbidden:
            assert phrase not in text, (
                f"R2-RA forbidden phrase {phrase!r} in {path.name}: §6.7 must "
                f"not frame the 0.0152 number as 'the' gap; the matched-budget "
                f"+0.0084 number is the true federation cost (R2 C1)."
            )


def test_centralized_25k_steps_table_row_present() -> None:
    """R2-RB: §6.7 table must include the C1 same-step centralized row
    (0.9243 / -0.0084) so the table's interpretive load matches the
    prose's matched-budget framing."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # Markdown table row
    assert ("Centralized LSTM, 25k steps" in md
            and "0.9243" in md and "−0.0084" in md), (
        "R2-RB §6.7 markdown table must contain a row for 'Centralized LSTM, "
        "25k steps' with 0.9243 / −0.0084 alongside the 1-epoch row."
    )
    # LaTeX table row
    assert ("Centralized LSTM, 25k steps" in tex and "0.9243" in tex), (
        "R2-RB §6.7 LaTeX table must contain the 25k-steps row."
    )


def test_matched_budget_cluster_ci_present() -> None:
    """R2-RC: §6.7 statistical caveat must report cluster CI for BOTH
    the matched-budget federation cost AND the historical 1-epoch
    reference gap, with explicit labelling. Prevents reverting to the
    ambiguous 'centralized-reference gap' singular."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    for phrase in ("matched-budget federation cost", "[+0.001, +0.015]"):
        assert phrase in md, (
            f"R2-RC §6.7 markdown must contain '{phrase}' so the matched-"
            f"budget cluster CI is distinguished from the 1-epoch reference."
        )
        assert phrase in tex, (
            f"R2-RC §6.7 LaTeX must contain '{phrase}'."
        )


def test_c3_within_arch_std_clarification_present() -> None:
    """R2-RD: §8 L2 must clarify that 0.0036 is cross-OBSERVATION std
    inflated by per-arch mean spread, with within-arch stds reported
    separately. Prevents regression to the misleading 'std 0.0036'
    parenthetical that implied per-cell noise was 0.4pp when actually
    it's 0.05-0.3pp depending on arch."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    for phrase in ("within-arch", "order of magnitude tighter"):
        assert phrase in md, (
            f"R2-RD §8 L2 markdown must contain '{phrase}' to clarify the "
            f"0.0036 cross-observation std vs the much tighter per-arch stds."
        )
        assert phrase in tex, (
            f"R2-RD §8 L2 LaTeX must contain '{phrase}'."
        )


def test_c4_robustness_scoped_to_lstm() -> None:
    """R2-RE: §7.1.6 must scope the 'C1 mechanism finding is robust to
    the embedding choice' claim to the LSTM × FedAvg × natural-by-BS
    configuration that C4 actually re-trained. Mamba/Spiking weren't
    re-tested with no-tr; their ≥90% structural claim still rests on
    the meanfix proxy, not the no-tr ablation."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # The bare claim must be gone; scoped claim must be present
    bare = "C1 mechanism finding is robust to the embedding choice."
    assert bare not in md, (
        "R2-RE §7.1.6 markdown must SCOPE the 'robust to embedding' claim "
        "to LSTM (C4 didn't re-test Mamba/Spiking)."
    )
    bare_tex = "C1 mechanism finding is robust to the embedding choice."
    assert bare_tex not in tex, (
        "R2-RE §7.1.6 LaTeX must SCOPE the 'robust to embedding' claim."
    )
    # Scoped phrasing must be present
    scoped_phrase = "the meanfix proxy above remains the basis"
    assert scoped_phrase in md, (
        f"R2-RE §7.1.6 markdown must explicitly note '{scoped_phrase}' "
        f"for Mamba/Spiking (no no-tr ablation on those archs)."
    )
    assert scoped_phrase in tex, (
        f"R2-RE §7.1.6 LaTeX must explicitly note '{scoped_phrase}'."
    )


# ---------- REM-A — §8 L16 privacy caveat (added 2026-05-07) ----------


def test_rem_a_l16_privacy_caveat_present() -> None:
    """REM-A: §8 L16 must exist with FL ≠ formal privacy framing + DLG
    cite (Zhu et al. 2019). Prevents regression where someone removes
    L16 leaving the abstract's 'see §8 L16' as a dangling forward-ref
    (which is what created REM-A in the first place when the abstract
    said L17 but §8 stopped at L15)."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # L16 must exist in §8
    assert "L16 (added 2026-05-07 per reviewer R2 #4" in md, (
        "REM-A §8 L16 markdown header missing"
    )
    assert "L16 (added 2026-05-07 per reviewer R2 \\#4" in tex, (
        "REM-A §8 L16 LaTeX header missing"
    )
    # FL ≠ formal privacy phrasing
    for path, text in [(PAPER_DRAFT, md), (MAIN_TEX, tex)]:
        has_fl_neq = ("FL ≠ formal privacy" in text or
                      "FL $\\neq$ formal privacy" in text)
        assert has_fl_neq, (
            f"REM-A §8 L16 in {path.name} must explicitly say "
            f"'FL ≠ formal privacy' (or LaTeX equivalent)"
        )
    # DLG citation (Zhu et al. 2019, arXiv:1906.08935)
    assert "1906.08935" in md, "REM-A §8 L16 must cite arXiv:1906.08935 (DLG)"
    assert "Zhu2019_DLG" in tex, "REM-A §8 L16 LaTeX must cite Zhu2019_DLG"


def test_rem_a_abstract_forward_ref_to_l16_not_l17() -> None:
    """REM-A: abstract forward-ref must say 'see §8 L16' (or LaTeX
    equivalent), NOT L17 — L17 doesn't exist; that was the dangling
    ref I created in B1 abstract fix and corrected here."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    assert "see §8 L16" in md, (
        "REM-A abstract markdown must forward-ref §8 L16"
    )
    assert "Section~\\ref{sec:limits} L16" in tex, (
        "REM-A abstract LaTeX must forward-ref Section~\\ref{sec:limits} L16"
    )
    # Forbidden: dangling L17 ref must NOT be reintroduced
    assert "see §8 L17" not in md, (
        "REM-A REGRESSION: 'see §8 L17' is dangling (L17 doesn't exist)"
    )
    assert "Section~\\ref{sec:limits} L17" not in tex, (
        "REM-A REGRESSION: 'Section ref L17' is dangling"
    )
