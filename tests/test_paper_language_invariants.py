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


def test_p15_naive_baselines_section_in_paper() -> None:
    """P1.5 integration guard: §6 must reference the naive baselines (P1.1
    GREEN result). Catches regression where someone removes the §6.7
    table or strips the artifacts/baselines/naive_results.json reference."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    assert "Naive baselines" in md, "§6 markdown must contain a 'Naive baselines' section"
    assert "0.5133" in md and "0.6258" in md and "0.6523" in md, (
        "§6 markdown must report the 3 naive baseline AUCs (0.5133, 0.6258, 0.6523)"
    )
    assert "naive_results.json" in md, "§6 markdown must reference the canonical results JSON"
    assert "Naive baselines" in tex or "naive-baselines" in tex.lower(), (
        "main.tex must mirror the §6 naive baseline section"
    )


def test_p15_tr_embedding_quantification_section_in_paper() -> None:
    """P1.5 integration guard: §7.1.6 must report the cross-arch P1.2
    quantification (LSTM 9.2%, Mamba 10.2%, Spiking 2.3%, all ≤10%)."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # 3-arch cross-arch result (R3.1)
    for pct in ("9.2%", "10.2%", "2.3%"):
        assert pct in md, f"§7.1.6 markdown must report {pct} per-arch shrinkage"
    assert "10%" in md and "90%" in md, (
        "§7.1.6 markdown must state the ≤10% bug / ≥90% structural conclusion"
    )
    assert "tr-embedding" in tex.lower() or "tr embedding" in tex.lower(), (
        "main.tex must mirror the tr-embedding-bug-confound subsection"
    )
    # The 3-arch table must be in main.tex too
    assert "tab:tr-embedding-shrinkage" in tex, (
        "main.tex must contain the per-arch shrinkage table"
    )


def test_p15_fedbn_reduction_proof_in_l2() -> None:
    """P1.5 integration guard: §8 L2 must use the FedBN reduction proof
    instead of the previous 'we expect FedBN to extend not overturn'
    framing."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    assert "reduces bit-exactly to FedAvg" in md, (
        "§8 L2 markdown must include the FedBN-reduces-to-FedAvg statement"
    )
    assert "fedbn_reduces_to_fedavg.md" in md, (
        "§8 L2 markdown must reference the audit doc with the proof"
    )
    # Also ensure the old soft-framing language is GONE
    assert "we expect the result of such a follow-up" not in md, (
        "§8 L2 markdown must not retain the pre-P1.5 soft framing"
    )
    assert "reduces bit-exactly to FedAvg" in tex, (
        "main.tex must mirror the FedBN reduction-to-FedAvg statement"
    )


def test_p2_loto_section_in_paper() -> None:
    """P2.1 LOTO cluster bootstrap result must be in §8 L15 (markdown + LaTeX).
    Reviewer MC5 answer: external uncertainty exceeds internal by 2-18×."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # σ ratio per arch (the headline finding)
    for ratio in ("17.8", "13.0", "2.3"):
        assert ratio in md, f"§8 L15 markdown must report ratio {ratio} per-arch"
    # Width ratio range — must use accurate "3.7" not rounded "4" (LOTO-A fix)
    assert "3.7" in md, "§8 L15 markdown must use accurate width-ratio lower bound 3.7×, not rounded 4×"
    assert "LOTO" in md or "leave-one-traffic-config-out" in md, (
        "§8 L15 markdown must name the LOTO methodology"
    )
    assert "tab:loto-variance" in tex, (
        "main.tex must contain the LOTO variance-decomposition table"
    )
    assert "results*.json" in md or "results*.json" in tex, (
        "§8 L15 must reference artifacts/p2_loto/results*.json"
    )


def test_p22_inference_latency_section_in_paper() -> None:
    """P2.2 inference latency + comm bytes (MC6) must be in §6.8."""
    md = _read(PAPER_DRAFT)
    tex = _read(MAIN_TEX)
    # Headline values — RTX 4080 GPU 1-sample latency
    for v in ("0.15", "0.52", "1.57"):
        assert v in md, f"§6.8 markdown must report GPU 1-sample latency {v} ms"
    # Communication bytes per client
    for v in ("174", "158", "170"):
        assert v in md, f"§6.8 markdown must report comm KiB/client {v}"
    # 10 ms RIC budget framing
    assert "10 ms" in md or "10\\,ms" in md or "10ms" in md, (
        "§6.8 markdown must mention the 10 ms near-RT RIC budget"
    )
    assert "tab:deployment-cost" in tex, (
        "main.tex must contain the deployment-cost table"
    )
    assert "p2_inference/results.json" in md or "p2_inference/results.json" in tex, (
        "must reference artifacts/p2_inference/results.json"
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
