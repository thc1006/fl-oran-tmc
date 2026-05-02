"""TDD red-gate for PAPER_DRAFT.md text correctness.

Each test asserts a known-true invariant about the paper draft. These run
BEFORE any rewrite work so we know exactly which N of 25+ assertions still
fail. After each rewrite pass, rerun to track RED→GREEN progress.

Evidence anchors (each invariant has explicit citation):
  * Dataset facts: artifacts/step1_factfinding.json (measured 2026-05-02
    from data/coloran_raw_unified.parquet)
  * License: docs/ADR-001-v5-tmc-paper-plan.md D-15 (Apache-2.0)
  * Hardware: NVIDIA RTX 4080 spec sheet (TSMC 4N, 320 W TGP)
  * Code behaviour: src/fl_oran/training/fl_v7.py:636-637 (global pos_weight),
    src/fl_oran/data_v2/partition.py (mode="iid" = natural-by-BS)

Run:  pytest tests/test_paper_draft_invariants.py --no-cov -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "docs" / "PAPER_DRAFT.md"


@pytest.fixture(scope="module")
def paper_text() -> str:
    return PAPER.read_text()


# ============================================================================
# Dataset cardinality (memory/dataset_facts_phase5.md)
# ============================================================================

class TestDatasetCardinality:
    """ColO-RAN public release has 7 BS, 3 slices, 3 schedulers, 28 traffic
    configs, 17 continuous features. The current PAPER_DRAFT v2 says 4
    slices / 4 schedulers / 29 features / bs 0..6 — all wrong."""

    def test_paper_does_not_say_4_slices(self, paper_text: str) -> None:
        """slice_id only takes 3 values [0, 1, 2] in the parquet."""
        forbidden = ["4 logical slices", "slice_id ∈ {0, 1, 2, 3}",
                     "slice_id in {0, 1, 2, 3}", "4 slices"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_4_schedulers(self, paper_text: str) -> None:
        forbidden = ["4 admission-control", "4 schedulers",
                     "4 scheduler policies", "4 admission control"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_29_continuous_features(self, paper_text: str) -> None:
        forbidden = ["29 transport-block", "29 continuous features",
                     "29 continuous-feature"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_bs_0_to_6(self, paper_text: str) -> None:
        """Actual bs_id range is [1..7], not [0..6]."""
        forbidden = ["bs_id 0..6", "bs_id ∈ {0..6}", "bs_id ∈ {0, …, 6}",
                     "bs_id in {0..6}", "bs_id 0-6"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_4dim_dirichlet(self, paper_text: str) -> None:
        """Dirichlet partition over 3 slices, not 4."""
        forbidden = ["Dirichlet([α, α, α, α])", "Dirichlet([alpha, alpha, alpha, alpha])"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_8_to_12_pos_rate(self, paper_text: str) -> None:
        """Measured global train pos rate is 30.9 %, not 8-12 %."""
        forbidden = ["8-12 %", "8-12%", "8 to 12 %", "8 to 12%", "8 - 12 %"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"


class TestDatasetCardinalityCorrected:
    """The corrected facts that SHOULD be in the rewritten draft."""

    def test_paper_says_3_slices(self, paper_text: str) -> None:
        accepted = ["3 logical slices", "3 slices", "slice_id ∈ {0, 1, 2}",
                    "slice_id ∈ {0..2}"]
        assert any(s in paper_text for s in accepted), (
            "paper must say 3 slices somewhere"
        )

    def test_paper_says_30_pct_pos_rate(self, paper_text: str) -> None:
        """Allow either '30.9' or '~30 %' or '30 percent' phrasing."""
        accepted = ["30.9", "~30 %", "≈30 %", "≈ 30 %", "30 percent"]
        assert any(s in paper_text for s in accepted), (
            "paper must report measured 30.9% positive rate"
        )

    def test_paper_says_17_continuous(self, paper_text: str) -> None:
        accepted = ["17 continuous features", "17 continuous-feature",
                    "17 continuous"]
        assert any(s in paper_text for s in accepted), (
            "paper must say 17 continuous features"
        )

    def test_paper_says_bs_id_1_to_7(self, paper_text: str) -> None:
        accepted = ["bs_id ∈ {1, …, 7}", "bs_id ∈ {1..7}",
                    "bs_id 1..7", "bs_id 1-7", "bs_id ∈ {1, ..., 7}",
                    "`bs_id` ∈ {1, …, 7}",  # backticked variant from §3.1 edit
                    "`bs_id` 1..7"]
        assert any(s in paper_text for s in accepted), (
            "paper must give bs_id range as 1..7"
        )

    def test_paper_says_3_schedulers(self, paper_text: str) -> None:
        accepted = ["3 schedulers", "3 admission-control",
                    "sched ∈ {0..2}", "3 scheduler policies",
                    "3 scheduling policies",  # D4 ColO-RAN-doc-aligned phrasing
                    "sched ∈ {0, 1, 2}"]
        assert any(s in paper_text for s in accepted), (
            "paper must say 3 schedulers"
        )

    def test_paper_says_3dim_dirichlet(self, paper_text: str) -> None:
        accepted = ["Dirichlet([α, α, α])", "Dirichlet([α]*3)",
                    "Dirichlet([alpha, alpha, alpha])"]
        assert any(s in paper_text for s in accepted), (
            "paper Dirichlet partition must be 3-dim"
        )


# ============================================================================
# License (ADR-001 D-15)
# ============================================================================

class TestLicense:
    def test_paper_does_not_say_mit_license(self, paper_text: str) -> None:
        """ADR-001 D-15 fixes license = Apache-2.0; paper inherited 'MIT'."""
        forbidden = ["MIT license", "MIT/Apache-2.0", "MIT-licensed"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_says_apache_2_0(self, paper_text: str) -> None:
        assert "Apache-2.0" in paper_text, "paper must declare Apache-2.0 license"


# ============================================================================
# Hardware facts (RTX 4080 spec sheet)
# ============================================================================

class TestHardwareFacts:
    def test_paper_does_not_say_12nm(self, paper_text: str) -> None:
        """RTX 4080 is TSMC 4N (5 nm class), not 12 nm. 12 nm was Turing."""
        forbidden = ["12 nm Lovelace", "12nm Lovelace", "12 nm process"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_does_not_say_120w_tgp(self, paper_text: str) -> None:
        """RTX 4080 has 320 W TGP, not 120 W."""
        forbidden = ["~120 W TGP", "120 W TGP", "120W TGP"]
        for s in forbidden:
            assert s not in paper_text, f"forbidden phrase still present: {s!r}"

    def test_paper_says_tsmc_4n_or_5nm(self, paper_text: str) -> None:
        accepted = ["TSMC 4N", "TSMC 4n", "5 nm Lovelace",
                    "5nm Lovelace", "5 nm class"]
        assert any(s in paper_text for s in accepted), (
            "paper must give correct RTX 4080 process node"
        )

    def test_paper_says_320w_tgp(self, paper_text: str) -> None:
        accepted = ["320 W TGP", "320W TGP", "320 W"]
        assert any(s in paper_text for s in accepted), (
            "paper must give correct RTX 4080 TGP"
        )


# ============================================================================
# Numerical consistency
# ============================================================================

class TestNumericalConsistency:
    def test_spiking_gap_is_qualified(self, paper_text: str) -> None:
        """§1 hook claims '≈0.10-AUC gap across all 5 algos'. The gap on
        Spiking is only ~0.022. Either drop the 0.10 claim or qualify
        'on dense backbones; smaller on Spiking'."""
        if "0.10-AUC gap" in paper_text or "≈0.10" in paper_text:
            # if the unqualified claim is present, must be paired with a
            # Spiking caveat
            assert "smaller" in paper_text or "0.022" in paper_text or \
                "Spiking-SSM" in paper_text and "smaller" in paper_text, (
                    "0.10 AUC gap claim must be qualified with Spiking caveat"
                )

    def test_mamba_scaffold_std_consistent_precision(self, paper_text: str) -> None:
        """§1 contribution 3 has '0.7609 ± 0.083'; should be 4-decimal
        '± 0.0830' for consistency with §6."""
        if "0.7609" in paper_text:
            assert "± 0.083" not in paper_text or "± 0.0830" in paper_text, (
                "Mamba×SCAFFOLD std should be 4-decimal '± 0.0830'"
            )


# ============================================================================
# Internal consistency
# ============================================================================

class TestInternalConsistency:
    def test_no_bonferroni_contradiction(self, paper_text: str) -> None:
        """§1 line 26 currently says 'Bonferroni multi-comparison correction'
        AND §4.5 says 'uncorrected'. Pick one."""
        s1_says_corrected = "Bonferroni multi-comparison correction" in paper_text
        s45_says_uncorrected = "All reported p-values are uncorrected" in paper_text
        assert not (s1_says_corrected and s45_says_uncorrected), (
            "paper must not say Bonferroni applied in §1 AND uncorrected in §4.5"
        )

    def test_no_false_71_ablation_promise(self, paper_text: str) -> None:
        """§7.1 currently says 'We provide a controlled ablation in §7.2'
        but §7.2 is about algorithmic flatness — false promise."""
        forbidden = ["We provide a controlled ablation in §7.2",
                     "controlled ablation in §7.2"]
        for s in forbidden:
            assert s not in paper_text, f"false forward-reference still present: {s!r}"

    def test_no_invalidated_uniform_pos_weight_ablation_in_section_9(self, paper_text: str) -> None:
        """§9 conclusion previously promised "§7.2 uniform-pos_weight
        ablation that would isolate the mechanism's (b) component" — but
        the (b) component (per-client pos_weight) was eliminated by Step 1
        verifying fl_v7 uses globally pooled pos_weight. The promised
        ablation is therefore based on an invalidated mechanism story."""
        forbidden = ["uniform-pos_weight ablation that would isolate",
                     "§7.2 uniform-pos_weight"]
        for s in forbidden:
            assert s not in paper_text, (
                f"§9 still references invalidated uniform-pos_weight ablation: {s!r}"
            )

    def test_zenodo_release_uses_future_tense(self, paper_text: str) -> None:
        """§9 must not claim Zenodo-archived release in present tense
        because §1 contribution 5 + §8 L5 explicitly defer release to
        paper acceptance."""
        forbidden = ["We release the full benchmark artefact"]
        accepted = ["Upon paper acceptance, we will release the full benchmark artefact",
                    "we will release the full benchmark artefact"]
        for s in forbidden:
            assert s not in paper_text, (
                f"§9 must use future tense for Zenodo release; found {s!r}"
            )
        assert any(s in paper_text for s in accepted), (
            "§9 must explicitly defer release to paper acceptance"
        )

    def test_has_abstract_section(self, paper_text: str) -> None:
        """T-M P0 #2: TMC requires Abstract before §1 Introduction."""
        assert "## Abstract" in paper_text, (
            "paper must have explicit '## Abstract' section before §1"
        )

    def test_has_keywords(self, paper_text: str) -> None:
        """T-M P0 #4: TMC requires keywords."""
        assert "Keywords" in paper_text, "paper must include keywords block"

    def test_has_author_block(self, paper_text: str) -> None:
        """T-M P0 #3: TMC requires author + affiliation + ORCID."""
        assert "Hao-Chun Tsai" in paper_text or "thc1006" in paper_text
        assert "ORCID" in paper_text, "paper must include ORCID identifier"

    def test_has_references_section(self, paper_text: str) -> None:
        """T-M P0 #7: TMC requires References / Bibliography section."""
        assert "## References" in paper_text, (
            "paper must have '## References' section"
        )

    def test_has_reproducibility_section(self, paper_text: str) -> None:
        """T-M P0 #6: TMC editor desk-reject if reproducibility section
        is missing for a paper claiming reproducible-benchmark contribution."""
        assert "## 5. Reproducibility" in paper_text, (
            "paper must have non-trivial '## 5. Reproducibility infrastructure'"
        )

    def test_status_block_removed(self, paper_text: str) -> None:
        """T-M P0 #1: status block (DRAFT v3 / v3 changes / etc.) must
        not appear in submission build."""
        forbidden = ["DRAFT v3", "v3 changes (2026", "v2 was 2026-05-01"]
        for s in forbidden:
            assert s not in paper_text, (
                f"draft-tracking text still present: {s!r}; "
                f"move to docs/PAPER_STATUS.md"
            )

    def test_no_task_tracking_comments(self, paper_text: str) -> None:
        """T-M P0 #5: internal task tracking comments must not appear."""
        forbidden = ["task #146", "task #147", "task #148", "is task #"]
        for s in forbidden:
            assert s not in paper_text, (
                f"internal task-tracking reference still in paper: {s!r}"
            )

    def test_figures_have_captions(self, paper_text: str) -> None:
        """T-M P0 #8: Figures 1-3 must have explicit captions."""
        assert "**Figure 1:" in paper_text, "Figure 1 must have explicit caption"
        assert "**Figure 2:" in paper_text, "Figure 2 must have explicit caption"
        assert "**Figure 3:" in paper_text, "Figure 3 must have explicit caption"

    def test_table_1_numbered(self, paper_text: str) -> None:
        """T-M P0 #9: Tables must be numbered."""
        assert "**Table 1:" in paper_text, "Table 1 must be numbered + captioned"

    def test_fedscam_date_consistent(self, paper_text: str) -> None:
        """FedSCAM is arxiv:2601.00853 (Jan 2026), not "Dec 2025".
        T-G audit verified the actual date; multiple paper sites must
        use the verified date."""
        forbidden = ["FedSCAM (Dec 2025)", "FedSCAM [Dec 2025]"]
        for s in forbidden:
            assert s not in paper_text, (
                f"stale FedSCAM date still present: {s!r}; "
                f"correct is 'arxiv:2601.00853, Jan 2026'"
            )


# ============================================================================
# Mechanism claims (T-A code reading)
# ============================================================================

class TestMechanismClaims:
    def test_paper_does_not_claim_per_client_pos_weight(self, paper_text: str) -> None:
        """fl_v7.py:636-637 confirms GLOBAL pos_weight, not per-client.
        The 'per-client pos_weight specialist' mechanism is invalidated."""
        forbidden = [
            "per-client pos_weight",
            "per-client local pos_weight",
            "per-client `pos_weight`",
            "per-client positive-class scarcity",
        ]
        for s in forbidden:
            assert s not in paper_text, f"invalidated mechanism claim still present: {s!r}"

    def test_paper_does_not_claim_natural_is_dirichlet_limit(self, paper_text: str) -> None:
        """Dirichlet partitions over slice_id; natural-by-BS partitions
        over bs_id. They are orthogonal axes, not the same family."""
        forbidden = [
            "limit case of Dirichlet",
            "Dirichlet's limit case",
            "natural-by-BS as the limit case",
            "natural base-station partition is the limit",
        ]
        for s in forbidden:
            assert s not in paper_text, f"invalidated framing still present: {s!r}"

    def test_paper_does_not_claim_fedavg_equals_swa(self, paper_text: str) -> None:
        """FedAvg averages independent client models; SWA averages
        trajectory iterates of a single model. The 'precisely the linear
        weight ensemble from SWA' claim is mechanism-level wrong."""
        forbidden = [
            "is precisely the linear weight ensemble from the SWA",
            "FedAvg `(1/N) Σ θ_i` is precisely",
        ]
        for s in forbidden:
            assert s not in paper_text, f"invalidated SWA framing still present: {s!r}"
