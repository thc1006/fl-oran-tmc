"""Hook to prevent paper-text quick-claims that don't match the underlying
data sources.

Each numerical claim in PAPER_DRAFT.md that cites a specific value (an AUC,
a delta, a KL divergence, a hardware-drift bound) MUST programmatically
match its source artifact. The source artifact is the real measurement file
(`artifacts/step1_factfinding.json`, `artifacts/step2_mechanism_search.json`,
`artifacts/v7_stage2_full/aggregated_phase5.json`,
`artifacts/v7_ablation_random_split/v7_*/summary.json`). If a claim does not
match its source within tolerance, the test fails red — preventing
"hallucinated number" type quick-claims that previously slipped through.

Pattern: each test loads the source, recomputes the claimed metric, and
greps PAPER_DRAFT.md for a substring containing the value at the expected
precision. Tolerance is documented per-test.

Run:  pytest tests/test_paper_claims_sources.py --no-cov -v

If a claim's source file does not exist (e.g. ablation not yet run), the
test SKIPs rather than fails — so the hook stays useful during incremental
draft work.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs" / "PAPER_DRAFT.md"
STEP1 = REPO / "artifacts" / "step1_factfinding.json"
STEP2 = REPO / "artifacts" / "step2_mechanism_search.json"
AGG_PHASE5 = REPO / "artifacts" / "v7_stage2_full" / "aggregated_phase5.json"
ABLATION_DIR = REPO / "artifacts" / "v7_ablation_random_split"


@pytest.fixture(scope="module")
def paper_text() -> str:
    return PAPER.read_text()


@pytest.fixture(scope="module")
def step1_data() -> dict:
    """GH#8: load step1_factfinding.json or skip the test cleanly with a
    regenerate hint. The artifact is gitignored (produced by
    ``python scripts/step1_fact_finding.py``); a fresh checkout will
    skip rather than fail-noisy when the file is absent."""
    if not STEP1.exists():
        pytest.skip(
            f"{STEP1.relative_to(REPO)} missing. "
            "Regenerate via: python scripts/step1_fact_finding.py"
        )
    return json.loads(STEP1.read_text())


@pytest.fixture(scope="module")
def step2_data() -> dict:
    """GH#8: load step2_mechanism_search.json or skip cleanly. Artifact
    is produced by ``python scripts/step2_mechanism_search.py``."""
    if not STEP2.exists():
        pytest.skip(
            f"{STEP2.relative_to(REPO)} missing. "
            "Regenerate via: python scripts/step2_mechanism_search.py"
        )
    return json.loads(STEP2.read_text())


# ============================================================================
# §3 dataset facts → step1_factfinding.json
# ============================================================================

class TestDatasetFactsAgainstStep1:

    def test_pos_rate_matches_step1(self, paper_text: str) -> None:
        if not STEP1.exists():
            pytest.skip("step1_factfinding.json not yet generated")
        s = json.loads(STEP1.read_text())
        true_pct = round(s["Q3_global_train_pos_rate"] * 100, 1)
        assert f"{true_pct}" in paper_text or f"{true_pct:.1f}" in paper_text, (
            f"paper must report measured pos rate {true_pct} %, not a placeholder"
        )

    def test_n_continuous_features_matches_step1(self, paper_text: str) -> None:
        if not STEP1.exists():
            pytest.skip("step1_factfinding.json not yet generated")
        s = json.loads(STEP1.read_text())
        n = s["Q1_n_continuous_features"]
        assert f"{n} continuous features" in paper_text, (
            f"paper must say '{n} continuous features'"
        )

    def test_slice_count_matches_step1(self, paper_text: str) -> None:
        if not STEP1.exists():
            pytest.skip("step1_factfinding.json not yet generated")
        s = json.loads(STEP1.read_text())
        n_slice = s["Q2_categorical_summary"]["slice_id"]["n_unique"]
        assert f"{n_slice} logical slices" in paper_text or \
               f"{n_slice} slices" in paper_text, (
                   f"paper must say '{n_slice} slices' (Step 1 measured)"
               )

    def test_bs_count_matches_step1(self, paper_text: str) -> None:
        if not STEP1.exists():
            pytest.skip("step1_factfinding.json not yet generated")
        s = json.loads(STEP1.read_text())
        n_bs = s["Q2_categorical_summary"]["bs_id"]["n_unique"]
        assert f"{n_bs} base stations" in paper_text, (
            f"paper must say '{n_bs} base stations'"
        )


# ============================================================================
# §7.1.2 KL features → step2_mechanism_search.json
# ============================================================================

def _step2_top_n_features(n: int) -> list[tuple[str, float]]:
    """Top-n continuous features by mean pairwise bs-KL."""
    if not STEP2.exists():
        return []
    s = json.loads(STEP2.read_text())
    feats = s["C1_per_feature_pairwise_bs_KL"]
    ranked = sorted(feats.items(), key=lambda kv: kv[1]["kl_mean"], reverse=True)
    return [(f, info["kl_mean"]) for f, info in ranked[:n]]


class TestSection712KLAgainstStep2:

    def test_top1_kl_feature_matches(self, paper_text: str) -> None:
        if not STEP2.exists():
            pytest.skip("step2 not yet run")
        top = _step2_top_n_features(5)
        f1, kl1 = top[0]
        # Must mention top-1 feature name + its KL value (rounded to 3 decimals)
        assert f1 in paper_text, f"§7.1.2 must name top-1 feature '{f1}'"
        kl_str = f"{kl1:.3f}"
        kl_alt = f"{kl1:.2f}"  # accept 2 or 3 decimal precision
        assert kl_str in paper_text or kl_alt in paper_text, (
            f"§7.1.2 must report KL={kl_str} for {f1}"
        )

    def test_top2_kl_feature_matches(self, paper_text: str) -> None:
        if not STEP2.exists():
            pytest.skip("step2 not yet run")
        top = _step2_top_n_features(5)
        f2, kl2 = top[1]
        assert f2 in paper_text, f"§7.1.2 must name top-2 feature '{f2}'"

    def test_no_fabricated_kl_features(self, paper_text: str) -> None:
        """If §7.1.2 mentions a feature with a specific KL value, that
        feature must be in the JSON's top-10 (cushion against future-edit
        hallucination)."""
        if not STEP2.exists():
            pytest.skip("step2 not yet run")
        top10 = [f for f, _ in _step2_top_n_features(10)]
        # We previously hallucinated dl_buffer_bytes / rx_brate_ul_Mbps /
        # slice_prb in the top-5; this test ensures regressions don't
        # reintroduce them as "top-5 features by bs-KL". They CAN appear
        # in the paper for other reasons (just not as top-N by bs-KL).
        # We check by looking for the suspicious phrase pattern.
        forbidden_top5_claims = [
            "rx_brate_ul_Mbps` (0.05",   # the hallucinated bracket
            "slice_prb` (0.04",
            "dl_buffer_bytes` (0.05",     # this one DOES appear at position 8 with KL 0.0163, NOT 0.05
        ]
        for s in forbidden_top5_claims:
            assert s not in paper_text, (
                f"§7.1.2 contains the previously-hallucinated phrase {s!r}; "
                f"verified top-5 are: {[f for f, _ in _step2_top_n_features(5)]}"
            )


# ============================================================================
# §7.1.1 ablation deltas → V100 cell summaries vs Phase 5 aggregator
# ============================================================================

def _v100_ablation_means_by_arch() -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for cell in glob.glob(str(ABLATION_DIR / "v7_*/summary.json")):
        s = json.loads(Path(cell).read_text())
        arch = s["config"]["arch"]
        out.setdefault(arch, []).append(s["test"]["auc"])
    return {arch: st.mean(aucs) for arch, aucs in out.items()}


def _phase5_iid_fedavg_means_by_arch() -> dict[str, float]:
    if not AGG_PHASE5.exists():
        return {}
    s = json.loads(AGG_PHASE5.read_text())
    out: dict[str, float] = {}
    for k, v in s["stats"].items():
        # k = "<arch>::<algo>::<pmode>::<alpha_tag>"
        parts = k.split("::")
        if len(parts) != 4:
            continue
        arch, algo, pmode, alpha_tag = parts
        if algo == "fedavg" and pmode == "iid":
            out[arch] = v["test_auc_mean"]
    return out


class TestSection711AblationDeltas:

    def test_lstm_random_split_mean_in_paper(self, paper_text: str) -> None:
        m = _v100_ablation_means_by_arch()
        if "lstm" not in m:
            pytest.skip("V100 ablation cells not yet aggregated")
        val = round(m["lstm"], 4)
        assert f"{val:.4f}" in paper_text, (
            f"§7.1.1 must report LSTM random_split mean = {val:.4f}"
        )

    def test_mamba_random_split_mean_in_paper(self, paper_text: str) -> None:
        m = _v100_ablation_means_by_arch()
        if "mamba" not in m:
            pytest.skip("V100 ablation cells not yet aggregated")
        val = round(m["mamba"], 4)
        assert f"{val:.4f}" in paper_text, (
            f"§7.1.1 must report Mamba random_split mean = {val:.4f}"
        )

    def test_spiking_random_split_mean_in_paper(self, paper_text: str) -> None:
        m = _v100_ablation_means_by_arch()
        if "spiking_expand2" not in m:
            pytest.skip("V100 ablation cells not yet aggregated")
        val = round(m["spiking_expand2"], 4)
        assert f"{val:.4f}" in paper_text, (
            f"§7.1.1 must report Spiking-SSM random_split mean = {val:.4f}"
        )

    def test_lstm_delta_in_paper(self, paper_text: str) -> None:
        m_v100 = _v100_ablation_means_by_arch()
        m_4080 = _phase5_iid_fedavg_means_by_arch()
        if "lstm" not in m_v100 or "lstm" not in m_4080:
            pytest.skip("required artifacts not yet present")
        delta = m_v100["lstm"] - m_4080["lstm"]
        # paper may use ASCII hyphen-minus '-' or Unicode minus '−'
        # (typeset minus). Accept either.
        ascii_signed = f"{delta:+.4f}"
        unicode_signed = ascii_signed.replace("-", "−")
        assert (ascii_signed in paper_text
                or unicode_signed in paper_text
                or ascii_signed.replace("+", "") in paper_text
                or unicode_signed.replace("+", "") in paper_text), (
            f"§7.1.1 must report LSTM Δ = {ascii_signed} or {unicode_signed} "
            f"(computed from V100 ablation vs Phase 5 IID FedAvg)"
        )


# ============================================================================
# §7.1.4 hardware drift → V100 random_split vs 4080 α=5.0 (FedAvg)
# ============================================================================

def _phase5_alpha5_fedavg_means_by_arch() -> dict[str, float]:
    if not AGG_PHASE5.exists():
        return {}
    s = json.loads(AGG_PHASE5.read_text())
    out: dict[str, float] = {}
    for k, v in s["stats"].items():
        parts = k.split("::")
        if len(parts) != 4:
            continue
        arch, algo, pmode, alpha_tag = parts
        if algo == "fedavg" and pmode == "dirichlet" and alpha_tag == "a5p00":
            out[arch] = v["test_auc_mean"]
    return out


class TestSection714HardwareDriftBound:

    def test_drift_bounds_match_recomputed(self, paper_text: str) -> None:
        m_v100 = _v100_ablation_means_by_arch()
        m_4080_a5 = _phase5_alpha5_fedavg_means_by_arch()
        if not m_v100 or not m_4080_a5:
            pytest.skip("required artifacts not yet present")
        for arch in ("lstm", "mamba", "spiking_expand2"):
            if arch not in m_v100 or arch not in m_4080_a5:
                continue
            drift = abs(m_v100[arch] - m_4080_a5[arch])
            drift_str = f"{drift:.4f}"
            # paper §7.1.4 must mention this exact drift (or rounded to 3 places)
            drift_alt = f"{drift:.3f}"
            assert drift_str in paper_text or drift_alt in paper_text, (
                f"§7.1.4 must report {arch} hardware drift = {drift_str} "
                f"(V100 random_split mean − 4080 Dirichlet α=5.0 mean)"
            )

    def test_no_two_orders_overclaim(self, paper_text: str) -> None:
        """Drift bound 0.007 vs signal 0.18 = 25× ratio. 'Two orders of
        magnitude' would mean 100×; 'an order of magnitude' is 10×.
        25× ≈ 'an order of magnitude' is honest; 'two orders' is overclaim."""
        forbidden = ["two orders of magnitude smaller",
                     "two orders of magnitude larger"]
        for s in forbidden:
            assert s not in paper_text, (
                f"§7.1.4 / §7.1.1 must not claim 'two orders of magnitude' "
                f"(actual ratio ≈ 25× = an order of magnitude, not two)"
            )


# ============================================================================
# License consistency (ADR-001 D-15)
# ============================================================================

class TestLicenseConsistency:

    def test_apache_in_paper_and_no_mit(self, paper_text: str) -> None:
        assert "Apache-2.0" in paper_text
        assert "MIT license" not in paper_text
        assert "MIT/Apache-2.0" not in paper_text


# ============================================================================
# Self-test of this hook itself
# ============================================================================

class TestHookFixturesValid:

    def test_step1_fixture_loads(self) -> None:
        if STEP1.exists():
            s = json.loads(STEP1.read_text())
            assert "Q3_global_train_pos_rate" in s
            assert "Q1_n_continuous_features" in s

    def test_step2_fixture_loads(self) -> None:
        if STEP2.exists():
            s = json.loads(STEP2.read_text())
            assert "C1_per_feature_pairwise_bs_KL" in s
