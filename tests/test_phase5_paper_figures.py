"""Tests for scripts/phase5_paper_figures.py — paper-figure cell parser.

GH#9: extend parse_cell_name to handle Phase 6 + T-ABLATION cell-name
formats. The Phase 5 parser only knew ``iid_n<N>_s<seed>`` and
``dirichlet_a<alpha>_n<N>_s<seed>``; Phase 6 introduced new partition
modes (``per_bs_dirichlet``, ``random_split``) and an iid+threshold
suffix variant. Without parser support, those cells silently drop out
of paper figures.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "phase5_paper_figures.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase5_paper_figures", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase5_paper_figures"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fig():
    return _load_module()


# ---------------------------------------------------------------------------
# Existing partition modes (regression guard)
# ---------------------------------------------------------------------------

def test_parse_cell_name_iid_baseline(fig):
    """Standard Phase 5 iid cell (regression guard)."""
    out = fig.parse_cell_name("v7_lstm_fedavg_iid_n7_s42")
    assert out is not None
    assert out["arch"] == "lstm"
    assert out["algo"] == "fedavg"
    assert out["partition_mode"] == "iid"
    assert out["alpha"] is None
    assert out["n_clients"] == 7
    assert out["seed"] == 42


def test_parse_cell_name_dirichlet_baseline(fig):
    """Standard Phase 5 dirichlet cell (regression guard)."""
    out = fig.parse_cell_name("v7_mamba_fedadam_dirichlet_a0p50_n7_s0")
    assert out is not None
    assert out["arch"] == "mamba"
    assert out["algo"] == "fedadam"
    assert out["partition_mode"] == "dirichlet"
    assert out["alpha"] == pytest.approx(0.5)
    assert out["n_clients"] == 7
    assert out["seed"] == 0


# ---------------------------------------------------------------------------
# GH#9 new partition modes
# ---------------------------------------------------------------------------

def test_parse_cell_name_perbsdir_phase6_rank3(fig):
    """Phase 6 Rank 3 per-BS Dirichlet cell name has ``perbsdir_a<alpha>_s<seed>``
    (no ``n<N>`` token; the per-BS partition implies the structure)."""
    out = fig.parse_cell_name("v7_lstm_fedavg_perbsdir_a0p05_s0")
    assert out is not None, "perbsdir cell name must parse"
    assert out["arch"] == "lstm"
    assert out["algo"] == "fedavg"
    assert out["partition_mode"] == "per_bs_dirichlet"
    assert out["alpha"] == pytest.approx(0.05)
    assert out["seed"] == 0


def test_parse_cell_name_randsplit_t_ablation(fig):
    """T-ABLATION random_split cell ``randsplit_n<N>_s<seed>`` (no alpha)."""
    out = fig.parse_cell_name("v7_mamba_fedavg_randsplit_n7_s2")
    assert out is not None, "randsplit cell name must parse"
    assert out["arch"] == "mamba"
    assert out["algo"] == "fedavg"
    assert out["partition_mode"] == "random_split"
    assert out["alpha"] is None
    assert out["n_clients"] == 7
    assert out["seed"] == 2


def test_parse_cell_name_iid_with_threshold_phase6_rank1(fig):
    """Phase 6 Rank 1 threshold-sensitivity cell:
    ``iid_n<N>_s<seed>_t<thr>`` (e.g. _t05 = 5% BLER threshold)."""
    out = fig.parse_cell_name("v7_lstm_fedavg_iid_n7_s3_t15")
    assert out is not None, "iid+threshold cell name must parse"
    assert out["arch"] == "lstm"
    assert out["algo"] == "fedavg"
    assert out["partition_mode"] == "iid"
    assert out["n_clients"] == 7
    assert out["seed"] == 3
    assert out["threshold"] == pytest.approx(0.15), (
        f"_t15 must decode to 0.15 (15% BLER threshold); got {out.get('threshold')!r}"
    )


def test_parse_cell_name_malformed_returns_none(fig):
    """Malformed inputs still return None (regression guard for new
    partition modes that share prefix with valid ones)."""
    assert fig.parse_cell_name("v7_lstm_fedavg_perbsdir_BADALPHA_s0") is None
    assert fig.parse_cell_name("v7_lstm_fedavg_randsplit_BADN_s0") is None
    assert fig.parse_cell_name("not_a_v7_cell") is None
