"""DESCRIPTIVE bs-heterogeneity shift analysis (covariate vs concept).

SECONDARY / descriptive only -- NOT a pass/fail gate.

Question: is base-station (bs_id) heterogeneity primarily
  - COVARIATE shift  : P(KPI) differs across bs, P(SLA|KPI) shared, OR
  - CONCEPT  shift   : P(SLA|KPI) itself differs across bs.

Method (no torch / no FL / no deep model trained):
  (2) covariate-shift magnitude  = domain classifier  bs_id <- 17 KPIs
                                    (balanced accuracy vs 1/7 chance).
  (3) concept-shift magnitude    = ΔNLL between
                                       model_A = P(SLA | KPI)            and
                                       model_B = P(SLA | KPI, bs onehot)
                                    on held-out CV folds, with bootstrap CI95.
                                    NLL (log-loss) is the PRIMARY metric;
                                    ΔBrier is a secondary check.
  (4) placebo floor              = relabel SLA from model_A's P(SLA|KPI)
                                    (bs ignored by construction) and recompute
                                    ΔNLL -> noise floor, should be ≈0.
  (5) sensitivity                = logistic vs lightgbm; full vs
                                    KPI-stratified (covariate-controlled) ΔNLL.

Fits/evaluation use ONLY the OOD-by-tr TRAIN split (tr in 0..21) to avoid
leakage of the held-out tr configs.

Subsampling: the full parquet is 18.3M rows. engineer_features needs intact
(run_id, slice_id) groups for the t+1 shift/rolling. So we subsample WHOLE
run_ids (fixed seed) BEFORE feature engineering, keeping each run's temporal
structure valid. Default target ~2.5M engineered TRAIN rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from fl_oran.data_v2.features import CLEAN_FEATURES, engineer_features

SEED = 1006
N_RUNS_SAMPLE = 600          # whole run_ids sampled (of 3080) -> ~ a few M rows
N_FOLDS = 5
N_BOOT = 2000
OUT = Path(__file__).resolve().parent
PARQUET = Path("/home/thc1006/dev/fl-oran-tmc/data/coloran_raw_unified.parquet")

CATEGORICAL_KEYS = ["bs_id", "slice_id", "sched", "tr"]
TREND_FEATURES = ["tx_brate_dl_roll3", "tx_brate_dl_volatility"]
# 17 continuous KPI features = CLEAN_FEATURES - categorical keys - engineered trend.
KPI_FEATURES = [c for c in CLEAN_FEATURES if c not in CATEGORICAL_KEYS + TREND_FEATURES]
TARGET = "y_sla_violation_next"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_train_split() -> pd.DataFrame:
    """Sample whole run_ids, engineer features, keep TRAIN tr in 0..21."""
    rng = np.random.default_rng(SEED)
    cols_needed = sorted(set(
        ["run_id", "step_idx", "slice_id", "sched", "tr", "bs_id"]
        + ["tx_brate_dl_Mbps", "rx_brate_ul_Mbps", "tx_pkts_dl", "rx_pkts_ul",
           "dl_buffer_bytes", "ul_buffer_bytes", "dl_bler", "ul_bler",
           "sum_requested_prbs", "sum_granted_prbs", "num_ues", "slice_prb",
           "dl_mcs", "ul_mcs", "dl_cqi", "ul_sinr", "ul_rssi"]
    ))
    log("reading run_id + tr index ...")
    idx = pd.read_parquet(PARQUET, columns=["run_id", "tr"])
    # restrict to TRAIN tr BEFORE sampling so all sampled runs are train-side
    train_runs = idx.loc[idx["tr"] <= 21, "run_id"].unique()
    log(f"train-side runs: {len(train_runs)} (of {idx['run_id'].nunique()})")
    n = min(N_RUNS_SAMPLE, len(train_runs))
    chosen = rng.choice(train_runs, size=n, replace=False)
    chosen_set = set(chosen.tolist())
    log(f"sampling {n} whole run_ids (seed={SEED}) ...")

    df = pd.read_parquet(PARQUET, columns=cols_needed)
    df = df[df["run_id"].isin(chosen_set)].reset_index(drop=True)
    log(f"raw rows in sampled runs: {len(df):,}")
    df = engineer_features(df)
    df = df[df["tr"] <= 21].reset_index(drop=True)   # belt-and-suspenders
    log(f"engineered TRAIN rows: {len(df):,}  pos_rate={df[TARGET].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# (2) covariate shift: domain classifier  bs_id <- KPI
# ---------------------------------------------------------------------------
def covariate_shift(df: pd.DataFrame) -> dict:
    log("=== (2) covariate-shift domain classifier bs_id <- 17 KPIs ===")
    X = df[KPI_FEATURES].to_numpy(np.float32)
    y = df["bs_id"].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = {}
    for name, mk in (
        ("logistic", lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, C=1.0, n_jobs=4))),
        ("lightgbm", lambda: lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            objective="multiclass", n_jobs=4, random_state=SEED, verbose=-1)),
    ):
        baccs = []
        for tr_i, te_i in skf.split(X, y):
            clf = mk()
            clf.fit(X[tr_i], y[tr_i])
            pred = clf.predict(X[te_i])
            baccs.append(balanced_accuracy_score(y[te_i], pred))
        baccs = np.array(baccs)
        out[name] = {
            "balanced_acc_mean": float(baccs.mean()),
            "balanced_acc_std": float(baccs.std(ddof=1)),
            "balanced_acc_folds": baccs.round(4).tolist(),
            "chance_floor": 1.0 / df["bs_id"].nunique(),
        }
        log(f"  {name}: balanced_acc={baccs.mean():.4f} +/- {baccs.std(ddof=1):.4f} "
            f"(chance={1.0/df['bs_id'].nunique():.4f})")
    return out


# ---------------------------------------------------------------------------
# concept shift helpers
# ---------------------------------------------------------------------------
def _onehot_bs(df: pd.DataFrame) -> np.ndarray:
    bs = df["bs_id"].to_numpy()
    cats = np.sort(df["bs_id"].unique())
    return np.stack([(bs == c).astype(np.float32) for c in cats], axis=1)


def _fit_predict_oof(X: np.ndarray, y: np.ndarray, kind: str,
                     folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Out-of-fold P(SLA=1) predictions over the SAME fold assignment."""
    p = np.zeros(len(y), dtype=np.float64)
    for tr_i, te_i in folds:
        if kind == "logistic":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, C=1.0, n_jobs=4))
        else:
            clf = lgb.LGBMClassifier(
                n_estimators=400, num_leaves=63, learning_rate=0.05,
                objective="binary", n_jobs=4, random_state=SEED, verbose=-1)
        clf.fit(X[tr_i], y[tr_i])
        p[te_i] = clf.predict_proba(X[te_i])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def _bootstrap_delta(per_row_a: np.ndarray, per_row_b: np.ndarray,
                     n_boot: int, rng: np.random.Generator) -> dict:
    """Bootstrap CI95 of mean(per_row_a) - mean(per_row_b) (paired by row)."""
    diff = per_row_a - per_row_b
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta_mean": float(diff.mean()),
            "ci95_lo": float(lo), "ci95_hi": float(hi)}


def _per_row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _per_row_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def concept_shift(df: pd.DataFrame, kind: str, label: str,
                  rng: np.random.Generator) -> dict:
    """ΔNLL = NLL(model_A: SLA|KPI) - NLL(model_B: SLA|KPI,bs).

    Positive ΔNLL => bs adds conditional info beyond KPI => concept shift.
    Also returns model_A OOF predictions (used for the placebo).
    """
    log(f"=== (3) concept-shift [{label}] ({kind}) ===")
    Xk = df[KPI_FEATURES].to_numpy(np.float32)
    bs1h = _onehot_bs(df)
    Xkb = np.concatenate([Xk, bs1h], axis=1)
    y = df[TARGET].to_numpy(np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(Xk, y))

    pA = _fit_predict_oof(Xk, y, kind, folds)
    pB = _fit_predict_oof(Xkb, y, kind, folds)

    nll_a = _per_row_logloss(y, pA)
    nll_b = _per_row_logloss(y, pB)
    br_a = _per_row_brier(y, pA)
    br_b = _per_row_brier(y, pB)

    dnll = _bootstrap_delta(nll_a, nll_b, N_BOOT, rng)
    dbri = _bootstrap_delta(br_a, br_b, N_BOOT, rng)
    log(f"  NLL_A={nll_a.mean():.5f} NLL_B={nll_b.mean():.5f} "
        f"ΔNLL={dnll['delta_mean']:.5e} CI95=[{dnll['ci95_lo']:.2e},{dnll['ci95_hi']:.2e}]")
    log(f"  Brier_A={br_a.mean():.5f} Brier_B={br_b.mean():.5f} "
        f"ΔBrier={dbri['delta_mean']:.5e}")
    return {
        "kind": kind, "label": label,
        "nll_A": float(nll_a.mean()), "nll_B": float(nll_b.mean()),
        "delta_nll": dnll,
        "brier_A": float(br_a.mean()), "brier_B": float(br_b.mean()),
        "delta_brier": dbri,
        "n_rows": int(len(y)),
        "_pA_oof": pA,         # internal, stripped before json dump
    }


# ---------------------------------------------------------------------------
# (4) placebo: relabel SLA from model_A's P(SLA|KPI) (bs ignored)
# ---------------------------------------------------------------------------
def placebo_floor(df: pd.DataFrame, pA: np.ndarray, kind: str,
                  rng: np.random.Generator) -> dict:
    log(f"=== (4) placebo (concept-homogenized) ({kind}) ===")
    # Sample new labels from model_A's per-row P(SLA|KPI); bs is conditionally
    # independent of the new label given KPI by construction => ΔNLL≈0 floor.
    y_pl = (rng.random(len(pA)) < pA).astype(np.float64)
    dfp = df.copy()
    dfp[TARGET] = y_pl
    res = concept_shift(dfp, kind, f"placebo[{kind}]", rng)
    res.pop("_pA_oof", None)
    res["placebo_pos_rate"] = float(y_pl.mean())
    return res


# ---------------------------------------------------------------------------
# (3b/5) covariate-controlled: KPI-stratified ΔNLL
# ---------------------------------------------------------------------------
def kpi_stratified_concept(df: pd.DataFrame, kind: str,
                           rng: np.random.Generator, n_bins: int = 8) -> dict:
    """Control covariate shift via coarse KPI stratification.

    Bin rows into KPI strata (quantile bins on the top-3 KPI principal-ish
    drivers: dl_cqi, ul_sinr, ul_rssi -- the physical-quality KPIs), then run
    the SAME ΔNLL within strata that contain >=2 bs and enough rows. This
    forces model_A and model_B comparison on overlapping KPI support.
    """
    log(f"=== (3b/5) KPI-stratified concept-shift ({kind}, {n_bins} bins) ===")
    strat_kpis = ["dl_cqi", "ul_sinr", "ul_rssi"]
    code = np.zeros(len(df), dtype=np.int64)
    for k in strat_kpis:
        try:
            b = pd.qcut(df[k], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            b = pd.cut(df[k], bins=n_bins, labels=False)
        b = b.fillna(-1).astype(np.int64).to_numpy()
        code = code * (n_bins + 1) + (b + 1)
    df = df.copy()
    df["_stratum"] = code

    # keep strata with >=2 bs present and >= 2000 rows
    g = df.groupby("_stratum")
    keep = g.filter(lambda x: x["bs_id"].nunique() >= 2 and len(x) >= 2000)
    cov = 1.0 - len(keep) / len(df)
    log(f"  strata kept rows={len(keep):,} ({100*len(keep)/len(df):.1f}% of train); "
        f"dropped {100*cov:.1f}% (single-bs / tiny strata)")
    if len(keep) < 5000:
        return {"note": "insufficient overlapping-support strata", "kind": kind}

    res = concept_shift(keep.reset_index(drop=True), kind,
                        f"kpi_stratified[{kind}]", rng)
    res.pop("_pA_oof", None)
    res["kept_row_frac"] = float(len(keep) / len(df))
    res["n_strata_kept"] = int(keep["_stratum"].nunique())
    return res


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_train_split()

    summary = {
        "meta": {
            "seed": SEED,
            "n_runs_sampled": N_RUNS_SAMPLE,
            "n_train_rows_engineered": int(len(df)),
            "pos_rate": float(df[TARGET].mean()),
            "kpi_features": KPI_FEATURES,
            "n_kpi_features": len(KPI_FEATURES),
            "categorical_keys": CATEGORICAL_KEYS,
            "trend_features_excluded": TREND_FEATURES,
            "n_bs": int(df["bs_id"].nunique()),
            "split": "OOD-by-tr TRAIN only (tr in 0..21)",
            "primary_metric": "delta_NLL (log-loss); delta_Brier secondary",
            "note": "DESCRIPTIVE / SECONDARY -- not a pass/fail gate. "
                    "Whole run_ids subsampled (seed) for memory; targets valid "
                    "because (run_id,slice_id) groups kept intact.",
        }
    }

    # (2) covariate shift
    summary["covariate_shift"] = covariate_shift(df)

    # (3) concept shift, two model families (sensitivity #1)
    concept = {}
    pA_by_kind = {}
    for kind in ("logistic", "lightgbm"):
        res = concept_shift(df, kind, f"full[{kind}]", rng)
        pA_by_kind[kind] = res.pop("_pA_oof")
        concept[f"full_{kind}"] = res

    # (4) placebo floor for each kind
    placebo = {}
    for kind in ("logistic", "lightgbm"):
        placebo[kind] = placebo_floor(df, pA_by_kind[kind], kind, rng)

    # (3b/5) covariate-controlled KPI-stratified (sensitivity #2)
    stratified = {}
    for kind in ("logistic", "lightgbm"):
        stratified[f"kpi_stratified_{kind}"] = kpi_stratified_concept(df, kind, rng)

    summary["concept_shift"] = concept
    summary["placebo_floor"] = placebo
    summary["concept_shift_kpi_stratified"] = stratified

    # ---- verdict logic (descriptive) ----
    bacc_lgb = summary["covariate_shift"]["lightgbm"]["balanced_acc_mean"]
    chance = summary["covariate_shift"]["lightgbm"]["chance_floor"]
    dnll_lgb = concept["full_lightgbm"]["delta_nll"]
    dnll_log = concept["full_logistic"]["delta_nll"]
    pl_lgb = placebo["lightgbm"]["delta_nll"]
    pl_log = placebo["logistic"]["delta_nll"]

    concept_real_lgb = dnll_lgb["ci95_lo"] > 0 and dnll_lgb["delta_mean"] > abs(pl_lgb["delta_mean"]) * 3
    concept_real_log = dnll_log["ci95_lo"] > 0 and dnll_log["delta_mean"] > abs(pl_log["delta_mean"]) * 3
    strat_ok = []
    for k, v in stratified.items():
        if "delta_nll" in v:
            strat_ok.append(v["delta_nll"]["ci95_lo"] > 0)

    summary["verdict"] = {
        "covariate_shift_strong": bool(bacc_lgb > 2 * chance),
        "covariate_balanced_acc_vs_chance": [bacc_lgb, chance],
        "concept_shift_significant_lightgbm": bool(concept_real_lgb),
        "concept_shift_significant_logistic": bool(concept_real_log),
        "concept_delta_nll_lightgbm": dnll_lgb,
        "concept_delta_nll_logistic": dnll_log,
        "placebo_floor_lightgbm": pl_lgb,
        "placebo_floor_logistic": pl_log,
        "kpi_stratified_still_significant": strat_ok,
        "interpretation": (
            "BOTH covariate and concept shift present"
            if (bacc_lgb > 2 * chance and (concept_real_lgb or concept_real_log))
            else ("primarily covariate (concept ~ placebo floor)"
                  if bacc_lgb > 2 * chance else "weak/ambiguous")
        ),
    }

    summary["meta"]["wall_time_sec"] = round(time.time() - t0, 1)
    out_json = OUT / "shift_analysis_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"WROTE {out_json}")
    log(f"VERDICT: {summary['verdict']['interpretation']}")


if __name__ == "__main__":
    main()
