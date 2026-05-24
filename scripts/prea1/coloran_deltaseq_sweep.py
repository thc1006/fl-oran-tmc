"""ColO-RAN Delta_seq controlled sweep (PREREG-A2 money figure).

Tests H1-H3: across a target family spanning sequence-essentiality, the fragmentation
AUC gap (intact - row-level) tracks Delta_seq (= AUC_intact_LSTM - AUC_singlestep_logreg).

Reuses the VALIDATED ColO-RAN machinery (no reimplementation, per ADR D-3 / hard rule 1):
  - fl_oran.data_v2.partition.partition_clients  (iid / run_dirichlet / dirichlet / random_split)
  - fl_oran.data_v2.sequences.build_run_sequences (partition-THEN-window per (run_id, slice_id))
  - fl_oran.data_v2.split.ood_split_by_tr         (OOD-by-tr train/test)
and the SAME minimal LSTM + FedAvg as the synthetic + Twinning experiments, so the
gap-vs-Delta_seq points are directly comparable across all three datasets.

Each target: source column(s) dropped from the model input (no label leak); label strictly
t+k. intact = mean(iid natural-by-BS, run_dirichlet); row = mean(dirichlet, random_split),
alpha=1.0. Delta_seq via a pooled single-step logreg on the last window step.

V100 (heavy): python coloran_deltaseq_sweep.py --keep-run-frac 0.3 --seeds 0,1,2,3,4
smoke:        python coloran_deltaseq_sweep.py --smoke
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fl_oran.training.centralized_v3 import V3_CONTINUOUS
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.partition import partition_clients
from fl_oran.data_v2.split import ood_split_by_tr

SEQ, NCLI, ROUNDS, LSTEPS, BATCH, LR = 5, 8, 50, 50, 128, 5e-4
TRAIN_TR, TEST_TR = list(range(22)), [25, 26, 27]
PARQ = "data/coloran_raw_unified.parquet"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INTACT, ROW = ("iid", "run_dirichlet"), ("dirichlet", "random_split")

# target family spanning Delta_seq (PREREG-A2); drop = source column(s) removed from features
TARGETS = [
    {"name": "bler_th05",   "col": "ul_bler", "cmp": "gt", "th": 0.05, "k": 1, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_th10",   "col": "ul_bler", "cmp": "gt", "th": 0.10, "k": 1, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_th20",   "col": "ul_bler", "cmp": "gt", "th": 0.20, "k": 1, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_k2",     "col": "ul_bler", "cmp": "gt", "th": 0.10, "k": 2, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_k3",     "col": "ul_bler", "cmp": "gt", "th": 0.10, "k": 3, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_k5",     "col": "ul_bler", "cmp": "gt", "th": 0.10, "k": 5, "drop": ["ul_bler", "dl_bler"]},
    {"name": "bler_trend5", "col": "ul_bler", "cmp": "gt", "th": 0.10, "k": 1, "smooth": 5, "drop": ["ul_bler", "dl_bler"]},
    {"name": "cqi_med",     "col": "dl_cqi",  "cmp": "lt", "q": 0.5, "k": 1, "drop": ["dl_cqi"]},
    {"name": "mcs_med",     "col": "dl_mcs",  "cmp": "lt", "q": 0.5, "k": 1, "drop": ["dl_mcs", "ul_mcs"]},
    # NOTE: ul_sinr dropped pre-results (PREREG-A2 amendment): median=0 with 64% of rows ==0,
    # so 1[ul_sinr_{t+1} < median] has pos-rate 0 -> AUC undefined. Technical exclusion, not
    # a result-based cherry-pick (its gap was never observed).
    {"name": "buffer_med",  "col": "dl_buffer_bytes", "cmp": "gt", "q": 0.5, "k": 1, "drop": ["dl_buffer_bytes", "ul_buffer_bytes"]},
    {"name": "brate_med",   "col": "tx_brate_dl_Mbps", "cmp": "lt", "q": 0.5, "k": 1, "drop": ["tx_brate_dl_Mbps"]},
]


def md5f(s, mod=1000):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % mod


def build_target(df, spec, train_tr):
    """Add y_sla_next = 1[ (smoothed) col_{t+k} {cmp} threshold ] per (run_id, slice_id);
    quantile threshold computed on TRAIN-tr rows only. Drops the last k rows of each group."""
    df = df.sort_values(["run_id", "slice_id", "step_idx"]).reset_index(drop=True)
    col = df[spec["col"]].astype("float64")
    if spec.get("smooth", 1) > 1:
        col = df.groupby(["run_id", "slice_id"], observed=True)[spec["col"]].transform(
            lambda s: s.rolling(spec["smooth"], min_periods=1).mean()).astype("float64")
    df["_c"] = col
    if "q" in spec:
        thr = float(df.loc[df["tr"].isin(train_tr), "_c"].quantile(spec["q"]))
    else:
        thr = float(spec["th"])
    fut = df.groupby(["run_id", "slice_id"], observed=True)["_c"].shift(-spec["k"])
    y = (fut > thr) if spec["cmp"] == "gt" else (fut < thr)
    df["y_sla_next"] = y.astype("float32")
    df.loc[fut.isna(), "y_sla_next"] = np.nan
    df = df.dropna(subset=["y_sla_next"]).drop(columns=["_c"]).reset_index(drop=True)
    return df, thr


def std_windows(X, mu, sd):
    return ((X - mu[None, None, :]) / sd[None, None, :]).astype(np.float32)


class LSTMClf(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.lstm = nn.LSTM(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


def fedavg(client_xy, Xte, yte, seed):
    torch.manual_seed(seed)
    f = Xte.shape[2]
    g = LSTMClf(f).to(DEV)
    loss_fn = nn.BCEWithLogitsLoss()
    clients = [(torch.tensor(x, device=DEV), torch.tensor(y, device=DEV))
               for x, y in client_xy if len(x) >= BATCH]
    if not clients:
        return float("nan")
    for _r in range(ROUNDS):
        states, sizes = [], []
        for X, y in clients:
            lc = LSTMClf(f).to(DEV)
            lc.load_state_dict(g.state_dict())
            opt = torch.optim.Adam(lc.parameters(), lr=LR)
            for _s in range(LSTEPS):
                bi = torch.randint(0, len(X), (BATCH,), device=DEV)
                opt.zero_grad()
                loss_fn(lc(X[bi]), y[bi]).backward()
                opt.step()
            states.append(lc.state_dict())
            sizes.append(len(X))
        w = np.array(sizes) / sum(sizes)
        g.load_state_dict({k: sum(w[i] * states[i][k] for i in range(len(states))) for k in states[0]})
    g.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(g(torch.tensor(Xte[i:i + 8192], device=DEV))).cpu().numpy()
                            for i in range(0, len(Xte), 8192)])
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def central_auc(Xtr, ytr, Xte, yte, seed, n_steps=2000, shuffle=False):
    """One centralized LSTM (pooled, no FL), fixed step budget. Used capacity-matched at
    seq_len=5 (full) vs seq_len=1 (last step only) so Delta_seq isolates the SEQUENCE value,
    not the linear-vs-nonlinear capacity gap (which fragmentation does not destroy).

    shuffle=True: independently permute each window's time axis (order DESTROYED, multiset /
    run-rate PRESERVED), same LSTM -> AUC_shuffle is the order-FREE ceiling; seq - shuffle =
    Delta_traj isolates the TRAJECTORY value (the partition-vulnerable part), separating it
    from the partition-invariant run-rate that a persistent target carries (post-hoc refinement
    motivated by the brate_med deviation; see autocorr: brate 0.98 persistent vs bler 0.02 white)."""
    if shuffle:
        rng = np.random.default_rng(seed + 777)

        def _shuf(X):
            n, ln = X.shape[0], X.shape[1]
            p = rng.permuted(np.tile(np.arange(ln), (n, 1)), axis=1)
            return X[np.arange(n)[:, None], p]

        Xtr, Xte = _shuf(Xtr), _shuf(Xte)
    torch.manual_seed(seed)
    m = LSTMClf(Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=DEV)
    Yt = torch.tensor(ytr, device=DEV)
    for _ in range(n_steps):
        bi = torch.randint(0, len(Xt), (BATCH,), device=DEV)
        opt.zero_grad()
        loss_fn(m(Xt[bi]), Yt[bi]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(m(torch.tensor(Xte[i:i + 8192], device=DEV))).cpu().numpy()
                            for i in range(0, len(Xte), 8192)])
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def run_target(df, spec, feats, seeds):
    df2, thr = build_target(df, spec, TRAIN_TR)
    sp = ood_split_by_tr(df2, TRAIN_TR, [22, 23, 24], TEST_TR)
    mu = sp.train[feats].to_numpy(np.float32).mean(0)
    sd = sp.train[feats].to_numpy(np.float32).std(0) + 1e-6
    Xte, Yte = build_run_sequences(sp.test, feats, ["y_sla_next"], seq_len=SEQ)
    Xte, yte = std_windows(Xte, mu, sd), Yte[:, 0]
    assert spec["col"] not in feats and len(Xte) and 0.005 < yte.mean() < 0.995, \
        f"sanity fail: leak/empty/posrate ({spec['name']}, pos={yte.mean():.3f})"
    # Delta_seq (capacity-matched): centralized LSTM on the full seq vs the SAME LSTM on a
    # length-1 window (last step only). Both centralized, same arch -> isolates SEQUENCE value.
    Xtr5, Ytr5 = build_run_sequences(sp.train, feats, ["y_sla_next"], seq_len=SEQ)
    Xtr5, ytr5 = std_windows(Xtr5, mu, sd), Ytr5[:, 0]
    seq_c = central_auc(Xtr5, ytr5, Xte, yte, seed=0)
    inst = central_auc(Xtr5[:, -1:, :], ytr5, Xte[:, -1:, :], yte, seed=0)
    single, delta_seq = inst, seq_c - inst
    aucs = {m: [] for m in INTACT + ROW}
    for seed in seeds:
        for mode in INTACT + ROW:
            kw = dict(seed=seed)
            if mode in ("dirichlet", "run_dirichlet"):
                kw.update(alpha=1.0, n_clients=NCLI)
            elif mode == "random_split":
                kw.update(n_clients=NCLI)
            cdfs = partition_clients(sp.train, mode=mode, **kw)
            cxy = []
            for d in cdfs.values():
                Xc, Yc = build_run_sequences(d, feats, ["y_sla_next"], seq_len=SEQ)
                if len(Xc):
                    cxy.append((std_windows(Xc, mu, sd), Yc[:, 0]))
            aucs[mode].append(fedavg(cxy, Xte, yte, seed))
    mean = {m: float(np.nanmean(aucs[m])) for m in aucs}
    intact = float(np.nanmean([mean[m] for m in INTACT]))
    row = float(np.nanmean([mean[m] for m in ROW]))
    return {"name": spec["name"], "thr": round(thr, 4), "pos": round(float(yte.mean()), 3),
            "n_feats": len(feats), "single_step": round(single, 4), "seq_central": round(seq_c, 4),
            "intact": round(intact, 4), "row": round(row, 4),
            "delta_seq": round(delta_seq, 4), "gap": round(intact - row, 4),
            "per_mode": {m: round(mean[m], 4) for m in mean},
            "per_mode_seeds": {m: [round(x, 4) for x in aucs[m]] for m in aucs}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-run-frac", type=float, default=0.3, help="fraction of run_ids kept (whole runs)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--targets", default="", help="comma names subset; default all")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parquet", default=PARQ, help="ColO-RAN unified parquet path")
    ap.add_argument("--out", default="", help="output json path (default auto); set per-GPU to avoid clobber")
    args = ap.parse_args()
    global ROUNDS
    ROUNDS = args.rounds
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(args.targets.split(",")) if args.targets else None
    targets = [t for t in TARGETS if (want is None or t["name"] in want)]
    if args.smoke:
        targets = [t for t in targets if t["name"] in ("bler_th10", "mcs_med", "cqi_med")]
        seeds = [0]
        args.keep_run_frac = min(args.keep_run_frac, 0.08)

    keep_cols = sorted(set(V3_CONTINUOUS) | {"run_id", "slice_id", "step_idx", "tr", "bs_id"})
    df = pd.read_parquet(args.parquet, columns=keep_cols)
    runs = df["run_id"].unique()
    keep = {r for r in runs if md5f(r) < int(args.keep_run_frac * 1000)}
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    print(f"[load] {len(df):,} rows, {len(keep)}/{len(runs)} runs kept "
          f"(frac={args.keep_run_frac}) | targets={[t['name'] for t in targets]} seeds={seeds} ROUNDS={ROUNDS}")
    print(f"{'target':>11} | {'pos':>5} {'feats':>5} | {'single':>7} {'intact':>7} {'row':>7} | "
          f"{'Delta_seq':>9} {'gap':>8}")
    out = []
    for spec in targets:
        feats = [c for c in V3_CONTINUOUS if c not in spec["drop"]]
        r = run_target(df, spec, feats, seeds)
        out.append(r)
        print(f"{r['name']:>11} | {r['pos']:>5} {r['n_feats']:>5} | {r['single_step']:>7.4f} "
              f"{r['intact']:>7.4f} {r['row']:>7.4f} | {r['delta_seq']:>+9.4f} {r['gap']:>+8.4f}", flush=True)
    ds = np.array([r["delta_seq"] for r in out])
    gp = np.array([r["gap"] for r in out])
    rho = float(np.corrcoef(ds, gp)[0, 1]) if len(ds) > 1 else float("nan")
    # rank correlation (Spearman) without scipy
    def rank(a):
        o = np.argsort(np.argsort(a)); return o
    srho = float(np.corrcoef(rank(ds), rank(gp))[0, 1]) if len(ds) > 1 else float("nan")
    print(f"\nPearson(gap, Delta_seq)={rho:+.3f}  Spearman={srho:+.3f}  (H1: Spearman>0.8)")
    res = {"config": vars(args), "targets": out,
           "pearson_gap_deltaseq": round(rho, 3), "spearman_gap_deltaseq": round(srho, 3)}
    Path("artifacts/prea1/twinning").mkdir(parents=True, exist_ok=True)
    op = args.out or f"artifacts/prea1/twinning/coloran_deltaseq{'_smoke' if args.smoke else ''}.json"
    Path(op).write_text(json.dumps(res, indent=2))
    print(f"WROTE {op}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done] {time.time() - t0:.0f}s")
