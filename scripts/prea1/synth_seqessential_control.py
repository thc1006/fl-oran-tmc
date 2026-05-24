"""Synthetic control: does the fragmentation AUC gap track sequence-essentiality (Delta_seq)?

Isolates sequence-essentiality as the causal driver, with every other factor held fixed.

Data: M multivariate AR(1) "runs" (synthetic channel-state, d channels, autocorr phi).
Target: precomputed per real step tau from the TRUE consecutive trajectory, with a tunable
knob lambda in [0,1]:
    driver(tau) = (1-lambda) * z[f0_tau]            (instantaneous level)
                +   lambda   * z[f0_tau - f0_{tau-4}] (4-step trajectory difference)
    y_tau = 1[ driver(tau) > median(driver) ]        (=> pos-rate ~ 0.5, no imbalance confound)
Then the SAME partition-then-window pipeline as the real experiments builds windows
[i..i+SEQ-1] with label = y at the NEXT row (idx[i+SEQ]) in the client's sorted order
(so the label is never inside the window -> no leak), and compares intact (run-Dirichlet,
whole runs -> one client) vs row-Dirichlet (rows scattered -> fragmented windows).

Per lambda we measure:
    Delta_seq = AUC_intact_LSTM - AUC_singlestep_logreg   (how much the trajectory adds)
    gap        = AUC_intact_LSTM - AUC_rowDirichlet_LSTM   (what fragmentation destroys)
Prediction: gap is a monotone (~linear-through-origin) function of Delta_seq.
  lambda=0 -> driver ~ instantaneous; AR makes f0_{tau} predict f0_{tau+1}, so single-step
             suffices -> Delta_seq ~ 0 -> gap ~ 0.
  lambda=1 -> driver needs two window points (f0_last and f0_{last-3}); single-step cannot
             form the difference, intact window can, scattered cannot -> Delta_seq high, gap high.

Deterministic (seeded RNG, no hash()); local-safe (small, sequential, single GPU).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEQ, NCLI, ROUNDS, LSTEPS, BATCH, LR = 5, 8, 30, 30, 128, 5e-4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen_runs(M, T, d, phi, seed):
    rng = np.random.default_rng(seed)
    runs = []
    s = np.sqrt(1.0 - phi * phi)
    for _ in range(M):
        x = np.empty((T, d), np.float32)
        x[0] = rng.standard_normal(d)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + s * rng.standard_normal(d)
        runs.append(x)
    return runs


def build_df(runs, lam):
    parts = []
    for ri, x in enumerate(runs):
        T = len(x)
        df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
        inst = x[:, 0].astype(np.float64)
        traj = np.zeros(T, np.float64)
        traj[4:] = x[4:, 0] - x[:-4, 0]
        df["_inst"], df["_traj"] = inst, traj
        df["run_uid"] = f"run{ri}"
        df["ue"] = f"run{ri}"           # entity == run for the synthetic
        df["step"] = np.arange(T)
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    zi = (full["_inst"] - full["_inst"].mean()) / (full["_inst"].std() + 1e-9)
    zt = (full["_traj"] - full["_traj"].mean()) / (full["_traj"].std() + 1e-9)
    drv = (1 - lam) * zi + lam * zt
    full["y"] = (drv > drv.median()).astype(np.float32)
    return full


def make_windows(df, feats, mu, sd):
    """label = y at the row AFTER the window (idx[i+SEQ]); label never inside X."""
    F = (df[feats].to_numpy(np.float32) - mu) / sd
    y = df["y"].to_numpy(np.float32)
    Xs, ys = [], []
    for _k, idx in df.groupby(["run_uid", "ue"], observed=True).indices.items():
        idx = np.sort(idx)
        if len(idx) < SEQ + 1:
            continue
        fb, yb = F[idx], y[idx]
        for i in range(len(idx) - SEQ):
            Xs.append(fb[i:i + SEQ])
            ys.append(yb[i + SEQ])
    if not Xs:
        return np.zeros((0, SEQ, len(feats)), np.float32), np.zeros(0, np.float32)
    return np.stack(Xs), np.asarray(ys, np.float32)


def assign(df, mode, n, rng, alpha=1.0):
    if mode == "run_dirichlet":      # intact: whole (run,ue) groups -> clients
        groups = list(df.groupby(["run_uid", "ue"], observed=True).indices.items())
        order = rng.permutation(len(groups))
        edges = (np.cumsum(rng.dirichlet([alpha] * n)) * len(groups)).astype(int)
        cof, ci = {}, 0
        for rank, gi in enumerate(order):
            while ci < n - 1 and rank >= edges[ci]:
                ci += 1
            cof[groups[gi][0]] = ci
        return np.array([cof[k] for k in zip(df["run_uid"], df["ue"])])
    if mode == "dirichlet":          # row-level: scatter rows -> fragmented windows
        return rng.choice(n, len(df), p=rng.dirichlet([alpha] * n))
    raise ValueError(mode)


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
                            for i in range(0, len(Xte), 8192)]) if len(Xte) else np.zeros(0)
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def single_step_auc(tr, te, feats):
    """logreg predicting y at NEXT row from the single current-step features (per group)."""
    def xy(d):
        g = d.groupby(["run_uid", "ue"], observed=True)
        nxt = g["y"].shift(-1)
        v = nxt.notna()
        return d.loc[v, feats].to_numpy(np.float64), nxt[v].to_numpy(np.float64)
    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=300).fit((Xtr - mu) / sd, ytr)
    return roc_auc_score(yte, lr.predict_proba((Xte - mu) / sd)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=300)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--phi", type=float, default=0.8)
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = gen_runs(args.M, args.T, args.d, args.phi, seed=0)   # data fixed across lambda
    feats = [f"f{i}" for i in range(args.d)]
    n_tr = int(0.7 * len(runs))

    print(f"[synth] M={args.M} T={args.T} d={args.d} phi={args.phi} | feats={feats} | "
          f"train_runs={n_tr} test_runs={len(runs) - n_tr} | SEQ={SEQ} NCLI={NCLI} ROUNDS={ROUNDS}")
    print(f"{'lambda':>7} | {'pos':>5} | {'single':>7} {'intact':>7} {'row':>7} | "
          f"{'Delta_seq':>9} {'gap':>8}")
    out = []
    for lam in lambdas:
        df = build_df(runs, lam)
        tr_ids = {f"run{i}" for i in range(n_tr)}
        tr = df[df["run_uid"].isin(tr_ids)].reset_index(drop=True)
        te = df[~df["run_uid"].isin(tr_ids)].reset_index(drop=True)
        mu = tr[feats].to_numpy(np.float32).mean(0)
        sd = tr[feats].to_numpy(np.float32).std(0) + 1e-6
        Xte, yte = make_windows(te, feats, mu, sd)
        ss = single_step_auc(tr, te, feats)
        intact_aucs, row_aucs = [], []
        for seed in seeds:
            for mode, bucket in (("run_dirichlet", intact_aucs), ("dirichlet", row_aucs)):
                rng = np.random.default_rng(seed)
                cl = assign(tr, mode, NCLI, rng)
                cxy = [make_windows(tr[cl == c], feats, mu, sd)
                       for c in range(NCLI) if (cl == c).sum() >= SEQ + 1]
                bucket.append(fedavg(cxy, Xte, yte, seed))
        intact, row = float(np.nanmean(intact_aucs)), float(np.nanmean(row_aucs))
        d_seq, gap = intact - ss, intact - row
        print(f"{lam:>7.2f} | {yte.mean():>5.2f} | {ss:>7.4f} {intact:>7.4f} {row:>7.4f} | "
              f"{d_seq:>+9.4f} {gap:>+8.4f}")
        out.append({"lambda": lam, "test_pos": float(yte.mean()), "single_step": round(ss, 4),
                    "intact": round(intact, 4), "row": round(row, 4),
                    "delta_seq": round(d_seq, 4), "gap": round(gap, 4),
                    "intact_seeds": [round(x, 4) for x in intact_aucs],
                    "row_seeds": [round(x, 4) for x in row_aucs]})
    # correlation of gap vs delta_seq across lambda
    ds = np.array([r["delta_seq"] for r in out])
    gp = np.array([r["gap"] for r in out])
    rho = float(np.corrcoef(ds, gp)[0, 1]) if len(ds) > 1 else float("nan")
    print(f"\nSpearman-ish Pearson(gap, Delta_seq) = {rho:+.3f}  (expect strongly positive)")
    res = {"config": vars(args), "rows": out, "pearson_gap_vs_deltaseq": round(rho, 3)}
    from pathlib import Path
    Path("artifacts/prea1/twinning/synth_deltaseq_control.json").write_text(json.dumps(res, indent=2))
    print("WROTE artifacts/prea1/twinning/synth_deltaseq_control.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done] {time.time() - t0:.0f}s")
