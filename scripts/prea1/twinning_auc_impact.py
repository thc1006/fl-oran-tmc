"""Twinning AUC-impact: clean (non-leaky) replication of the sequence-integrity artifact.

Fixes the retracted leaky experiment:
  * NEXT-STEP target: window [t-SEQ+1 .. t] of channel-state -> predict an event at t+1
    (the old code took the label at the window's LAST row == nowcast of its own features).
  * target source column is DROPPED from the model input features (no copy of the label in X).
  * deterministic OOD-by-run split via md5 (the old code used salted hash() -> non-reproducible).
The partition-THEN-window mechanism (row-level partitions fragment per-client windows; intact
run/entity partitions keep each run's consecutive timesteps) is unchanged from the audited code.

Target = 1[<target_col>_{t+1} < quantile_q(train)]  (a next-step channel-quality-drop event,
the Twinning analog of ColO-RAN's 1[ul_bler_{t+1} > 0.10]).

CONFIRM (artifact replicates): intact (entity_ue/run_random/run_dirichlet) test AUC >> row-level
  (dirichlet/random_split), gap >> seed sigma. REFUTE: intact ~= row.

Usage (V100, ~/twinning_data):
  python twinning_auc_impact.py --target dl_cqi --smoke      # 1 seed, 4 modes (gate)
  python twinning_auc_impact.py --target dl_cqi              # 5 seeds, 5 modes (factorial)
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PARQ = Path("twinning_subset.parquet")
SEQ_LEN, N_CLIENTS = 5, 8
ROUNDS, LOCAL_STEPS, BATCH, LR = 50, 40, 128, 5e-4
ALL_FEATS = ["dl_mcs", "dl_buffer", "tx_brate_dl", "tx_pkts_dl", "tx_errors_dl", "dl_cqi",
             "ul_mcs", "ul_buffer", "rx_brate_ul", "rx_errors_ul", "ul_rssi", "ul_sinr",
             "phr", "sum_requested_prbs", "sum_granted_prbs"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def md5_bucket(s: str, mod: int) -> int:
    """Deterministic, process-independent replacement for hash()%mod."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def make_windows(df, feats, target_col, mu, sd, thr):
    """Per (run_uid, ue): sliding window [i:i+SEQ] of standardized feats; label = 1[target_col
    at the NEXT row (i+SEQ) < thr]. target_col is NOT in feats, so X never contains the label."""
    F = ((df[feats].to_numpy(np.float32) - mu) / sd)
    tgt = df[target_col].to_numpy(np.float64)
    g = df.groupby(["run_uid", "ue"], observed=True).indices
    Xs, ys = [], []
    for _k, idx in g.items():
        idx = np.sort(idx)
        if len(idx) < SEQ_LEN + 1:           # need one extra row for the next-step target
            continue
        fb, tb = F[idx], tgt[idx]
        for i in range(len(idx) - SEQ_LEN):  # window idx[i..i+SEQ-1]; target at idx[i+SEQ]
            Xs.append(fb[i:i + SEQ_LEN])
            ys.append(1.0 if tb[i + SEQ_LEN] < thr else 0.0)
    if not Xs:
        return np.zeros((0, SEQ_LEN, len(feats)), np.float32), np.zeros(0, np.float32)
    return np.stack(Xs), np.asarray(ys, np.float32)


class LSTMClf(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.lstm = nn.LSTM(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


def assign(df, mode, n, rng, alpha=1.0):
    """Row-level modes (random_split, dirichlet) scatter rows -> fragmented per-client windows.
    Intact modes (entity_ue, run_random, run_dirichlet) keep whole (run_uid,ue) groups together."""
    if mode == "entity_ue":
        ues = sorted(df["ue"].unique())
        cmap = {u: i % n for i, u in enumerate(ues)}
        return df["ue"].map(cmap).to_numpy()
    if mode in ("run_random", "run_dirichlet"):
        groups = list(df.groupby(["run_uid", "ue"], observed=True).indices.items())
        order = rng.permutation(len(groups))
        cof = {}
        if mode == "run_random":
            for j, gi in enumerate(order):
                cof[groups[gi][0]] = j % n
        else:
            edges = (np.cumsum(rng.dirichlet([alpha] * n)) * len(groups)).astype(int)
            ci = 0
            for rank, gi in enumerate(order):
                while ci < n - 1 and rank >= edges[ci]:
                    ci += 1
                cof[groups[gi][0]] = ci
        keys = list(zip(df["run_uid"], df["ue"]))
        return np.array([cof[k] for k in keys])
    if mode == "random_split":
        return rng.integers(0, n, len(df))
    if mode == "dirichlet":
        return rng.choice(n, len(df), p=rng.dirichlet([alpha] * n))
    raise ValueError(mode)


def fedavg(client_xy, Xte, yte, seed, rounds):
    torch.manual_seed(seed)
    f = Xte.shape[2]
    g = LSTMClf(f).to(DEV)
    loss_fn = nn.BCEWithLogitsLoss()
    clients = [(torch.tensor(x, device=DEV), torch.tensor(y, device=DEV))
               for x, y in client_xy if len(x) >= BATCH]
    if not clients:
        return float("nan")
    for _r in range(rounds):
        states, sizes = [], []
        for X, y in clients:
            lc = LSTMClf(f).to(DEV)
            lc.load_state_dict(g.state_dict())
            opt = torch.optim.Adam(lc.parameters(), lr=LR)
            for _s in range(LOCAL_STEPS):
                bi = torch.randint(0, len(X), (BATCH,), device=DEV)
                opt.zero_grad()
                loss_fn(lc(X[bi]), y[bi]).backward()
                opt.step()
            states.append(lc.state_dict())
            sizes.append(len(X))
        w = np.array(sizes) / sum(sizes)
        g.load_state_dict({k: sum(w[i] * states[i][k] for i in range(len(states))) for k in states[0]})
    g.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 8192):
            preds.append(torch.sigmoid(g(torch.tensor(Xte[i:i + 8192], device=DEV))).cpu().numpy())
    p = np.concatenate(preds) if preds else np.zeros(0)
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target source column, e.g. dl_cqi / dl_mcs")
    ap.add_argument("--quantile", type=float, default=0.5, help="event = 1[target_{t+1} < q-quantile]")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="1 seed, intact-vs-row subset")
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    assert args.target in ALL_FEATS, f"{args.target} not a known column"
    feats = [c for c in ALL_FEATS if c != args.target]      # DROP the target source from X
    assert args.target not in feats, "LEAK: target column still in model features"

    df = pd.read_parquet(PARQ).sort_values(["run_uid", "ue", "step"]).reset_index(drop=True)
    runs = df["run_uid"].unique()
    test_runs = {r for r in runs if md5_bucket("test|" + str(r), 10) < 3}   # ~30% whole-run OOD
    tr = df[~df["run_uid"].isin(test_runs)].reset_index(drop=True)
    te = df[df["run_uid"].isin(test_runs)].reset_index(drop=True)
    mu = tr[feats].to_numpy(np.float32).mean(0)
    sd = tr[feats].to_numpy(np.float32).std(0) + 1e-6
    thr = float(np.quantile(tr[args.target].to_numpy(np.float64), args.quantile))

    Xte, yte = make_windows(te, feats, args.target, mu, sd, thr)
    print(f"[setup] target=1[{args.target}_t+1 < {thr:.4g}] (q={args.quantile}) | feats={len(feats)} "
          f"(dropped {args.target}) | train_runs={len(runs) - len(test_runs)} test_runs={len(test_runs)}")
    print(f"[test] {len(Xte):,} windows, pos-rate {yte.mean():.3f} (must be ~{args.quantile})", flush=True)
    assert args.target not in feats and len(Xte) > 0 and 0.05 < yte.mean() < 0.95, "sanity failed"

    modes = (["entity_ue", "run_dirichlet", "dirichlet", "random_split"] if args.smoke
             else ["entity_ue", "run_random", "run_dirichlet", "random_split", "dirichlet"])
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    res = {}
    for mode in modes:
        aucs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            cl = assign(tr, mode, N_CLIENTS, rng, args.alpha)
            client_xy = [make_windows(tr[cl == c], feats, args.target, mu, sd, thr)
                         for c in range(N_CLIENTS) if (cl == c).sum() >= SEQ_LEN + 1]
            auc = fedavg(client_xy, Xte, yte, seed, args.rounds)
            aucs.append(auc)
            print(f"  [{mode:>13} s{seed}] AUC={auc:.4f}", flush=True)
        a = np.array(aucs, float)
        res[mode] = {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4),
                     "seeds": seeds, "aucs": [round(x, 4) for x in aucs]}
        print(f"== {mode}: {res[mode]['mean']} +- {res[mode]['std']}", flush=True)

    out = Path(f"twinning_auc_{args.target}{'_smoke' if args.smoke else ''}.json")
    out.write_text(json.dumps({"config": vars(args), "thr": thr, "test_pos_rate": float(yte.mean()),
                               "n_feats": len(feats), "results": res}, indent=2))
    intact = np.nanmean([res[m]["mean"] for m in res if m in ("entity_ue", "run_random", "run_dirichlet")])
    row = np.nanmean([res[m]["mean"] for m in res if m in ("random_split", "dirichlet")])
    print(f"\nWROTE {out}\nVERDICT[{args.target}]: intact {intact:.4f} vs row {row:.4f} "
          f"gap {intact - row:+.4f} -> {'CONFIRM' if intact - row > 0.03 else 'weak/REFUTE'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done] {time.time() - t0:.0f}s")
