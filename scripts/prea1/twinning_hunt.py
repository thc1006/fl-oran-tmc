"""Twinning-hunt (#87): is there a SEQUENCE-ESSENTIAL target on Twinning -> a within-Twinning
positive point for the Delta_traj law (to complement the cqi/mcs LEVEL negatives)?

Level targets on Twinning are persistent (run-rate) -> gap ~ 0. Change/drop EVENTS have low/negative
autocorr (dl_mcs drop>2: -0.10; dl_cqi drop>2: -0.07) -> candidates. We PRE-SCREEN with the
capacity-matched diagnostic (central LSTM seq5 vs seq1 = Delta_seq; vs shuffle-within-window =
Delta_traj) BEFORE any factorial: if the event is unpredictable (seq5 ~ 0.5) or order-free
(Delta_traj ~ 0), there is no within-Twinning positive and we report that honestly. Only if
Delta_traj is materially > 0 do we run intact (run_dirichlet) vs row (dirichlet) for the gap.

Local 4060 Ti: fp32, batched eval, no leak (the event's future value is not in the window), cage the
launch (systemd-run -p MemoryMax=24G -p MemorySwapMax=0). Reuses twinning_auc_impact (LSTMClf/assign/
fedavg) + central_auc(shuffle) mirrored from coloran_deltaseq_sweep.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import twinning_auc_impact as T   # LSTMClf, assign, fedavg, md5_bucket, ALL_FEATS, SEQ, NCLI, DEV

PARQ = "artifacts/prea1/twinning/data/twinning_subset.parquet"
DEV = T.DEV
# candidate change-event targets: (source col, drop delta). low-autocorr events from pre-analysis.
CANDIDATES = [("dl_mcs", 2.0), ("dl_cqi", 2.0), ("dl_mcs", 1.0)]


def make_event_windows(df, feats, src, delta, mu, sd):
    """window rows [i..i+SEQ-1] of standardized feats; label = 1[ src drops > delta from the last
    input step to the NEXT step ] = 1[c[i+SEQ] < c[i+SEQ-1] - delta]. The future step c[i+SEQ] is
    NOT in the window -> no leak."""
    F = ((df[feats].to_numpy(np.float32) - mu) / sd)
    c = df[src].to_numpy(np.float64)
    Xs, ys = [], []
    for _k, idx in df.groupby(["run_uid", "ue"], observed=True).indices.items():
        idx = np.sort(idx)
        if len(idx) < T.SEQ_LEN + 1:
            continue
        fb, cb = F[idx], c[idx]
        for i in range(len(idx) - T.SEQ_LEN):
            Xs.append(fb[i:i + T.SEQ_LEN])
            ys.append(1.0 if cb[i + T.SEQ_LEN] < cb[i + T.SEQ_LEN - 1] - delta else 0.0)
    if not Xs:
        return np.zeros((0, T.SEQ_LEN, len(feats)), np.float32), np.zeros(0, np.float32)
    return np.stack(Xs), np.asarray(ys, np.float32)


def central_auc(Xtr, ytr, Xte, yte, seed, n_steps=2000, shuffle=False):
    if shuffle:
        rng = np.random.default_rng(seed + 777)

        def _shuf(X):
            n, ln = X.shape[0], X.shape[1]
            p = rng.permuted(np.tile(np.arange(ln), (n, 1)), axis=1)
            return X[np.arange(n)[:, None], p]

        Xtr, Xte = _shuf(Xtr), _shuf(Xte)
    torch.manual_seed(seed)
    m = T.LSTMClf(Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=T.LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    for _ in range(n_steps):
        bi = torch.randint(0, len(Xt), (T.BATCH,), device=DEV)
        opt.zero_grad()
        loss_fn(m(Xt[bi]), Yt[bi]).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(m(torch.tensor(Xte[i:i + 8192], device=DEV))).cpu().numpy()
                            for i in range(0, len(Xte), 8192)])
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--gate-deltatraj", type=float, default=0.03, help="run factorial only if Delta_traj exceeds this")
    args = ap.parse_args()
    df = pd.read_parquet(PARQ).sort_values(["run_uid", "ue", "step"]).reset_index(drop=True)
    runs = df["run_uid"].unique()
    test = {r for r in runs if T.md5_bucket("test|" + str(r), 10) < 3}
    tr = df[~df["run_uid"].isin(test)].reset_index(drop=True)
    te = df[df["run_uid"].isin(test)].reset_index(drop=True)
    feats = list(T.ALL_FEATS)
    mu = tr[feats].to_numpy(np.float32).mean(0)
    sd = tr[feats].to_numpy(np.float32).std(0) + 1e-6
    print(f"[hunt] train_runs={len(runs)-len(test)} test_runs={len(test)} | candidates={CANDIDATES} gate Delta_traj>{args.gate_deltatraj}")
    print(f"{'target':>16} {'pos':>5} | {'inst':>6} {'seqC':>6} {'shufC':>6} | {'D_seq':>7} {'D_traj':>7} | verdict")
    out = []
    for src, delta in CANDIDATES:
        name = f"{src}_drop{delta:g}"
        Xte, yte = make_event_windows(te, feats, src, delta, mu, sd)
        Xtr, ytr = make_event_windows(tr, feats, src, delta, mu, sd)
        pos = float(yte.mean())
        if not (0.02 < pos < 0.98) or len(np.unique(yte)) < 2:
            print(f"{name:>16} {pos:>5.3f} | degenerate pos-rate -> skip"); continue
        seq_c = central_auc(Xtr, ytr, Xte, yte, seed=0)
        inst = central_auc(Xtr[:, -1:, :], ytr, Xte[:, -1:, :], yte, seed=0)
        shuf = central_auc(Xtr, ytr, Xte, yte, seed=0, shuffle=True)
        d_seq, d_traj = seq_c - inst, seq_c - shuf
        predictable = seq_c > 0.55
        seq_essential = d_traj > args.gate_deltatraj
        verdict = ("RUN factorial" if (predictable and seq_essential)
                   else ("unpredictable (seqC~chance)" if not predictable else "order-free (D_traj~0)"))
        print(f"{name:>16} {pos:>5.3f} | {inst:>6.4f} {seq_c:>6.4f} {shuf:>6.4f} | {d_seq:>+7.4f} {d_traj:>+7.4f} | {verdict}", flush=True)
        rec = {"target": name, "pos": round(pos, 3), "inst": round(inst, 4), "seq_central": round(seq_c, 4),
               "shuffle_central": round(shuf, 4), "delta_seq": round(d_seq, 4), "delta_traj": round(d_traj, 4),
               "predictable": predictable, "seq_essential": seq_essential}
        if predictable and seq_essential:
            aucs = {m: [] for m in ("run_dirichlet", "dirichlet")}
            for seed in (0, 1, 2):
                for mode in ("run_dirichlet", "dirichlet"):
                    rng = np.random.default_rng(seed)
                    cl = T.assign(tr, mode, T.N_CLIENTS, rng, 1.0)
                    cxy = [make_event_windows(tr[cl == c], feats, src, delta, mu, sd)
                           for c in range(T.N_CLIENTS) if (cl == c).sum() >= T.SEQ_LEN + 1]
                    cxy = [(x, y) for x, y in cxy if len(x) >= T.BATCH]
                    aucs[mode].append(T.fedavg(cxy, Xte, yte, seed, args.rounds))
            intact, row = float(np.nanmean(aucs["run_dirichlet"])), float(np.nanmean(aucs["dirichlet"]))
            rec.update({"intact": round(intact, 4), "row": round(row, 4), "gap": round(intact - row, 4),
                        "gap_seeds": {m: [round(x, 4) for x in aucs[m]] for m in aucs}})
            print(f"{'':>16} {'':>5} | factorial: intact={intact:.4f} row={row:.4f} GAP={intact-row:+.4f}", flush=True)
        out.append(rec)
    Path("artifacts/prea1/twinning/twinning_hunt.json").write_text(json.dumps({"targets": out}, indent=2))
    print("\nWROTE artifacts/prea1/twinning/twinning_hunt.json")
    pos_found = [r for r in out if r.get("gap", 0) > 0.03]
    print(f"VERDICT: {'within-Twinning POSITIVE found: ' + ', '.join(r['target'] for r in pos_found) if pos_found else 'NO within-Twinning sequence-essential positive (negative control stands; Twinning targets are persistent/random)'}")


if __name__ == "__main__":
    main()
