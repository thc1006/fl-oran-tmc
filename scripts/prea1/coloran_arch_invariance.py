"""Architecture-invariance of the fragmentation gap (PREREG-A2 H4) -- LOCAL 4060 Ti design.

Claim: (a) a NO-sequence model (mean-pool MLP) shows gap ~ 0 for EVERY target incl high-Delta_seq
BLER (sanity: the gap is sequence-specific, not a generic partition artifact); (b) sequence
models (LSTM, GRU, Transformer) all show the gap rising with the task's Delta_seq -> the law is
architecture-invariant.

Hardware-aware for the local RTX 4060 Ti (16 GiB VRAM, 30 GB RAM, 20 cores), per the
local-box-crash lesson:
  * frac 0.15 (~2.7M rows; windows ~0.9 GB on GPU, << 16 GiB), all fp32 (no Mamba fp16 NaN).
  * tiny models (~40-60K params); batch 128; eval batched at 8192.
  * SEQUENTIAL (no joblib); windows built ONCE per (target, mode, seed) and shared across the
    4 archs (4x less windowing). Cage the launch: systemd-run -p MemoryMax=24G -p MemorySwapMax=0.
Mamba (selective scan) deferred to V100 (fp16-sensitive); GRU is the stable extra recurrent arch.

Reuses the validated sweep driver (imported): build_target / std_windows / partition / windowing.
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import coloran_deltaseq_sweep as S
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.partition import partition_clients
from fl_oran.data_v2.split import ood_split_by_tr

DEV = S.DEV
INTACT, ROW = ("iid", "run_dirichlet"), ("dirichlet", "random_split")


class MeanPoolMLP(nn.Module):          # NO sequence: order+position invariant -> Delta_seq~0; gap sanity
    def __init__(self, f, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x.mean(1)).squeeze(-1)


class LSTMClf(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.rnn = nn.LSTM(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class GRUClf(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.rnn = nn.GRU(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class TinyTransformer(nn.Module):
    def __init__(self, f, d=64, nhead=4):
        super().__init__()
        self.proj = nn.Linear(f, d)
        self.enc = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=128, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.enc(self.proj(x)).mean(1)).squeeze(-1)


ARCHS = {"meanpool_mlp": MeanPoolMLP, "lstm": LSTMClf, "gru": GRUClf, "transformer": TinyTransformer}


def fedavg(make_model, clients, Xte, yte, seed, rounds):
    """clients: list of (X_gpu_tensor, y_gpu_tensor). Generalized over the model class."""
    torch.manual_seed(seed)
    f = Xte.shape[2]
    g = make_model(f).to(DEV)
    loss_fn = nn.BCEWithLogitsLoss()
    if not clients:
        return float("nan")
    for _r in range(rounds):
        states, sizes = [], []
        for X, y in clients:
            lc = make_model(f).to(DEV)
            lc.load_state_dict(g.state_dict())
            opt = torch.optim.Adam(lc.parameters(), lr=S.LR)
            for _s in range(50):
                bi = torch.randint(0, len(X), (S.BATCH,), device=DEV)
                opt.zero_grad()
                loss_fn(lc(X[bi]), y[bi]).backward()
                opt.step()
            states.append(lc.state_dict())
            sizes.append(len(X))
        w = np.array(sizes) / sum(sizes)
        g.load_state_dict({k: sum(w[i] * states[i][k] for i in range(len(states))) for k in states[0]})
    g.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(g(Xte[i:i + 8192])).cpu().numpy() for i in range(0, len(Xte), 8192)])
    return float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-run-frac", type=float, default=0.15)   # local-safe
    ap.add_argument("--parquet", default=S.PARQ)
    ap.add_argument("--targets", default="bler_th10,bler_trend5,brate_med,mcs_med")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default="artifacts/prea1/twinning/coloran_arch_invariance.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = set(args.targets.split(","))
    targets = [t for t in S.TARGETS if t["name"] in names]

    # task Delta_seq/Delta_traj (LSTM, from the sweep + addendum) to plot against
    task = {}
    for f in glob.glob("artifacts/prea1/twinning/coloran_deltaseq_g*.json"):
        for r in json.load(open(f))["targets"]:
            task[r["name"]] = {"delta_seq": r["delta_seq"]}
    try:
        for r in json.load(open("artifacts/prea1/twinning/coloran_deltatraj.json"))["targets"]:
            task.setdefault(r["name"], {})["delta_traj"] = r["delta_traj"]
    except FileNotFoundError:
        pass

    keep_cols = sorted(set(S.V3_CONTINUOUS) | {"run_id", "slice_id", "step_idx", "tr", "bs_id"})
    df = pd.read_parquet(args.parquet, columns=keep_cols)
    runs = df["run_id"].unique()
    keep = {r for r in runs if S.md5f(r) < int(args.keep_run_frac * 1000)}
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    print(f"[load] {len(df):,} rows, {len(keep)}/{len(runs)} runs (frac={args.keep_run_frac}) | "
          f"archs={list(ARCHS)} targets={[t['name'] for t in targets]} seeds={seeds} rounds={args.rounds}")

    # aucs[arch][target][mode] = list over seeds
    aucs = {a: {t["name"]: {m: [] for m in INTACT + ROW} for t in targets} for a in ARCHS}
    for spec in targets:
        feats = [c for c in S.V3_CONTINUOUS if c not in spec["drop"]]
        df2, _ = S.build_target(df, spec, S.TRAIN_TR)
        sp = ood_split_by_tr(df2, S.TRAIN_TR, [22, 23, 24], S.TEST_TR)
        mu = sp.train[feats].to_numpy(np.float32).mean(0)
        sd = sp.train[feats].to_numpy(np.float32).std(0) + 1e-6
        Xte_np, Yte = build_run_sequences(sp.test, feats, ["y_sla_next"], seq_len=S.SEQ)
        Xte = torch.tensor(S.std_windows(Xte_np, mu, sd), device=DEV)
        yte = Yte[:, 0]
        for mode in INTACT + ROW:
            for seed in seeds:
                kw = dict(seed=seed)
                if mode in ("dirichlet", "run_dirichlet"):
                    kw.update(alpha=1.0, n_clients=8)
                elif mode == "random_split":
                    kw.update(n_clients=8)
                cdfs = partition_clients(sp.train, mode=mode, **kw)
                # build windows ONCE per (target, mode, seed); share across the 4 archs
                clients = []
                for d in cdfs.values():
                    Xc, Yc = build_run_sequences(d, feats, ["y_sla_next"], seq_len=S.SEQ)
                    if len(Xc) >= S.BATCH:
                        clients.append((torch.tensor(S.std_windows(Xc, mu, sd), device=DEV),
                                        torch.tensor(Yc[:, 0], device=DEV)))
                for arch, mk in ARCHS.items():
                    aucs[arch][spec["name"]][mode].append(fedavg(mk, clients, Xte, yte, seed, args.rounds))
                del clients
                torch.cuda.empty_cache()
        print(f"  done target {spec['name']}", flush=True)

    out = []
    print(f"\n{'arch':>13} {'target':>11} {'Delta_seq':>9} {'intact':>7} {'row':>7} {'gap':>8}")
    for arch in ARCHS:
        for spec in targets:
            t = spec["name"]
            intact = float(np.nanmean([np.nanmean(aucs[arch][t][m]) for m in INTACT]))
            row = float(np.nanmean([np.nanmean(aucs[arch][t][m]) for m in ROW]))
            out.append({"arch": arch, "target": t, "delta_seq": task.get(t, {}).get("delta_seq"),
                        "delta_traj": task.get(t, {}).get("delta_traj"),
                        "intact": round(intact, 4), "row": round(row, 4), "gap": round(intact - row, 4),
                        "per_mode_seeds": {m: [round(x, 4) for x in aucs[arch][t][m]] for m in INTACT + ROW}})
            print(f"{arch:>13} {t:>11} {str(task.get(t,{}).get('delta_seq')):>9} "
                  f"{intact:>7.4f} {row:>7.4f} {intact - row:>+8.4f}")
    # MLP sanity verdict
    mlp_gaps = [r["gap"] for r in out if r["arch"] == "meanpool_mlp"]
    print(f"\nMLP-sanity: max |gap| over targets = {max(abs(g) for g in mlp_gaps):.4f} "
          f"(H4 sanity: a no-sequence model must show gap ~ 0 even for high-Delta_seq BLER)")
    Path(args.out).write_text(json.dumps({"config": vars(args), "rows": out}, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
