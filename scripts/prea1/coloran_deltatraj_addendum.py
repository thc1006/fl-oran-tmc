"""Delta_traj addendum (post-hoc refinement): order-free shuffle-within-window baseline that
isolates the TRAJECTORY value (partition-vulnerable) from the run-rate (partition-invariant).

Motivated by the brate_med deviation in the pre-registered Delta_seq law (brate Delta_seq=0.111
but gap~0): autocorr shows brate is persistent (0.98) so its multi-step value is order-free
denoising, NOT trajectory. Delta_traj = AUC(seq LSTM) - AUC(shuffled-window LSTM) should pull
brate to ~0 and predict the gap at least as well as Delta_seq.

Reuses the VALIDATED sweep driver (imported, no reimplementation): build_target, central_auc
(shuffle param), std_windows, windowing, target family. Diagnostic-only (no FL) -> fast; merges
the existing coloran_deltaseq_g*.json gaps. Cross-checks seq_central against the sweep run.

V100: python coloran_deltatraj_addendum.py --keep-run-frac 0.3
smoke: python coloran_deltatraj_addendum.py --keep-run-frac 0.08 --targets brate_med,bler_th10,mcs_med
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import coloran_deltaseq_sweep as S
from fl_oran.data_v2.sequences import build_run_sequences
from fl_oran.data_v2.split import ood_split_by_tr


def rank(a):
    return np.argsort(np.argsort(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-run-frac", type=float, default=0.3)
    ap.add_argument("--parquet", default=S.PARQ)
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="artifacts/prea1/twinning/coloran_deltatraj.json")
    args = ap.parse_args()
    want = set(args.targets.split(",")) if args.targets else None
    targets = [t for t in S.TARGETS if (want is None or t["name"] in want)]

    sweep = {}
    for f in glob.glob("artifacts/prea1/twinning/coloran_deltaseq_g*.json"):
        for r in json.load(open(f))["targets"]:
            sweep[r["name"]] = r

    keep_cols = sorted(set(S.V3_CONTINUOUS) | {"run_id", "slice_id", "step_idx", "tr", "bs_id"})
    df = pd.read_parquet(args.parquet, columns=keep_cols)
    runs = df["run_id"].unique()
    keep = {r for r in runs if S.md5f(r) < int(args.keep_run_frac * 1000)}
    df = df[df["run_id"].isin(keep)].reset_index(drop=True)
    print(f"[load] {len(df):,} rows, {len(keep)}/{len(runs)} runs (frac={args.keep_run_frac})")
    print(f"{'target':>11} | {'inst':>6} {'seqC':>6} {'shufC':>6} | {'D_seq':>7} {'D_traj':>7} {'gap':>8}  seqC_match")

    out = []
    for spec in targets:
        feats = [c for c in S.V3_CONTINUOUS if c not in spec["drop"]]
        df2, _ = S.build_target(df, spec, S.TRAIN_TR)
        sp = ood_split_by_tr(df2, S.TRAIN_TR, [22, 23, 24], S.TEST_TR)
        mu = sp.train[feats].to_numpy(np.float32).mean(0)
        sd = sp.train[feats].to_numpy(np.float32).std(0) + 1e-6
        Xte, Yte = build_run_sequences(sp.test, feats, ["y_sla_next"], seq_len=S.SEQ)
        Xte, yte = S.std_windows(Xte, mu, sd), Yte[:, 0]
        Xtr5, Ytr5 = build_run_sequences(sp.train, feats, ["y_sla_next"], seq_len=S.SEQ)
        Xtr5, ytr5 = S.std_windows(Xtr5, mu, sd), Ytr5[:, 0]
        seq_c = S.central_auc(Xtr5, ytr5, Xte, yte, seed=0)
        inst = S.central_auc(Xtr5[:, -1:, :], ytr5, Xte[:, -1:, :], yte, seed=0)
        shuf_c = S.central_auc(Xtr5, ytr5, Xte, yte, seed=0, shuffle=True)
        d_seq, d_traj = seq_c - inst, seq_c - shuf_c
        ref = sweep.get(spec["name"], {})
        gap, seqc_ref = ref.get("gap"), ref.get("seq_central")
        match = "n/a" if seqc_ref is None else f"|d|={abs(seq_c - seqc_ref):.3f}"
        out.append({"name": spec["name"], "inst": round(inst, 4), "seq_central": round(seq_c, 4),
                    "shuffle_central": round(shuf_c, 4), "delta_seq": round(d_seq, 4),
                    "delta_traj": round(d_traj, 4), "gap": gap, "seq_central_sweep_ref": seqc_ref})
        print(f"{spec['name']:>11} | {inst:>6.4f} {seq_c:>6.4f} {shuf_c:>6.4f} | "
              f"{d_seq:>+7.4f} {d_traj:>+7.4f} {str(gap):>8}  {match}", flush=True)

    g = [(r["delta_traj"], r["delta_seq"], r["gap"]) for r in out if r["gap"] is not None]
    if len(g) > 1:
        dt = np.array([x[0] for x in g]); dse = np.array([x[1] for x in g]); gp = np.array([x[2] for x in g])
        print(f"\nvs gap:  Delta_traj  Pearson={np.corrcoef(dt, gp)[0,1]:+.3f} Spearman={np.corrcoef(rank(dt), rank(gp))[0,1]:+.3f}"
              f"   |   Delta_seq Pearson={np.corrcoef(dse, gp)[0,1]:+.3f} Spearman={np.corrcoef(rank(dse), rank(gp))[0,1]:+.3f}")
        print("(Delta_traj should match or beat Delta_seq as a gap predictor; brate_med should move onto the line)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"targets": out}, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
