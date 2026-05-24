"""Aggregate the Delta_seq law: ColO-RAN sweep (g0-g3) + synthetic control -> figure + stats.

The central result of the standalone methods paper: the fragmentation AUC gap (intact - row)
is a monotone function of Delta_seq (the capacity-matched sequence value). Plots ColO-RAN's
real-target points + the synthetic controlled backbone, with the y=x line (the FALSIFIED
'gap <= Delta_seq' bound) for reference.
"""
import glob, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "artifacts/prea1/twinning"


def load_coloran():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/coloran_deltaseq_g*.json")):
        rows += json.load(open(f))["targets"]
    return sorted(rows, key=lambda r: r["delta_seq"])


def load_synth():
    p = f"{OUT}/synth_deltaseq_control.json"
    try:
        return json.load(open(p))["rows"]
    except FileNotFoundError:
        return []


def rank(a):
    return np.argsort(np.argsort(a))


def boot_slope_ci(x, y, n=10000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(x)))
    slopes = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        xs, ys = x[s], y[s]
        if np.ptp(xs) < 1e-9:
            continue
        slopes.append(np.polyfit(xs, ys, 1)[0])
    slopes.sort()
    return slopes[int(0.025 * len(slopes))], slopes[int(0.975 * len(slopes))]


def main():
    cr = load_coloran()
    ds = np.array([r["delta_seq"] for r in cr])
    gp = np.array([r["gap"] for r in cr])
    pear = float(np.corrcoef(ds, gp)[0, 1])
    spear = float(np.corrcoef(rank(ds), rank(gp))[0, 1])
    slope, intercept = np.polyfit(ds, gp, 1)
    lo, hi = boot_slope_ci(ds, gp)

    print(f"=== ColO-RAN Delta_seq law ({len(cr)} targets, 5 seeds, frac 0.3, 50 rounds) ===")
    print(f"{'target':>11} | {'Delta_seq':>9} {'gap':>8}")
    for r in cr:
        print(f"{r['name']:>11} | {r['delta_seq']:>+9} {r['gap']:>+8}")
    print(f"\nPearson(gap,Delta_seq)={pear:+.3f}  Spearman={spear:+.3f}  (H1: Spearman>0.8)")
    print(f"OLS gap = {slope:.3f}*Delta_seq + {intercept:+.3f}  slope CI95 [{lo:.3f}, {hi:.3f}]  "
          f"(H1: slope CI excludes 0 -> {'PASS' if lo > 0 else 'FAIL'})")
    print(f"H1 verdict: {'CONFIRMED' if spear > 0.8 and lo > 0 else 'NOT confirmed'}")

    # figure
    sy = load_synth()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lim = max(ds.max(), gp.max(), 0.3) * 1.08
    ax.plot([0, lim], [0, lim], ls=":", c="0.6", lw=1, label="gap = $\\Delta_{seq}$ (falsified bound)")
    ax.axhline(0, c="0.8", lw=0.8)
    if sy:
        sds = np.array([r["delta_seq"] for r in sy])
        sgp = np.array([r["gap"] for r in sy])
        ax.plot(sds, sgp, "-^", c="tab:gray", ms=6, lw=1.2, alpha=0.8,
                label=f"synthetic control (Pearson {np.corrcoef(sds, sgp)[0,1]:+.2f})")
    is_bler = np.array(["bler" in r["name"] for r in cr])
    ax.scatter(ds[is_bler], gp[is_bler], c="tab:red", s=55, zorder=5, label="ColO-RAN BLER targets")
    ax.scatter(ds[~is_bler], gp[~is_bler], c="tab:blue", s=55, zorder=5, label="ColO-RAN other targets")
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, slope * xs + intercept, c="tab:red", lw=1.4, alpha=0.7,
            label=f"OLS fit (slope {slope:.2f}, $\\rho_s$={spear:.2f})")
    for r in cr:
        ax.annotate(r["name"], (r["delta_seq"], r["gap"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("$\\Delta_{seq}$  =  AUC(seq LSTM) $-$ AUC(single-step LSTM)")
    ax.set_ylabel("fragmentation gap  =  AUC(intact) $-$ AUC(row-level)")
    ax.set_title("Fragmentation AUC gap tracks sequence-essentiality ($\\Delta_{seq}$)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-0.02, lim)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/deltaseq_law.{ext}", dpi=150)
    print(f"\nWROTE {OUT}/deltaseq_law.pdf/.png")
    json.dump({"n_targets": len(cr), "pearson": round(pear, 3), "spearman": round(spear, 3),
               "ols_slope": round(float(slope), 3), "ols_intercept": round(float(intercept), 3),
               "slope_ci95": [round(lo, 3), round(hi, 3)],
               "h1_confirmed": bool(spear > 0.8 and lo > 0),
               "targets": cr}, open(f"{OUT}/deltaseq_law_summary.json", "w"), indent=2)
    print(f"WROTE {OUT}/deltaseq_law_summary.json")


if __name__ == "__main__":
    main()
