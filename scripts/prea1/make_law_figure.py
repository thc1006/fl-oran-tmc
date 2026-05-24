"""Two-panel money figure for the standalone paper: fragmentation gap vs Delta_seq (left) and
vs Delta_traj (right), with the synthetic backbone + OLS fit. Reads the committed law JSONs;
saves to artifacts/figures/ (the tracked paper-figure dir) so paper/seq_integrity.tex can include it.
"""
import glob, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = "artifacts/prea1/twinning"


def load():
    cr = {}
    for f in glob.glob(f"{T}/coloran_deltaseq_g*.json"):
        for r in json.load(open(f))["targets"]:
            cr[r["name"]] = {"delta_seq": r["delta_seq"], "gap": r["gap"]}
    for r in json.load(open(f"{T}/coloran_deltatraj.json"))["targets"]:
        cr.setdefault(r["name"], {})["delta_traj"] = r["delta_traj"]
    rows = [dict(name=k, **v) for k, v in cr.items() if "delta_seq" in v and "delta_traj" in v]
    try:
        sy = json.load(open(f"{T}/synth_deltaseq_control.json"))["rows"]
    except FileNotFoundError:
        sy = []
    return rows, sy


def boot_slope_ci(x, y, n=10000, seed=0):
    rng = random.Random(seed); idx = list(range(len(x))); s = []
    for _ in range(n):
        b = [rng.choice(idx) for _ in idx]
        if np.ptp(x[b]) > 1e-9:
            s.append(np.polyfit(x[b], y[b], 1)[0])
    s.sort(); return s[int(0.025 * len(s))], s[int(0.975 * len(s))]


def rank(a):
    return np.argsort(np.argsort(a))


def panel(ax, x, gp, names, sy, xkey, xlabel):
    is_bler = np.array(["bler" in n for n in names])
    lim = max(x.max(), gp.max(), 0.3) * 1.08
    ax.plot([0, lim], [0, lim], ls=":", c="0.6", lw=1, label="gap = x (falsified bound)")
    ax.axhline(0, c="0.85", lw=0.8)
    if sy and xkey == "delta_seq":
        sx = np.array([r["delta_seq"] for r in sy]); sg = np.array([r["gap"] for r in sy])
        ax.plot(sx, sg, "-^", c="tab:gray", ms=5, lw=1.1, alpha=0.8, label="synthetic control")
    ax.scatter(x[is_bler], gp[is_bler], c="tab:red", s=45, zorder=5, label="ColO-RAN BLER")
    ax.scatter(x[~is_bler], gp[~is_bler], c="tab:blue", s=45, zorder=5, label="ColO-RAN other")
    sl, ic = np.polyfit(x, gp, 1); lo, hi = boot_slope_ci(x, gp)
    sp = np.corrcoef(rank(x), rank(gp))[0, 1]
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, sl * xs + ic, c="tab:red", lw=1.3, alpha=0.7,
            label=f"OLS slope {sl:.2f}, $\\rho_s$={sp:.2f}")
    # label only the two story points (the deviator + the low-BLER transition); dense clusters
    # are color-coded (red BLER high, blue other low) and left unlabeled to avoid an unreadable blob
    label = {"brate_med", "bler_trend5"}
    for n, xv, gv in zip(names, x, gp):
        if n in label:
            ax.annotate(n, (xv, gv), fontsize=7, xytext=(5, -2), textcoords="offset points")
    ax.set_xlabel(xlabel); ax.set_xlim(-0.02, lim); ax.grid(alpha=0.2)
    return sl, (lo, hi), sp


def main():
    rows, sy = load()
    rows.sort(key=lambda r: r["delta_seq"])
    names = [r["name"] for r in rows]
    ds = np.array([r["delta_seq"] for r in rows]); dt = np.array([r["delta_traj"] for r in rows])
    gp = np.array([r["gap"] for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    s1 = panel(a1, ds, gp, names, sy, "delta_seq", "$\\Delta_{seq}$ (seq $-$ single-step LSTM)")
    a1.set_ylabel("fragmentation gap  (intact $-$ row-level AUC)")
    a1.legend(fontsize=6.5, loc="upper left")
    s2 = panel(a2, dt, gp, names, sy, "delta_traj", "$\\Delta_{traj}$ (seq $-$ shuffled-window LSTM)")
    a2.legend(fontsize=6.5, loc="upper left")
    a1.set_title(f"(a) gap vs $\\Delta_{{seq}}$  ($\\rho_s$={s1[2]:.2f})")
    a2.set_title(f"(b) gap vs $\\Delta_{{traj}}$  ($\\rho_s$={s2[2]:.2f})")
    fig.tight_layout()
    import os
    os.makedirs("artifacts/figures", exist_ok=True)
    for p in ("artifacts/figures/seq_integrity_law", f"{T}/deltaseq_law_2panel"):
        for ext in ("pdf", "png"):
            fig.savefig(f"{p}.{ext}", dpi=150)
    print(f"gap vs Delta_seq: slope {s1[0]:.3f} CI{s1[1]} rho_s {s1[2]:.3f}")
    print(f"gap vs Delta_traj: slope {s2[0]:.3f} CI{s2[1]} rho_s {s2[2]:.3f}")
    print("WROTE artifacts/figures/seq_integrity_law.pdf/.png")


if __name__ == "__main__":
    main()
