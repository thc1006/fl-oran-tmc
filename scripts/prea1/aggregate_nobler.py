"""Aggregate + adversarial-review the no-BLER ablation vs the with-BLER factorial.
Checks: every no-BLER cell actually dropped dl_bler/ul_bler; no NaN/missing cells;
paired gap (run_dirichlet - dirichlet) by seed with bootstrap CI; with-BLER reference
pulled from the ACTUAL prea1_factorial summaries (not memory); seed-set difference flagged."""
import glob, json, random, statistics
from collections import defaultdict

NOBLER = "artifacts/prea1_nobler_ablation"
WITHBLER = "artifacts/prea1_factorial"


def boot_ci(deltas, n=10000, seed=0):
    rng = random.Random(seed)
    out = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def load(root, drop_required):
    cells = defaultdict(dict)
    bad = []
    files = glob.glob(f"{root}/v7_*/summary.json")
    for f in files:
        c = json.load(open(f)); g = c["config"]
        dc = sorted(g.get("drop_continuous", []))
        if dc != sorted(drop_required):
            bad.append((f.split("/")[-2], f"drop_continuous={dc}"))
        auc = c.get("test_auc")
        if auc is None or auc != auc:
            bad.append((f.split("/")[-2], f"bad_auc={auc}")); continue
        cells[(g["partition_mode"], g.get("alpha"))][g["seed"]] = auc
    return cells, bad, len(files)


nob, bad_nob, n_nob = load(NOBLER, ["dl_bler", "ul_bler"])
wb, bad_wb, _ = load(WITHBLER, [])

print("=== ADVERSARIAL CHECKS (no-BLER) ===")
print(f"  files: {n_nob} | cells with valid auc: {sum(len(v) for v in nob.values())}")
print(f"  wrong drop_continuous or NaN: {bad_nob if bad_nob else 'NONE — all cells dropped dl_bler,ul_bler with valid AUC'}")

print("\n=== no-BLER per (mode, alpha) ===")
for k in sorted(nob, key=lambda x: (x[0], x[1] or 0)):
    v = nob[k]
    print(f"  {k[0]:>13} a={k[1]}: mean={statistics.mean(v.values()):.4f} "
          f"std={statistics.pstdev(v.values()):.4f} seeds={sorted(v)}")

print("\n=== paired gap run_dirichlet - dirichlet (PAIRED by seed) ===")
print(f"{'':>6} | {'no-BLER':>34} || {'with-BLER (prea1_factorial)':>34}")
for a in [0.1, 1.0]:
    rd, dd = nob.get(("run_dirichlet", a), {}), nob.get(("dirichlet", a), {})
    cm = sorted(set(rd) & set(dd)); d = [rd[s] - dd[s] for s in cm]
    lo, hi = boot_ci(d) if d else (float("nan"), float("nan"))
    nob_str = f"gap={statistics.mean(d):+.4f} CI[{lo:+.4f},{hi:+.4f}] n={len(cm)}" if d else "n/a"
    rw, dw = wb.get(("run_dirichlet", a), {}), wb.get(("dirichlet", a), {})
    cw = sorted(set(rw) & set(dw)); dwd = [rw[s] - dw[s] for s in cw]
    wb_str = f"gap={statistics.mean(dwd):+.4f} n={len(cw)} (seeds {cw})" if dwd else "n/a"
    print(f"  a={a} | {nob_str:>34} || {wb_str:>34}")

print("\n=== iid (natural) anchor ===")
for lbl, src in (("no-BLER", nob), ("with-BLER", wb)):
    v = src.get(("iid", 0.5)) or src.get(("iid", None)) or {}
    if v:
        print(f"  {lbl}: iid mean={statistics.mean(v.values()):.4f} seeds={sorted(v)}")
print("\nSEED-SET CAVEAT: no-BLER uses seeds 0-4 (n=5); with-BLER prea1_factorial used seeds 0,1,2 (n=3). "
      "Gaps are compared, not paired across the two runs.")
