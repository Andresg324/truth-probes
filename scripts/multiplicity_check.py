import json, os
from math import sqrt
import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODELS = ["0.5B", "1.5B", "3B", "gemma-2b"]
ALPHA = 0.05
Z95 = norm.ppf(0.975)


def run(tag):
    d = json.load(open(f"results/{tag}/RESULTS.json"))["phase2/specificity_curve"]
    crit = int(np.load(f"results/{tag}/ablation_curve.npz")["crit"])
    layers = sorted(int(k[1:]) for k in d)
    m = len(layers)                                   # cells tested in this family
    zc = norm.ppf(1 - ALPHA / (2 * m))                # Bonferroni z
    scale = zc / Z95

    rows, kept_raw, kept_corr = [], [], []
    for L in layers:
        e = d[f"L{L}"]
        v = e["value"] if isinstance(e, dict) and "value" in e else e
        asym, (lo, hi) = v["asym"], v["ci"]
        h = (hi - lo) / 2.0
        lo_c, hi_c = asym - h * scale, asym + h * scale
        sig_raw = (lo > 0) or (hi < 0)
        sig_corr = (lo_c > 0) or (hi_c < 0)
        if sig_raw:
            kept_raw.append(L)
        if sig_corr:
            kept_corr.append(L)
        if sig_raw or sig_corr:
            mark = " <- crit" if L == crit else ""
            rows.append(f"    L{L:<3d} asym {asym:+.3f}  95% [{lo:+.3f}, {hi:+.3f}] "
                        f"-> corr [{lo_c:+.3f}, {hi_c:+.3f}]  "
                        f"{'KEEP' if sig_corr else 'DROP'}{mark}")

    print(f"\n{tag}: {m} cells tested, Bonferroni z = {zc:.3f} (widening factor {scale:.3f})")
    for r in rows:
        print(r)
    print(f"    significant uncorrected: {kept_raw}")
    print(f"    significant corrected:   {kept_corr}")
    return {"n_cells": m, "widening": round(float(scale), 4),
            "sig_uncorrected": kept_raw, "sig_corrected": kept_corr,
            "crit_survives": crit in kept_corr}


if __name__ == "__main__":
    print("Approximate Bonferroni over per-layer specificity cells (normal approximation).")
    out = {t: run(t) for t in MODELS}
    print("\n" + "=" * 72)
    for t, r in out.items():
        print(f"{t:9s} {r['n_cells']:2d} cells | uncorrected {len(r['sig_uncorrected'])} "
              f"-> corrected {len(r['sig_corrected'])} | "
              f"crit layer {'survives' if r['crit_survives'] else 'DROPS'}")
    json.dump(out, open("results/multiplicity_approx.json", "w"), indent=2)
    print("\nwrote results/multiplicity_approx.json")