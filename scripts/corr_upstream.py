"""
Correlation between norm shrinkage and metric loss, upstream layers only.

norm_ratio.py sweeps every layer in the recovery band, but layers downstream of the readout
cannot affect it: those cells come back at exactly (c = 1, acc drop = 0, AUC drop = 0) and sit
on the origin, inflating any correlation computed over the full band. This recomputes the two
correlations over layers at or below the readout only.

Reports both readouts if both runs are present.

Usage:  python scripts/corr_upstream.py
"""

import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODELS = ["0.5B", "1.5B", "3B", "gemma-2b"]
RUNS = [("deployed", "NORM_RATIO.json"), ("local", "NORM_RATIO_LOCAL.json")]


def corr(a, v):
    return float(np.corrcoef(a, v)[0, 1]) if len(a) > 2 and a.std() > 0 and v.std() > 0 else float("nan")


print(f"{'model':9s} {'readout':9s} {'n(all)':>7s} {'n(up)':>6s} "
      f"{'corr(1-c,acc)':>14s} {'corr(1-c,auc)':>14s} {'R2@crit':>9s}")
print("-" * 76)

for mode, fname in RUNS:
    for t in MODELS:
        p = f"results/{t}/{fname}"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        per, s = d["per_layer"], d["summary"]
        read_L = s.get("readout_layer")
        ks = [k for k in per if int(k[1:]) <= read_L]        # drop cells downstream of readout
        c = np.array([1 - per[k]["c_mean"] for k in ks])
        a = np.array([per[k]["acc_drop"] for k in ks])
        u = np.array([per[k]["auc_drop"] for k in ks])
        print(f"{t:9s} {mode:9s} {len(per):>7d} {len(ks):>6d} "
              f"{corr(c, a):>+14.3f} {corr(c, u):>+14.3f} "
              f"{str(s.get('rescale_r2_at_crit')):>9s}")
    print()

print("Read: the registered corroborating check asks whether corr(1-c, acc drop) is positive")
print("and materially larger than corr(1-c, AUC drop). R2@crit is the registered statistic;")
print("negative values mean the radial-rescaling model predicts worse than the mean.")