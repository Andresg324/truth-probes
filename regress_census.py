import json, os, warnings
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# Nested layer section
def nested_layer(acts3d, y, groups, seed=0):
    """3-way grouped split, choose the best layer on VAL, never on the reported TEST fold"""
    trv, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=seed).split(acts3d, y, groups))
    tr_r, va_r = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(acts3d[trv], y[trv], groups[trv]))
    tr, va = trv[tr_r], trv[va_r]
    accs = [LogisticRegression(max_iter=2000, C=0.1).fit(acts3d[tr, L, :], y[tr]).score(acts3d[va, L, :], y[va]) for L in range(acts3d.shape[1])]
    return int(np.argmax(accs)), tr, va, te

# ---- model (identical to deception_probe.py) ----
TAGS = ["0.5B", "1.5B", "3B", "7B", "14B", "gemma-2b", "gemma-9b", "llama-3b", "llama-8b"]

def root(tag):
    a = f"results/{tag}"
    b = os.path.expanduser(f"~/Desktop/truth-probes/results_gpu/{tag}")
    return a if os.path.exists(f"{a}/p4_rows.json") else b

items = json.load(open("data/mixed.json"))
yd = np.array([j for _ in items for j in (0,1)]); grp = np.arange(len(yd)) // 2

hdr = f"{'model':9s} {'instr(leak)':>13s} {'decep(XOR)':>13s} {'n':>5s} lie-dist"
print(hdr); print("-" * len(hdr))
star = lambda p: "***" if p< 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "

for tag in TAGS:
    R = root(tag)
    if not os.path.exists(f"{R}/p4_rows.json"):
        print(f"{tag:9s} (no transfer data)"); continue
    acts = np.load(f"{R}/acts_deception_balanced.npy")
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(acts, yd, grp))
    bl = int(np.argmax([LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], yd[tr]).score(acts[te,L,:], yd[te]) for L in range(acts.shape[1])]))
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, bl, :], yd[tr])

    # intsructed transfer set, compliant 
    rows = json.load(open(f"{R}/p4_rows.json")); X4 = np.load(f"{R}/p4_acts.npy")
    mask = [i for i, r in enumerate(rows) if r["ans"] in ("yes", "no")]
    s = probe.decision_function(X4[mask])
    s = (s - s.mean()) / (s.std() + 1e-9)
    instr = np.array([1.0 if rows[i]["cond"] == "lie" else 0.0 for i in mask])
    truth = np.array([float(rows[i]["stmt_true"]) for i in mask])
    polar = np.array([1.0 if rows[i]["ans"] == "yes" else 0.0 for i in mask])
    inter = truth * polar
    lie = [rows[i]["ans"] for i in mask if rows[i]["cond"] == "lie"]
    ny, nn = sum(a == "yes" for a in lie), sum(a == "no" for a in lie)

    Xc = sm.add_constant(np.column_stack([instr, truth, polar, inter]))
    cond = np.linalg.cond(Xc)
    if min(ny, nn) < 20 or min(ny, nn) / max(ny + nn, 1) < 0.20 or cond > 30:
        print(f"{tag:9s} {'-':12s} {'-':>12s} {len(mask):>5d} {ny}y/{nn}n DEGENERATE (cond={cond:.0f})")
        continue
    m = sm.OLS(s, Xc).fit()
    ci, pi = m.params[1], m.pvalues[1]
    cx, px = m.params[4], m.pvalues[4]
    if abs(ci) > 5 or abs(cx) > 5:
        print(f"{tag:9s} {'-':>12s} {'-':>12s} {len(mask):>5d} {ny}y/{nn}n QUASI-SEPARATION"); continue
    print(f"{tag:9s} {ci:>9.2f}{star(pi):<4s} {cx:>9.2f}{star(px):<4s} {len(mask):>5d} {ny}y/{nn}n cond={cond:.0f}")