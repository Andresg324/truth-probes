import json, warnings
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from reporting import report, manifest

warnings.filterwarnings("ignore")

M = sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"

manifest(M, script="standardize_diag")

def deconf(tag, suf, standardize):
    acts = np.load(f"results/{tag}/acts_deception_balanced{suf}.npy")
    z = np.load(f"results/{tag}/split.npz")
    bl, yd = int(z["best_layer"]), z["y_decep"]
    trva = np.concatenate([z["tr_idx"], z["va_idx"]])
    Xtr = acts[trva, bl, :]
    sc = StandardScaler().fit(Xtr) if standardize else None
    tf = (lambda X: sc.transform(X)) if standardize else (lambda X: X)
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(tf(Xtr), yd[trva])
    rows = json.load(open(f"results/{tag}/p4_rows{suf}.json"))
    X4 = np.load(f"results/{tag}/p4_acts{suf}.npy")

    nc = [i for i, r in enumerate(rows) if r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]==r["truth"]]
    hon = [i for i, r in enumerate(rows) if r["cond"]=="honest" and r["ans"]==r["truth"]]
    yc = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
    s = np.r_[probe.decision_function(tf(X4[hon])), probe.decision_function(tf(X4[nc]))]
    return bl, len(nc), roc_auc_score(yc, s)

print(f"{'basis':16s} {'scaling':13s} best_layer n_nc de-confound")
print("-" * 60)
for name, suf in [("from_pretrained", ""), ("no_processing", "_noproc")]:
    for std in (False, True):
        bl, n, dc = deconf(M, suf, std)
        print(f"{name:16s} {'standardized' if std else 'raw':13s} {bl:9d} {n:4d} {dc:.3f}")
        report(M, "standardize_diag", f"{name}_{'standardized' if std else 'raw'}",
               {"best_layer": bl, "n_nc":n, "deconfound": float(dc)})

# Center - test

def center_last(X):
    return X - X.mean(axis=-1, keepdims=True)

def deconf_center(tag, suf, center):
    acts = np.load(f"results/{tag}/acts_deception_balanced{suf}.npy")
    z = np.load(f"results/{tag}/split.npz")
    bl, yd = int(z["best_layer"]), z["y_decep"]
    trva = np.concatenate([z["tr_idx"], z["va_idx"]])
    rows = json.load(open(f"results/{tag}/p4_rows{suf}.json"))
    X4 = np.load(f"results/{tag}/p4_acts{suf}.npy")
    
    Xfull = acts[:, bl, :]
    if center:
        Xfull = center_last(Xfull); X4 = center_last(X4)    
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(Xfull[trva], yd[trva])

    nc = [i for i, r in enumerate(rows) if r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]==r["truth"]]
    hon = [i for i, r in enumerate(rows) if r["cond"]=="honest" and r["ans"]==r["truth"]]
    yc = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
    s = np.r_[probe.decision_function(X4[hon]), probe.decision_function(X4[nc])]
    return bl, roc_auc_score(yc, s)

print(f"{'basis':16s} {'centering':10s} best_layer de-confound")
print("-" * 52)
for name, suf in [("from_pretrained", ""), ("no_processing", "_noproc")]:
    for ctr in (False, True):
        bl, dc = deconf_center(M, suf, ctr)
        print(f"{name:16s} {'centered' if ctr else 'raw':10s} {bl:9d} {dc:.3f}")
        report(M, "standardize_diag", f"center_{name}_{'centered' if ctr else 'raw'}",
               {"best_layer": bl, "deconfound": float(dc)})