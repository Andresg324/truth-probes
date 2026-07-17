"""
leak_census.py — instruction-leak census across the ladder (laptop, no GPU).
For each model: rebuild the frozen teacher-forced probe, score the instructed transfer
set, and report positive control + transfer AUCs (deception/truth/polarity) + the
NON-COMPLIANCE de-confound (honest vs told-to-lie-but-truthful). High de-confound = the
probe reads the instruction, not the answer => that model's transfer number is a leak.
Run from the repo root after the census run:  python leak_census.py
"""

import json, os, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

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

def safe(y, x):
    return roc_auc_score(y, x) if len(set(y.tolist())) == 2 else float("nan")

hdr = f"{'model':9s} {'pos_ctrl':>8s} {'decep':>7s} {'truth':>7s} {'polar':>7s} {'deconf':>7s} {'n':>5s} verdict"
print(hdr); print("-" * len(hdr))
for tag in TAGS:
    R = root(tag)
    if not os.path.exists(f"{R}/p4_rows.json"):
        print(f"{tag:9s} (no transfer data)"); continue
    acts = np.load(f"{R}/acts_deception_balanced.npy")
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(acts, yd, grp))
    bl = int(np.argmax([LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], yd[tr]).score(acts[te,L,:], yd[te]) for L in range(acts.shape[1])]))
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, bl, :], yd[tr])
    pc = roc_auc_score(yd[te], probe.decision_function(acts[te, bl, :]))

    # intsructed transfer set, compliant 
    rows = json.load(open(f"{R}/p4_rows.json")); X4 = np.load(f"{R}/p4_acts.npy")
    keep = [i for i, r in enumerate(rows)
            if (r["cond"] == "honest" and r["ans"] == r["truth"])
            or (r["cond"] == "lie" and r["ans"] in ("yes", "no") and r["ans"] != r["truth"])]
    s = probe.decision_function(X4[keep])
    decep = np.array([1 if rows[i]["cond"] == "lie" else 0 for i in keep])
    truth = np.array([rows[i]["stmt_true"] for i in keep])
    polar = np.array([1 if rows[i]["ans"] == "yes" else 0 for i in keep])
    a_d, a_t, a_p = safe(decep, s), safe(truth, s), safe(polar, s)

    # De - confound
    nc = [i for i, r in enumerate(rows) if r["cond"] == "lie" and r["ans"] in ("yes", "no") and r["ans"] == r["truth"]]
    hon = [i for i, r in enumerate(rows) if r["cond"] == "honest" and r["ans"] == r["truth"]]
    if len(nc) >= 5 and len(hon) >= 5:
        yc = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
        dc = roc_auc_score(yc, np.r_[probe.decision_function(X4[hon]), probe.decision_function(X4[nc])])
    else:
        dc = float("nan")
    
    leak = (not np.isnan(dc)) and abs(dc - 0.5) > 0.15
    verdict = "LEAK (drop)" if leak else ("clean" if not np.isnan(dc) else "n/a (few nc)")
    
    print(f"{tag:9s} {pc:8.3f} {a_d:7.3f} {a_t:7.3f} {a_p:7.3f} {dc:7.3f} {len(keep):5d} {verdict}")