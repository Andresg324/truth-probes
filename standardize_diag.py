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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

items = json.load(open("data/mixed.json"))
yd = np.array([j for _ in items for j in (0, 1)]); grp = np.arange(len(yd)) // 2

M = sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"

def deconf(root, standardize):
    acts = np.load(f"{root}/acts_deception_balanced.npy")
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(acts, yd, grp))
    bl = int(np.argmax([LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], yd[tr]).score(acts[te, L, :], yd[te]) for L in range(acts.shape[1])]))
    Xtr = acts[tr, bl, :]
    sc = StandardScaler().fit(Xtr) if standardize else None
    tf = (lambda X: sc.transform(X)) if standardize else (lambda X:X)
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(tf(Xtr), yd[tr])

    rows = json.load(open(f"{root}/p4_rows.json")); X4 = np.load(f"{root}/p4_acts.npy")
    nc = [i for i, r in enumerate(rows) if r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]==r["truth"]]
    hon = [i for i, r in enumerate(rows) if r["cond"]=="honest" and r["ans"]==r["truth"]]
    yc = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
    s = np.r_[probe.decision_function(tf(X4[hon])), probe.decision_function(tf(X4[nc]))]
    return bl, len(nc), roc_auc_score(yc, s)

print(f"{'basis':16s} {'scaling':13s} best_layer n_nc de-confound")
print("-" * 60)
for name, root in [("from_pretrained", f"results/{M}"), ("no_processing", f"results_census/{M}")]:
    for std in (False, True):
        bl, n, dc = deconf(root, std)
        print(f"{name:16s} {'standardized' if std else 'raw':13s} {bl:9d} {n:4d} {dc:.3f}")

# Center - test

def center_last(X):
    return X - X.mean(axis=-1, keepdims=True)

def deconf_center(root, center):
    acts = np.load(f"{root}/acts_deception_balanced.npy")
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(acts, yd, grp))
    bl = int(np.argmax([LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], yd[tr]).score(acts[te, L, :], yd[te]) for L in range(acts.shape[1])]))
    Xfull = acts[:, bl, :]
    rows = json.load(open(f"{root}/p4_rows.json")); X4 = np.load(f"{root}/p4_acts.npy")
    if center:
        Xfull = center_last(Xfull); X4 = center_last(X4)    
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(Xfull[tr], yd[tr])
    nc = [i for i, r in enumerate(rows) if r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]==r["truth"]]
    hon = [i for i, r in enumerate(rows) if r["cond"]=="honest" and r["ans"]==r["truth"]]
    yc = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
    s = np.r_[probe.decision_function(X4[hon]), probe.decision_function(X4[nc])]
    return bl, roc_auc_score(yc, s)

print(f"{'basis':16s} {'centering':10s} best_layer de-confound")
print("-" * 52)
for name, root in [("no_processing", f"results_census/{M}"), ("from_pretrained", f"results/{M}")]:
    for ctr in (False, True):
        bl, dc = deconf_center(root, ctr)
        print(f"{name:16s} {'centered' if ctr else 'raw':10s} {bl:9d} {dc:.3f}")