import json, numpy as np, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

TAG = "gemma-2b"
rows = json.load(open(f"results/{TAG}/p4_rows.json"))
X4 = np.load(f"results/{TAG}/p4_acts.npy")

# Rebuild frozen probe
items = json.load(open("data/mixed.json"))
yd = np.array([j for _ in items for j in (0, 1)]); grp = np.arange(len(yd))//2
acts = np.load(f"results/{TAG}/acts_deception_balanced.npy")
tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(acts, yd, grp))
bl = int(np.argmax([LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], yd[tr]).score(acts[te,L,:], yd[te]) for L in range(acts.shape[1])]))
probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr,bl,:], yd[tr])

keep = [i for i, r in enumerate(rows)
        if (r["cond"]=="honest" and r["ans"]==r["truth"]) or (r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]!=r["truth"])]
Xk = X4[keep]

truth = np.array([rows[i]["stmt_true"] for i in keep])
polarity = np.array([1 if rows[i]["ans"]=="yes" else 0 for i in keep])
decep = np.array([1 if rows[i]["cond"]=="lie" else 0 for i in keep])

s = probe.decision_function(Xk)

# regression: does the probe score track deception

D = sm.add_constant(np.column_stack([truth, polarity, truth*polarity]))
m = sm.OLS(s, D).fit()
print(m.summary(xname=["const", "truth", "polarity", "truthxpolarity(=Deception)"]))

print("\nper-cell deception AUC (fixing truth & polarity):")
print(f"\nAUC vs deception : {roc_auc_score(decep, s):.3f} (chance = probe doesn't read it)")
print(f"AUC vs truth       : {roc_auc_score(truth, s):.3f}")
print(f"AUC vs polarity    : {roc_auc_score(polarity, s):.3f}")
print(f"\nn = {len(keep)} lies from TRUE stmts: {int(((decep==1)&(truth==1)).sum())}, "
      f"from FALSE: {int(((decep==1)&(truth==0)).sum())}")