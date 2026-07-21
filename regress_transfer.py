import json, numpy as np, statsmodels.api as sm, sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from reporting import report, manifest

TAG = sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"
ROOT = f"results/{TAG}"

print("reading from:", ROOT)
rows = json.load(open(f"{ROOT}/p4_rows.json"))
X4 = np.load(f"{ROOT}/p4_acts.npy")

manifest(TAG, script="regress_transfer")


# Rebuild frozen probe
acts = np.load(f"{ROOT}/acts_deception_balanced.npy")
z = np.load(f"{ROOT}/split.npz")
bl, te, yd = int(z["best_layer"]), z["te_idx"], z["y_decep"]
trva = np.concatenate([z["tr_idx"], z["va_idx"]])
probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, bl, :], yd[trva])

mask = [i for i, r in enumerate(rows) if r["ans"] in ("yes", "no")]
s_all = probe.decision_function(X4[mask])
instr = np.array([1 if rows[i]["cond"] == "lie" else 0 for i in mask])
adecep = np.array([1 if rows[i]["ans"] != rows[i]["truth"] else 0 for i in mask])
D = sm.add_constant(np.column_stack([instr, adecep]))
mreg = sm.OLS(s_all, D).fit()
report(TAG, "regress_transfer", "leak_model",
       {"instruction": float(mreg.params[1]), "instruction_p": float(mreg.pvalues[1]),
        "actual_deception": float(mreg.params[2]), "actual_deception_p": float(mreg.pvalues[2])},
        n = int(len(mask)))

print(mreg.summary(xname=["const", "instruction (=leak)", "actual_deception"]))
print("LEAK = coefficient on 'instruction' holding actual deception fixed. ~0 = clean, large + significant = the probe reads the prompt")

nc = [i for i, r in enumerate(rows) if r["cond"]=="lie" and r["ans"] in ("yes", "no") and r["ans"]==r["truth"]]
hon = [i for i, r in enumerate(rows) if r["cond"]=="honest" and r["ans"]==r["truth"]]
if len(nc) >= 5:
    s_nc, s_hon = probe.decision_function(X4[nc]), probe.decision_function(X4[hon])
    y = np.r_[np.zeros(len(hon)), np.ones(len(nc))]
    dc = roc_auc_score(y, np.r_[s_hon, s_nc])
    report(TAG, "regress_transfer", "deconfound_auc", float(dc), n=len(nc))
    print(f"\nDE-CONFOUND (non-compliant lies = told to lie, told truth anyway; n={len(nc)}):")
    print(f" probe AUC(honest vs. told-to-lie-but-truthful) = {dc:.3f}")
    print(" ~0.5 -> probe reads the ANSWER => (clean test)")
    print(" high -> probe reads the INSTRUCTION => (it's a context leak, drop it from this model)")
else:
    print(f"\nDE-CONFOUND: only {len(nc)} non-compliant lies - too few to test")

pc = roc_auc_score(yd[te], probe.decision_function(acts[te, bl, :]))
print(f"POSITIVE CONTROL (held out teacher forced) AUC = {pc:.3f}")
print(" transfer AUC at/above this = confound, not a result\n")
report(TAG, "regress_transfer", "positive_control_auc", float(pc), n=int(len(te)))

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

report(TAG, "regress_transfer", "polarity_controlled",
       {"truth": float(m.params[1]), "polarity": float(m.params[2]), "interaction": float(m.params[3]),
        "truth_p": float(m.pvalues[1]), "polarity_p": float(m.pvalues[2]), "interaction_p": float(m.pvalues[3])},
        n=int(len(keep)))
report(TAG, "regress_transfer", "subgroup_aucs",
       {"deception": float(roc_auc_score(decep, s)), "truth": float(roc_auc_score(truth, s)),
        "polarity": float(roc_auc_score(polarity, s))})

print("\nper-cell deception AUC (fixing truth & polarity):")
print(f"\nAUC vs deception : {roc_auc_score(decep, s):.3f} (chance = probe doesn't read it)")
print(f"AUC vs truth       : {roc_auc_score(truth, s):.3f}")
print(f"AUC vs polarity    : {roc_auc_score(polarity, s):.3f}")
print(f"\nn = {len(keep)} lies from TRUE stmts: {int(((decep==1)&(truth==1)).sum())}, "
      f"from FALSE: {int(((decep==1)&(truth==0)).sum())}")