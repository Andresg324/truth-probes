"""Composition-matched de-confound. CPU only, no GPU, seconds to run.

The de-confound statistic separates honest-condition truthful responses from
lie-condition truthful (non-compliant) responses. Under denial bias those two groups
differ in composition: a model that lies only by denying truths is truthful under the
lie instruction only on FALSE statements, while its honest truthful responses span both
truth values. A probe reading polarity would then separate the groups without reading
the instruction at all.

This script recomputes the statistic holding composition fixed:
  deconfound_all         the published value (reproduction check)
  deconfound_false_only  both groups restricted to false statements
  deconfound_true_only   both groups restricted to true statements (often empty)
  instruction_coef       OLS of standardized score on instruction + truth + polarity,
                         over truthful responses only

EXPLORATORY. Added after the pre-specified analysis, in response to review. Not covered
by any registration; reported as a robustness check, not a replacement.

Usage:  python scripts/deconfound_matched.py
"""

import json, os, warnings
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from reporting import report

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TAGS = ["0.5B", "1.5B", "3B", "7B", "14B", "gemma-2b", "gemma-9b", "llama-3b", "llama-8b"]
MIN_N = 5


def auc_or_nan(y, s):
    return roc_auc_score(y, s) if len(set(y.tolist())) == 2 else float("nan")


def run(tag):
    R = f"results/{tag}"
    if not (os.path.exists(f"{R}/p4_rows.json") and os.path.exists(f"{R}/split.npz")):
        print(f"{tag:9s} (no transfer data)")
        return None

    acts = np.load(f"{R}/acts_deception_balanced.npy")
    z = np.load(f"{R}/split.npz")
    bl, yd = int(z["best_layer"]), z["y_decep"]
    trva = np.concatenate([z["tr_idx"], z["va_idx"]])
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, bl, :], yd[trva])

    rows = json.load(open(f"{R}/p4_rows.json"))
    X4 = np.load(f"{R}/p4_acts.npy")

    # the two groups, exactly as in the published statistic
    hon = [i for i, r in enumerate(rows) if r["cond"] == "honest" and r["ans"] == r["truth"]]
    nc  = [i for i, r in enumerate(rows) if r["cond"] == "lie"
           and r["ans"] in ("yes", "no") and r["ans"] == r["truth"]]

    def deconf(h_ix, n_ix):
        if len(h_ix) < MIN_N or len(n_ix) < MIN_N:
            return float("nan"), len(h_ix), len(n_ix)
        y = np.r_[np.zeros(len(h_ix)), np.ones(len(n_ix))]
        s = np.r_[probe.decision_function(X4[h_ix]), probe.decision_function(X4[n_ix])]
        return auc_or_nan(y, s), len(h_ix), len(n_ix)

    def restrict(ix, truth_val):
        return [i for i in ix if rows[i]["stmt_true"] == truth_val]

    d_all, nh_a, nn_a = deconf(hon, nc)
    d_false, nh_f, nn_f = deconf(restrict(hon, 0), restrict(nc, 0))
    d_true,  nh_t, nn_t = deconf(restrict(hon, 1), restrict(nc, 1))

    # polarity-controlled: instruction coefficient over truthful responses only
    keep = hon + nc
    s = probe.decision_function(X4[keep])
    s = (s - s.mean()) / (s.std() + 1e-9)
    instr = np.array([1.0 if rows[i]["cond"] == "lie" else 0.0 for i in keep])
    truth = np.array([float(rows[i]["stmt_true"]) for i in keep])
    polar = np.array([1.0 if rows[i]["ans"] == "yes" else 0.0 for i in keep])
    D = sm.add_constant(np.column_stack([instr, truth, polar]))
    cond = float(np.linalg.cond(D))
    if np.linalg.matrix_rank(D) < D.shape[1] or cond > 1e6:
        coef, pval = float("nan"), float("nan")     # polarity collinear with instruction
    else:
        m = sm.OLS(s, D).fit()
        coef, pval = float(m.params[1]), float(m.pvalues[1])

    res = {
        "deconfound_all": round(float(d_all), 4),
        "deconfound_false_only": round(float(d_false), 4),
        "deconfound_true_only": round(float(d_true), 4),
        "n_honest_all": nh_a, "n_noncompliant_all": nn_a,
        "n_false_cell": [nh_f, nn_f], "n_true_cell": [nh_t, nn_t],
        "instruction_coef_polarity_controlled": None if np.isnan(coef) else round(coef, 4),
        "instruction_p": None if np.isnan(pval) else round(pval, 6),
        "design_cond": round(cond, 1),
    }
    print(f"{tag:9s} all {d_all:6.3f} | false-only {d_false:6.3f} (n={nh_f},{nn_f}) | "
          f"true-only {d_true:6.3f} (n={nh_t},{nn_t}) | "
          f"instr coef {'n/a' if np.isnan(coef) else f'{coef:+.3f}'} (cond {cond:.0f})")

    for k, v in res.items():
        report(tag, "deconfound_matched", k, v)
    return res


if __name__ == "__main__":
    hdr = (f"{'model':9s} {'all':>10s} {'false-only':>22s} {'true-only':>22s} "
           f"{'polarity-controlled':>24s}")
    print(hdr); print("-" * len(hdr))
    out = {t: run(t) for t in TAGS}
    print("\nRead: values near 0.5 mean the probe reads the answer, not the instruction.")
    print("false-only and true-only hold truth and polarity fixed by construction, so any")
    print("separation there cannot come from statement composition. n/a in the last column")
    print("means polarity is collinear with instruction in that model (extreme denial bias),")
    print("which is itself the finding.")