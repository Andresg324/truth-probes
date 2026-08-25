"""
phase2.py judges each cell by whether a grouped-bootstrap interval excludes zero, with no
correction across the cells tested in a model. It saves interval endpoints, not draws, so an
exact procedure cannot be run from the released artifacts. This script recomputes the same
statistic with the draws retained, derives a bootstrap p-value per cell, and applies
Benjamini-Hochberg within each model. It supersedes the normal-approximation Bonferroni proxy
in scripts/multiplicity_check.py, which is retained for provenance.

Everything below reproduces phase2's spec_at() exactly: three probes (deception, truth,
polarity) fit on the TRAIN fold at the ablated layer, evaluated on the held-out fold, paired
asymmetry = (truth drop) - (deception drop), grouped bootstrap over statements, 1000 draws,
seed 0.

Usage:  python scripts/fdr_specificity.py [MODEL ...]      default: all four curve models
"""

import json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer
from reporting import report, manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MODELS = {"0.5B": "Qwen/Qwen2.5-0.5B-Instruct", "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
          "3B": "Qwen/Qwen2.5-3B-Instruct", "gemma-2b": "google/gemma-2-2b-it"}
CURVE = ["0.5B", "1.5B", "3B", "gemma-2b"]
DEVICE, DTYPE = ("cuda" if torch.cuda.is_available() else "cpu"), torch.float16
SEC, NBOOT, Q = "fdr_specificity", 1000, 0.05
RESID_ONLY = lambda n: n.endswith("hook_resid_post")


def benjamini_hochberg(pvals, q=Q):
    """Return boolean mask of rejections under BH at level q."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        keep[order[: kmax + 1]] = True
    return keep


def run(TAG):
    R = f"results/{TAG}"
    z = np.load(f"{R}/split.npz")
    tr, te, yd = z["tr_idx"], z["te_idx"], z["y_decep"]
    groups = z["groups"]
    acts = np.load(f"{R}/acts_deception_balanced.npy")
    items = json.load(open("data/mixed.json"))
    crit = int(np.load(f"{R}/ablation_curve.npz")["crit"])
    band = sorted(int(k[1:]) for k in
                  json.load(open(f"{R}/RESULTS.json"))["phase2/ablation_curve"])

    # labels, exactly as phase2 builds them
    y_truth = np.array([items[i // 2]["label"] for i in range(acts.shape[0])]).astype(int)
    y_pol = np.array([1 if (i % 2 == 0) == bool(items[i // 2]["label"]) else 0
                      for i in range(acts.shape[0])])
    # even index = honest example, so its answer is Yes iff the statement is true
    labs = {"deception": yd, "truth": y_truth, "polarity": y_pol}

    model = HookedTransformer.from_pretrained(MODELS[TAG], device=DEVICE, dtype=DTYPE)
    manifest(TAG, model=MODELS[TAG], device=DEVICE, dtype=str(DTYPE), seed=0, script=SEC)

    def build(stmt, ans):
        m = [{"role": "user",
              "content": f"Statement: {stmt}\nIs this statement true? Answer Yes or No."}]
        return model.tokenizer.apply_chat_template(m, tokenize=False,
                                                   add_generation_prompt=True) + ans

    examples = []
    for d in items:
        examples.append(build(d["statement"], "Yes" if d["label"] else "No"))
        examples.append(build(d["statement"], "No" if d["label"] else "Yes"))
    assert len(examples) == acts.shape[0]

    # grouped bootstrap indices over the held-out fold, as phase2
    te_groups = groups[te]
    uniq = np.unique(te_groups)
    g_rows = {g: np.where(te_groups == g)[0] for g in uniq}

    def grouped_ix(rng):
        gs = rng.choice(uniq, len(uniq), replace=True)
        return np.concatenate([g_rows[g] for g in gs])

    def zero_hook(v, hook):
        return v * 0

    ydv, ytv, ypv = yd[te], y_truth[te], y_pol[te]
    rows = {}
    print(f"\n=== {TAG} === crit L{crit}, {len(band)} cells, BH q={Q}")
    print(f"{'layer':>6} {'asym':>8} {'boot p':>10} {'95% CI':>22}")
    for L in band:
        Xa = []
        for i in te:
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", zero_hook)]):
                    _, c = model.run_with_cache(model.to_tokens(examples[i]),
                                                names_filter=RESID_ONLY)
            Xa.append(c["resid_post", L][0, -1, :].float().cpu().numpy())
        Xa = np.array(Xa)

        P = {nm: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], lab[tr])
             for nm, lab in labs.items()}
        Xc = acts[te, L, :]
        s = {nm: (P[nm].decision_function(Xc), P[nm].decision_function(Xa)) for nm in P}
        Y = {"deception": ydv, "truth": ytv, "polarity": ypv}
        auc = {nm: (roc_auc_score(Y[nm], s[nm][0]), roc_auc_score(Y[nm], s[nm][1])) for nm in P}
        point = (auc["truth"][0] - auc["truth"][1]) - (auc["deception"][0] - auc["deception"][1])

        rng = np.random.default_rng(0)
        ds = []
        for _ in range(NBOOT):
            ix = grouped_ix(rng)
            if len(np.unique(ydv[ix])) < 2 or len(np.unique(ytv[ix])) < 2:
                continue
            d = ((roc_auc_score(ytv[ix], s["truth"][0][ix]) -
                  roc_auc_score(ytv[ix], s["truth"][1][ix])) -
                 (roc_auc_score(ydv[ix], s["deception"][0][ix]) -
                  roc_auc_score(ydv[ix], s["deception"][1][ix])))
            ds.append(d)
        ds = np.array(ds)
        # two-sided bootstrap p, floored at 1/n since 0 draws beyond is not p=0
        frac = min((ds > 0).mean(), (ds < 0).mean())
        pval = max(2.0 * frac, 1.0 / len(ds))
        lo, hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

        rows[L] = {"asym": round(float(point), 4), "p": round(float(pval), 5),
                   "ci": [round(lo, 4), round(hi, 4)], "n_draws": int(len(ds))}
        np.save(f"{R}/bootdraws_L{L}.npy", ds)          # retained for any later procedure
        star = " <- crit" if L == crit else ""
        print(f"{L:>6} {point:>+8.3f} {pval:>10.5f} [{lo:>+7.3f}, {hi:>+7.3f}]{star}")

    Ls = sorted(rows)
    p = [rows[L]["p"] for L in Ls]
    keep_bh = benjamini_hochberg(p, Q)
    bonf = np.array(p) <= (Q / len(p))
    raw = np.array([(rows[L]["ci"][0] > 0) or (rows[L]["ci"][1] < 0) for L in Ls])

    summary = {"n_cells": len(Ls), "q": Q,
               "sig_uncorrected": [int(L) for L, k in zip(Ls, raw) if k],
               "sig_bh": [int(L) for L, k in zip(Ls, keep_bh) if k],
               "sig_bonferroni": [int(L) for L, k in zip(Ls, bonf) if k],
               "crit_layer": crit,
               "crit_significant_uncorrected": bool(raw[Ls.index(crit)]) if crit in Ls else None,
               "crit_survives_bh": bool(keep_bh[Ls.index(crit)]) if crit in Ls else None}
    print(f"  uncorrected {summary['sig_uncorrected']}")
    print(f"  BH q={Q}     {summary['sig_bh']}")
    print(f"  Bonferroni   {summary['sig_bonferroni']}")

    for L in Ls:
        report(TAG, SEC, f"L{L}", rows[L])
    for k, v in summary.items():
        report(TAG, SEC, k, v)
    json.dump({"per_layer": {f"L{L}": rows[L] for L in Ls}, "summary": summary},
              open(f"{R}/FDR_SPECIFICITY.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    out = {}
    for t in (sys.argv[1:] or CURVE):
        try:
            out[t] = run(t)
        except Exception as e:
            print(f"  FAILED {t}: {type(e).__name__}: {e}")
    print("\n" + "=" * 78)
    for t, s in out.items():
        print(f"{t:9s} {s['n_cells']:2d} cells | uncorrected {len(s['sig_uncorrected'])} "
              f"-> BH {len(s['sig_bh'])} -> Bonf {len(s['sig_bonferroni'])} | "
              f"crit {'survives BH' if s['crit_survives_bh'] else 'not significant under BH'}")