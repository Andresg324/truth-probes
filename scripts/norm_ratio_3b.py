"""
Testing Qwen-3B only: for every layer in the recovery band, measure the per-item norm ratio
c = ||x_ablated|| / ||x_clean|| at the readout, and ask how much of the observed score change
the rescaling model predicts.

Usage:  python scripts/norm_ratio_3B.py [MODEL]
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

MODELS = {"3B": "Qwen/Qwen2.5-3B-Instruct", "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
          "gemma-2b": "google/gemma-2-2b-it", "0.5B": "Qwen/Qwen2.5-0.5B-Instruct"}
DEVICE, DTYPE = ("cuda" if torch.cuda.is_available() else "cpu"), torch.float16
SEC = "norm_ratio"
RESID_ONLY = lambda n: n.endswith("hook_resid_post")


def run(TAG):
    R = f"results/{TAG}"
    z = np.load(f"{R}/split.npz")
    bl, te, yd = int(z["best_layer"]), z["te_idx"], z["y_decep"]
    trva = np.concatenate([z["tr_idx"], z["va_idx"]])
    acts = np.load(f"{R}/acts_deception_balanced.npy")
    items = json.load(open("data/mixed.json"))

    model = HookedTransformer.from_pretrained(MODELS[TAG], device=DEVICE, dtype=DTYPE)
    manifest(TAG, model=MODELS[TAG], device=DEVICE, dtype=str(DTYPE), seed=0,
             script="norm_ratio")

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

    probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, bl, :], yd[trva])
    w, b = probe.coef_[0], float(probe.intercept_[0])
    E = bl // 2
    crit = int(np.load(f"{R}/ablation_curve.npz")["crit"]) if os.path.exists(f"{R}/ablation_curve.npz") else None
    band = sorted(int(k[1:]) for k in json.load(open(f"{R}/RESULTS.json"))["phase2/ablation_curve"])

    Xc = acts[te, bl, :]
    n_clean = np.linalg.norm(Xc, axis=1)
    s_clean = Xc @ w + b
    y = yd[te]
    acc_clean = ((s_clean > 0).astype(int) == y).mean()
    auc_clean = roc_auc_score(y, s_clean)
    print(f"\n=== {TAG} === readout L{bl}, crit L{crit}, clean acc {acc_clean:.3f} "
          f"AUC {auc_clean:.3f}, intercept {b:+.3f}")

    def zero_hook(v, hook):
        return v * 0

    out = {}
    print(f"{'layer':>6} {'c(norm)':>9} {'acc drop':>9} {'AUC drop':>9} {'rescale R2':>11}")
    for L in band:
        Xa = []
        for i in te:
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", zero_hook)]):
                    _, c = model.run_with_cache(model.to_tokens(examples[i]),
                                                names_filter=RESID_ONLY)
            Xa.append(c["resid_post", bl][0, -1, :].float().cpu().numpy())
        Xa = np.array(Xa)

        n_abl = np.linalg.norm(Xa, axis=1)
        cvec = n_abl / n_clean                       # per-item rescaling factor
        s_abl = Xa @ w + b
        # rescaling model: if the ablation only shrinks the stream, the score becomes
        # c*(clean - b) + b.  R2 of that prediction against what we actually observe.
        pred = cvec * (s_clean - b) + b
        ss_res = float(((s_abl - pred) ** 2).sum())
        ss_tot = float(((s_abl - s_abl.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        acc_drop = acc_clean - ((s_abl > 0).astype(int) == y).mean()
        auc_drop = auc_clean - roc_auc_score(y, s_abl)
        out[f"L{L}"] = {"c_mean": round(float(cvec.mean()), 4),
                        "c_std": round(float(cvec.std()), 4),
                        "acc_drop": round(float(acc_drop), 4),
                        "auc_drop": round(float(auc_drop), 4),
                        "rescale_r2": round(float(r2), 4)}
        star = " <- crit" if L == crit else ""
        print(f"{L:>6} {cvec.mean():>9.4f} {acc_drop:>9.4f} {auc_drop:>9.4f} {r2:>11.4f}{star}")

    Ls = list(out)
    one_minus_c = np.array([1 - out[k]["c_mean"] for k in Ls])
    accs = np.array([out[k]["acc_drop"] for k in Ls])
    aucs = np.array([out[k]["auc_drop"] for k in Ls])
    corr = lambda a, v: float(np.corrcoef(a, v)[0, 1]) if len(a) > 2 and a.std() > 0 and v.std() > 0 else float("nan")
    summary = {"corr_1minusc_accdrop": round(corr(one_minus_c, accs), 4),
               "corr_1minusc_aucdrop": round(corr(one_minus_c, aucs), 4),
               "rescale_r2_at_crit": out.get(f"L{crit}", {}).get("rescale_r2"),
               "intercept": round(b, 4), "readout_layer": bl, "crit_layer": crit,
               "clean_acc": round(float(acc_clean), 4), "clean_auc": round(float(auc_clean), 4)}

    r2c = summary["rescale_r2_at_crit"]
    if r2c is None:
        summary["verdict"] = "CRIT LAYER NOT IN BAND"
    elif r2c >= 0.90:
        summary["verdict"] = "RESCALING DOMINATES (de-calibration has a mechanical explanation)"
    elif r2c <= 0.50:
        summary["verdict"] = "NOT RESCALING (de-calibration needs another explanation)"
    else:
        summary["verdict"] = "PARTIAL"
    print(f"\n  corr(1-c, acc drop) {summary['corr_1minusc_accdrop']:+.3f} | "
          f"corr(1-c, AUC drop) {summary['corr_1minusc_aucdrop']:+.3f}")
    print(f"  rescale R2 at crit: {r2c} -> {summary['verdict']}")

    for k, v in out.items():
        report(TAG, SEC, k, v)
    for k, v in summary.items():
        report(TAG, SEC, k, v)
    json.dump({"per_layer": out, "summary": summary},
              open(f"{R}/NORM_RATIO.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["3B"]):
        run(t)