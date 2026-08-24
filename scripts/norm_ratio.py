"""PROTOCOL section 7: is the de-calibration a residual-norm rescaling artifact?

Rushing and Nanda attribute roughly 30% of measured self-repair to normalization rescaling.
The analogue here: if ablating an MLP shrinks the residual stream at the readout by a factor
c, a linear probe with intercept b reports c*(w.x) + b instead of (w.x) + b. Scores compress
toward the intercept, so a fixed decision threshold is effectively moved while rank order is
preserved exactly. That is the accuracy-falls-AUC-holds signature.

Two readouts:

  DEPLOYED (default)  reads at the model's readout layer with the deployed probe. This is the
                      registered section 7 procedure. Qwen-3B carries the registered verdict;
                      the other three are the contrast case.

  LOCAL_READOUT=1     reads at the critical layer with a probe fit at that layer on the train
                      fold, matching phase2's local specificity probe. This is the cell the
                      paper's Table 1 exhibit actually reports. EXPLORATORY: added after the
                      registered run, because section 7's procedure reads at the deployed
                      readout while its motivation refers to a local-readout quantity. Same
                      thresholds applied, but not a registered outcome.

Usage:
  python scripts/norm_ratio.py [MODEL ...]                  registered, default all four
  LOCAL_READOUT=1 python scripts/norm_ratio.py              exploratory
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
CURVE = ["0.5B", "1.5B", "3B", "gemma-2b"]     # 0.5B first: smallest, fastest smoke test
REGISTERED = "3B"                               # the model the section 7 verdict rules apply to
LOCAL = os.environ.get("LOCAL_READOUT") == "1"
SEC = "norm_ratio_local" if LOCAL else "norm_ratio"
OUTFILE = "NORM_RATIO_LOCAL.json" if LOCAL else "NORM_RATIO.json"
DEVICE, DTYPE = ("cuda" if torch.cuda.is_available() else "cpu"), torch.float16
RESID_ONLY = lambda n: n.endswith("hook_resid_post")


def run(TAG):
    R = f"results/{TAG}"
    z = np.load(f"{R}/split.npz")
    bl, te, yd = int(z["best_layer"]), z["te_idx"], z["y_decep"]
    tr = z["tr_idx"]
    trva = np.concatenate([z["tr_idx"], z["va_idx"]])
    acts = np.load(f"{R}/acts_deception_balanced.npy")
    items = json.load(open("data/mixed.json"))
    crit = int(np.load(f"{R}/ablation_curve.npz")["crit"])
    band = sorted(int(k[1:]) for k in
                  json.load(open(f"{R}/RESULTS.json"))["phase2/ablation_curve"])

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
    assert len(examples) == acts.shape[0], "examples do not match cached activations"

    # which readout, and which probe reads it
    if LOCAL:
        read_L = crit
        probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, crit, :], yd[tr])
    else:
        read_L = bl
        probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, bl, :], yd[trva])
    w, b = probe.coef_[0], float(probe.intercept_[0])

    Xc = acts[te, read_L, :]
    n_clean = np.linalg.norm(Xc, axis=1)
    s_clean = Xc @ w + b
    y = yd[te]
    acc_clean = ((s_clean > 0).astype(int) == y).mean()
    auc_clean = roc_auc_score(y, s_clean)

    # sanity: an unablated pass must reproduce the cached activations, or c is meaningless
    chk = []
    for i in te[:40]:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(examples[i]), names_filter=RESID_ONLY)
        chk.append(c["resid_post", read_L][0, -1, :].float().cpu().numpy())
    rel = float(np.abs(np.linalg.norm(np.array(chk), axis=1) - n_clean[:40]).max()
                / n_clean[:40].mean())
    assert rel < 1e-2, f"fresh pass does not reproduce cached acts (rel {rel:.4f})"

    mode = f"LOCAL readout L{read_L}" if LOCAL else f"DEPLOYED readout L{read_L}"
    print(f"\n=== {TAG} === {mode}, crit L{crit}, clean acc {acc_clean:.3f} "
          f"AUC {auc_clean:.3f}, intercept {b:+.3f}")
    if LOCAL:
        print(f"  (compare phase2/specificity_crit_* for this model: clean acc and AUC "
              f"should match the paper's Table 1 cell)")
    print(f"  cache check ok (max rel norm diff {rel:.5f})")

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
            Xa.append(c["resid_post", read_L][0, -1, :].float().cpu().numpy())
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
    corr = lambda a, v: (float(np.corrcoef(a, v)[0, 1])
                         if len(a) > 2 and a.std() > 0 and v.std() > 0 else float("nan"))
    r2c = out.get(f"L{crit}", {}).get("rescale_r2")
    summary = {"readout_mode": "local" if LOCAL else "deployed",
               "readout_layer": read_L, "crit_layer": crit,
               "corr_1minusc_accdrop": round(corr(one_minus_c, accs), 4),
               "corr_1minusc_aucdrop": round(corr(one_minus_c, aucs), 4),
               "rescale_r2_at_crit": r2c,
               "intercept": round(b, 4),
               "clean_acc": round(float(acc_clean), 4), "clean_auc": round(float(auc_clean), 4),
               "registered": (TAG == REGISTERED) and not LOCAL}

    if LOCAL:
        tail = "  [EXPLORATORY, local readout]"
    elif TAG == REGISTERED:
        tail = ""
    else:
        tail = "  [CONTRAST, not registered]"
    if r2c is None:
        summary["verdict"] = "CRIT LAYER NOT IN BAND" + tail
    elif r2c >= 0.90:
        summary["verdict"] = "RESCALING DOMINATES" + tail
    elif r2c <= 0.50:
        summary["verdict"] = "NOT RESCALING" + tail
    else:
        summary["verdict"] = "PARTIAL" + tail

    print(f"\n  corr(1-c, acc drop) {summary['corr_1minusc_accdrop']:+.3f} | "
          f"corr(1-c, AUC drop) {summary['corr_1minusc_aucdrop']:+.3f}")
    print(f"  rescale R2 at crit: {r2c} -> {summary['verdict']}")

    for k, v in out.items():
        report(TAG, SEC, k, v)
    for k, v in summary.items():
        report(TAG, SEC, k, v)
    json.dump({"per_layer": out, "summary": summary}, open(f"{R}/{OUTFILE}", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    todo = sys.argv[1:] or CURVE
    res = {}
    for t in todo:
        try:
            res[t] = run(t)
        except Exception as e:
            print(f"  FAILED {t}: {type(e).__name__}: {e}")
    print("\n" + "=" * 82)
    for t, s in res.items():
        print(f"{t:10s} [{s['readout_mode']:8s} L{s['readout_layer']:>2d}] "
              f"R2@crit {str(s['rescale_r2_at_crit']):>7s} | "
              f"corr(1-c,acc) {s['corr_1minusc_accdrop']:+.3f} | "
              f"corr(1-c,auc) {s['corr_1minusc_aucdrop']:+.3f} | {s['verdict']}")