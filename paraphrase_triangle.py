#------------- Imports ----------------------
import json, os, warnings
import numpy as np
import torch
import sys

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm
from sklearn.model_selection import GroupShuffleSplit
from reporting import report, manifest

warnings.filterwarnings("ignore")

# ---- model (identical to deception_probe.py) ----
MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B"  : "Qwen/Qwen2.5-3B-Instruct"  ,
    "7B"  : "Qwen/Qwen2.5-7B-Instruct"  ,
    "14B" : "Qwen/Qwen2.5-14B-Instruct" ,
    "gemma-2b": "google/gemma-2-2b-it"  ,
    "gemma-9b": "google/gemma-2-9b-it"  ,
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16

TAG =  sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"
PARA = sys.argv[2] if len(sys.argv) > 2 else "tf"

SEC = f"paraphrase_{PARA}"

loader = HookedTransformer.from_pretrained
model = loader(MODELS[TAG], device=device, dtype=DTYPE)

RESULTS = f"results/{TAG}"; os.makedirs(RESULTS, exist_ok=True)
RESID_ONLY = lambda name: name.endswith("hook_resid_post")
print(f"Model loaded: {TAG} | paraphrase: {PARA}")
manifest(TAG, fname=f"MANIFEST_para_{PARA}.json", model=MODELS[TAG], device=device, dtype=str(DTYPE), seed=0, script="paraphrase_triangle", para = PARA)

# ------ Paraphrase templates (user-content only, chat template applied after)
PARAPHRASES = {
    "orig":  lambda s: f"Statement: {s}\nIs this statement true? Answer Yes or No.",
    "claim": lambda s: f"Consider the claim: {s}\nIs it correct? Respond Yes or No.",
    "tf":    lambda s: f'"{s}"\nTrue or false? Reply Yes or No.',
}
user_content = PARAPHRASES[PARA]

def build_example(statement, forced_answer):
    m = [{"role": "user", "content": user_content(statement)}]
    return model.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + forced_answer

def nested_layer(acts3d, y, groups, seed=0, val_seeds=range(10), tol=0.02):
    """3-way grouped split, choose the best layer on VAL, never on the reported TEST fold"""
    trv, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=seed).split(acts3d, y, groups))
    layer_val = np.zeros(acts3d.shape[1])
    for s in val_seeds:
        tr_r, va_r = next(GroupShuffleSplit(1, test_size=0.25, random_state=s).split(acts3d[trv], y[trv], groups[trv]))
        tr, va = trv[tr_r], trv[va_r]
        for L in range(acts3d.shape[1]):
            layer_val[L] += LogisticRegression(max_iter=2000, C=0.1).fit(acts3d[tr, L, :], y[tr]).score(acts3d[va, L, :], y[va])
    layer_val /= len(val_seeds)
    peak = layer_val.max()
    bl = int(np.argmax(layer_val >= peak - tol))
    tr_r, va_r = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(acts3d[trv], y[trv], groups[trv]))
    return bl, trv[tr_r], trv[va_r], te



# ---- examples (identical order) ----
items = json.load(open("data/mixed.json"))
if os.environ.get("SMOKE"): items = items[:20]

examples = []
for d in items:
    stmt, is_true = d["statement"], d["label"]
    examples.append({"prompt": build_example(stmt, "Yes" if is_true else "No"),  "deceptive": 0, "answer": "Yes" if is_true else "No" })
    examples.append({"prompt": build_example(stmt, "No"  if is_true else "Yes"), "deceptive": 1, "answer": "No"  if is_true else "Yes"})
y_decep = np.array([e["deceptive"] for e in examples])
y_truth = np.array([items[i // 2]["label"] for i in range(len(examples))])
y_pol = np.array([1 if examples[i]["answer"] == "Yes" else 0 for i in range(len(examples))])
groups = np.array([i // 2 for i in range(len(examples))])

def gsplit(y, seed = 0, test_size = 0.25):
    return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(np.zeros(len(y)), y, groups))

# Extract all-layer acts at the forced-answer token, UNDER THE PARAPHRASE
CACHE = f"{RESULTS}/acts_paraphrase_{PARA}.npy"
if os.path.exists(CACHE):
    acts = np.load(CACHE); print("loaded cached paraphrase acts", acts.shape)
else:
    acts = []
    for e in examples:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(e["prompt"]), names_filter=RESID_ONLY)
        acts.append([c["resid_post", L][0,-1,:].cpu().numpy() for L in range(model.cfg.n_layers)])
    acts = np.array(acts); np.save(CACHE, acts); print("extracted paraphrase acts", acts.shape)

# ---- best layer + probe + direction
best_layer, tr_idx, va_idx, te_idx = nested_layer(acts, y_decep, groups)
te_groups = groups[te_idx]
uniq_g = np.unique(te_groups)
g_rows = {g: np.where(te_groups == g)[0] for g in uniq_g}
report(TAG, SEC, "best_layer", int(best_layer))

def grouped_ix(rng):
    gs = rng.choice(uniq_g, len(uniq_g), replace=True)
    return np.concatenate([g_rows[g] for g in gs])

trva = np.concatenate([tr_idx, va_idx])
probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, best_layer, :], y_decep[trva])
best_dir_t = torch.tensor(probe.coef_[0] / norm(probe.coef_[0]), dtype=DTYPE, device=device)
layer_probes = {L: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, L, :], y_decep[tr_idx])
                for L in range(model.cfg.n_layers)}

DECEP = [i for i in te_idx if examples[i]["deceptive"] == 1]
E = best_layer // 2
acc_te = probe.score(acts[te_idx, best_layer, :], y_decep[te_idx])
print(f"[{PARA}] best_layer = {best_layer} probe acc{acc_te:.3f} n_decep_eval={len(DECEP)}")
report(TAG, SEC, "probe_acc", float(acc_te), n=int(len(te_idx)))

# ----- hooks + bootstrap -------------
def sub_hook(alpha, d_t):
    def h(v, hook): v[:, :, :] = v[:, :, :] - alpha * d_t; return v
    return h

def zero_hook():
    def h(v, hook): return v*0
    return h

def boot_ci(vals, n=1000):
    vals = np.array(vals); rng = np.random.default_rng(0)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def detect_at_best(hooks):
    preds = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        preds.append(layer_probes[best_layer].predict(c["resid_post", best_layer][0, -1, :].cpu().numpy().reshape(1, -1))[0])
    return np.array(preds)

def det_after_ablate(ablate_L, read_L, idxs):
    hooks = [(f"blocks.{ablate_L}.hook_mlp_out", zero_hook())]
    preds = []
    for i in idxs:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        preds.append(layer_probes[read_L].predict(c["resid_post", read_L][0, -1, :].cpu().numpy().reshape(1, -1))[0])
    return np.array(preds)

def det_clean(read_L, idxs):
    return np.array([layer_probes[read_L].predict(acts[i, read_L].reshape(1, -1))[0] for i in idxs])

def score_after_ablate(ablate_L, read_L, idxs):
    hooks = [(f"blocks.{ablate_L}.hook_mlp_out", zero_hook())]
    s = []
    for i in idxs:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        s.append(layer_probes[read_L].decision_function(c["resid_post", read_L][0, -1, :].cpu().numpy().reshape(1, -1))[0])
    return float(np.mean(s))

def score_clean(read_L, idxs):
    return float(np.mean([layer_probes[read_L].decision_function(acts[i, read_L].reshape(1, -1))[0] for i in idxs]))

def auc_pair(pr, Xc, Xa, y, n=1000):
    sc, sa = pr.decision_function(Xc), pr.decision_function(Xa)
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = grouped_ix(rng)
        if len(np.unique(y[ix])) < 2: continue
        b.append(roc_auc_score(y[ix], sa[ix]))
    return roc_auc_score(y, sc), roc_auc_score(y, sa), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def acc_ci(pr, X, y, n=1000):
    correct = (pr.predict(X) == y).astype(float)
    rng = np.random.default_rng(0)
    b = [correct[grouped_ix(rng)].mean() for _ in range(n)]
    return correct.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# collapse sweep
alphas = [2, 4, 6, 8, 12, 16, 24]
collapse = {a: detect_at_best([(f"blocks.{E}.hook_resid_post", sub_hook(a, best_dir_t))]).mean() for a in alphas}
print("collapse sweep:", {a: round(v, 2) for a, v in collapse.items()})
report(TAG, SEC, "collapse_sweep", {str(a): float(v) for a, v in collapse.items()})

A2 = float(next((a for a in alphas if collapse[a] < 0.10), alphas[-1]))
print(f"min_alpha (paraphrase = {A2}")
report(TAG, SEC, "min_alpha", float(A2))

# Necessity -------------
R = list(range(E + 1, best_layer + 1))
sel_dec = [i for i in tr_idx if examples[i]["deceptive"] == 1][:120]
det_base = {L: det_clean(L, sel_dec).mean() for L in R}
cand = [L for L in R if det_base[L] > 0.70]
if not cand: cand = [max(det_base, key=det_base.get)]
local_drop = {L: score_clean(L, sel_dec) - score_after_ablate(L, L, sel_dec) for L in cand}
crit_L = max(local_drop, key=local_drop.get)
report(TAG, SEC, "crit_L", int(crit_L))

bl_loc = boot_ci(det_clean(crit_L, DECEP)); nec_loc = boot_ci(det_after_ablate(crit_L, crit_L, DECEP))
bl_dep = boot_ci(det_clean(best_layer, DECEP)); nec_dep = boot_ci(det_after_ablate(crit_L, best_layer, DECEP))
print(f"\n[{PARA}] crit MLP L{crit_L} (unified rule, chosen on train-fold items)")
print(f" (Don't Cite) Local (read L{crit_L}): {bl_loc[0]:.2f} [{bl_loc[1]:.2f}, {bl_loc[2]:.2f}] -> {nec_loc[0]:.2f} [{nec_loc[1]:.2f}, {nec_loc[2]:.2f}]")
print(f" (Don't Cite) Deployed (read L{best_layer}): {bl_dep[0]:.2f} [{bl_dep[1]:.2f}, {bl_dep[2]:.2f}] -> {nec_dep[0]:.2f} [{nec_dep[1]:.2f}, {nec_dep[2]:.2f}]")

ctrl_Ls = sorted([L for L in cand if L != crit_L], key = lambda L: local_drop[L])[:3]
km, ctrl_L = None, crit_L
for j, cL in enumerate(ctrl_Ls):
    m, lo, hi = boot_ci(det_after_ablate(cL, cL, DECEP))
    print(f" non-crit control L{cL} (local): {m:.2f} [{lo:.2f}, {hi:.2f}]")
    if j == 0: km, ctrl_L = m, cL
base_m, cm = bl_dep[0], nec_dep[0]
if km is None: km = cm
report(TAG, SEC, "control", {"layer": int(ctrl_L), "detection_recall": float(km)})

Xa_ctrl = []
for i in te_idx:
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{ctrl_L}.hook_mlp_out", zero_hook())]):
            _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
    Xa_ctrl.append(c["resid_post", ctrl_L][0, -1, :].cpu().numpy())
Xa_ctrl = np.array(Xa_ctrl)
cac, caa, clo2, chi2 = auc_pair(layer_probes[ctrl_L], acts[te_idx, ctrl_L, :], Xa_ctrl, y_decep[te_idx])
print(f" non-crit control L{ctrl_L} AUC: {cac:.2f} -> {caa:.2f} [{clo2:.2f}, {chi2:.2f}]")
report(TAG, SEC, "control_auc", {"layer": int(ctrl_L), "clean": cac, "abl": caa, "ci": [clo2, chi2]})

# --- Sufficiency
def recovery_patch_clean(L):
    clean = {}
    def save(v, hook): clean["v"] = v.clone(); return v
    def steer(v, hook): v[:, :, :] = v[:, :, :] - A2 * best_dir_t; return v
    def paste(v, hook): return clean["v"]
    preds = []
    for i in DECEP:
        tok = model.to_tokens(examples[i]["prompt"])
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", save)]):
                model.run_with_cache(tok, names_filter=RESID_ONLY)
            with model.hooks(fwd_hooks=[(f"blocks.{E}.hook_resid_post", steer), (f"blocks.{L}.hook_mlp_out", paste)]):
                _, c = model.run_with_cache(tok, names_filter=RESID_ONLY)
        preds.append(layer_probes[best_layer].predict(c["resid_post", best_layer][0,-1,:].cpu().numpy().reshape(1, -1))[0])
    return np.array(preds)

nm, nlo, nhi = boot_ci(detect_at_best([(f"blocks.{E}.hook_resid_post", sub_hook(A2, best_dir_t))]))
sm, slo, shi = boot_ci(recovery_patch_clean(crit_L))

print(f"\nSUFFICIENCY steered, no patch: {nm:.2f} [{nlo:.2f},{nhi:.2f}]")
print(f"SUFFICIENCY + clean L{crit_L} patch: {sm:.2f} [{slo:.2f}, {shi:.2f}] (restoration)")
report(TAG, SEC, "sufficiency", {"steered": nm, "patched": sm})

# Specificity 

labs = {"deception": y_decep, "truth": y_truth, "polarity": y_pol}
Ploc = {nm: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, crit_L, :], lab[tr_idx]) for nm, lab in [("deception", y_decep), ("truth", y_truth), ("polarity", y_pol)]}
Xabl_loc = []
Xabl_dep = []
for i in te_idx:
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{crit_L}.hook_mlp_out", zero_hook())]):
            _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
    Xabl_loc.append(c["resid_post", crit_L][0, -1, :].cpu().numpy())
    Xabl_dep.append(c["resid_post", best_layer][0, -1, :].cpu().numpy())
Xabl_loc = np.array(Xabl_loc)
Xabl_dep = np.array(Xabl_dep)

dep_ac, dep_aa, dep_lo, dep_hi = auc_pair(layer_probes[best_layer], acts[te_idx, best_layer, :], Xabl_dep, labs["deception"][te_idx])
print(f" Deployed AUC (read L{best_layer}): {dep_ac:.2f} -> {dep_aa:.2f} [{dep_lo:.2f}, {dep_hi:.2f}]")
report(TAG, SEC, "necessity_deployed_auc", {"clean": dep_ac, "abl": dep_aa},
       ci = [dep_lo, dep_hi])

print(f"\nSpecificity at LOCAL readout L{crit_L} - AUC (clean -> ablated):")
spec_auc = {}
for nmr in ["deception", "truth", "polarity"]:
    ac, aa, lo, hi = auc_pair(Ploc[nmr], acts[te_idx, crit_L, :], Xabl_loc, labs[nmr][te_idx])
    spec_auc[nmr] = {"clean": ac, "abl": aa, "ci": [lo, hi]}
    print(f" {nmr:9}: AUC {ac:.2f} -> {aa:.2f} [{lo:.2f}, {hi:.2f}]")
report(TAG, SEC, "specificity_local_auc", spec_auc)

dc_m, dc_lo, dc_hi = acc_ci(layer_probes[best_layer], acts[te_idx, best_layer, :], labs["deception"][te_idx])
da_m, da_lo, da_hi = acc_ci(layer_probes[best_layer], Xabl_dep, labs["deception"][te_idx])
print(f" Deployed (balanced acc, read L{best_layer}): "
      f" {dc_m:.2f} [{dc_lo:.2f}, {dc_hi:.2f}] -> {da_m:.2f} [{da_lo:.2f}, {da_hi:.2f}]")
report(TAG, SEC, "necessity_deployed_acc", {"clean": dc_m, "abl": da_m},
       ci = {"clean": [dc_lo, dc_hi], "abl": [da_lo, da_hi]})

print(f"\nSpecificity at LOCAL readout L{crit_L} (clean -> ablated), held-out:")
spec_acc = {}
for nmr in ["deception", "truth", "polarity"]:
    c_mean, clo, chi = acc_ci(Ploc[nmr], acts[te_idx, crit_L, :], labs[nmr][te_idx])
    a_mean, alo, ahi = acc_ci(Ploc[nmr], Xabl_loc, labs[nmr][te_idx])
    spec_acc[nmr] = {"clean": c_mean, "abl": a_mean, "ci": [alo, ahi]}
    print(f" {nmr:9}: {c_mean:.2f} [{clo:.2f}, {chi:.2f}] -> {a_mean:.2f} [{alo:.2f}, {ahi:.2f}]")
report(TAG, SEC, "specificity_local_acc", spec_acc)
    

print(f"""
================= TRIANGLE @ paraphrase='{PARA}', model={TAG} =================
  crit MLP under this paraphrase: L{crit_L} (compare to the phase 2 orig-format crit)
  NECESSITY (acc): {dc_m:.2f} -> {da_m:.2f}
  NECESSITY (AUC): {dep_ac:.2f} -> {dep_aa:.2f} (Cite this one)
  CONTROL        : {base_m:.2f} -> {km:.2f} non-crit L{ctrl_L}
  SUFFICIENCY    : {nm:.2f} -> {sm:.2f} clean patch restores?

  SPECIFICITY.  : polarity should survive (see above)
  Reading: crit collapses + control holds + patch restores + polarity survives
    => mechanism MOVED to L{crit_L} (triangle replicates) => thesis checks out
    no layer collapses means the paraphrase breaks the mechanism
      """)