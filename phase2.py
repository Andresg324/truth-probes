import json, os, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

from reporting import report, manifest
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

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

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16

device = DEVICE
SEC = "Phase2"
SIZE = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
TAG = SIZE
model_name = MODELS[SIZE]
model = HookedTransformer.from_pretrained(model_name, device=DEVICE, dtype=DTYPE)

RESULTS, FIGS = f"results/{TAG}", f"figures/{TAG}"

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

RESID_ONLY = lambda name: name.endswith("hook_resid_post")

LEGACY = "--legacy" in sys.argv

print("Model Loaded")
manifest(TAG, model = model_name, device=DEVICE, dtype=str(DTYPE), seed=0, script = "phase2") # Tracks which model is loaded

# Nested layer section
def nested_layer(acts3d, y, groups, seed=0, val_seeds=range(10), tol=0.02):
    # 3-way grouped split, choose the best layer on VAL, never on the reported TEST fold
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

# ---- prompts (identical) ----
def build_example(statement, forced_answer):
    messages = [{"role": "user", "content": f"Statement: {statement}\nIs this statement true? Answer Yes or No."}]
    chat = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat + forced_answer

def answer_prompt(statement):
    messages = [{"role": "user", "content": f'Statement: {statement}\nIs this statement true? Reply with exactly one word: "Yes" or "No".'}]
    return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ------ Additional Functions -----------
def det_clean(read_L, idxs):
    return np.array([layer_probes[read_L].predict(acts[i, read_L].reshape(1, -1))[0] for i in idxs])


# ---- examples (identical order) ----
with open("data/mixed.json") as f:
    items = json.load(f)
examples = []
for d in items:
    stmt, is_true = d["statement"], d["label"]
    honest_ans = "Yes" if is_true else "No"
    lying_ans  = "No"  if is_true else "Yes"
    examples.append({"prompt": build_example(stmt, honest_ans), "deceptive": 0, "answer": honest_ans})
    examples.append({"prompt": build_example(stmt, lying_ans),  "deceptive": 1, "answer": lying_ans})
y_decep = np.array([e["deceptive"] for e in examples])

#----------------------- Grouped Split function --------------------
groups = np.array([i // 2 for i in range(len(examples))])

# ---- load cached activations ----
acts = np.load(f"{RESULTS}/acts_deception_balanced.npy")
assert acts.shape[0] == len(examples), "acts.npy doesn't match examples — re-extract in the main script"
print("Loaded cached activations", acts.shape)

# ---- best_layer + split-trained probe + direction (identical derivation) ----
best_layer, tr_idx, va_idx, te_idx = nested_layer(acts, y_decep, groups)
trva = np.concatenate([tr_idx, va_idx])
probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, best_layer, :], y_decep[trva])
deceptive_eval = [i for i in te_idx if examples[i]["deceptive"] == 1]
best_dir_t = torch.tensor(probe.coef_[0] / norm(probe.coef_[0]), dtype=DTYPE, device=device)

# ---- all-data per-layer probes (identical) ----
layer_probes = {L: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, L, :], y_decep[tr_idx])
                for L in range(model.cfg.n_layers)}

# ---- min_alpha from cached collapse curve ----
collapse = {int(k): v for k, v in json.load(open(f"{RESULTS}/collapse_main.json")).items()}
min_alpha = next((a for a in sorted(collapse) if collapse[a] < 0.10), max(collapse))

A2 = float(min_alpha)
DECEP = deceptive_eval
E = best_layer // 2            # genuinely-early injection point
print(f"best_layer={best_layer}  min_alpha={min_alpha}  E={E}")
# -- Saves values from above analyses -
report(TAG, f"{SEC}/setup", "best_layer", int(best_layer))
report(TAG, f"{SEC}/setup", "min_alpha",  float(min_alpha))
report(TAG, f"{SEC}/setup", "E",          int(E))
report(TAG, f"{SEC}/setup", "n_test",     int(len(te_idx)), n=int(len(DECEP)))


#----------------- PHASE 2 ANALYSIS -----------------
# 2A & 2B - Persistent vs single-layer steering

def make_sub_hook(alpha, d_t):
    def h(value, hook):
        value[:, :, :] = value[:, :, :] - alpha * d_t
        return value
    return h

def detbylayer_multi(steer_layers, alpha, d_t):
    hooks = [(f"blocks.{L}.hook_resid_post", make_sub_hook(alpha, d_t)) for L in steer_layers]
    steered = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        steered.append([c["resid_post", L][0,-1,:].cpu().numpy() for L in range(model.cfg.n_layers)])
    steered = np.array(steered)
    return [layer_probes[L].predict(steered[:, L, :]).mean() for L in range(model.cfg.n_layers)]

if LEGACY:
    det_single_early = detbylayer_multi([E], A2, best_dir_t)
    det_single_late = detbylayer_multi([best_layer], A2, best_dir_t)
    det_persistent = detbylayer_multi(list(range(0, best_layer + 1)), A2, best_dir_t)

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(model.cfg.n_layers)
    ax.plot(xs, det_single_early, "-o", label=f"single early (L{E})")
    ax.plot(xs, det_single_late,  "-s", label=f"single late (L{best_layer})")
    ax.plot(xs, det_persistent,   "-^", label=f"persistent (0..{best_layer})")
    ax.axhline(0.5, ls="--", color="gray"); ax.set_xlabel("Measured layer"); ax.set_ylabel("detection")
    ax.set_title("Fig 8. Does broad steering defeat recomputation?")
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(f"{FIGS}/fig8_persistent.png", dpi=150); plt.close()
    print(f"Saved {FIGS}/fig8_persistent.png")

def gen_under_multi(statement, steer_layers, alpha, d_t, n_new=20):
    hooks = [(f"blocks.{L}.hook_resid_post", make_sub_hook(alpha, d_t)) for L in steer_layers]
    tok = model.to_tokens(answer_prompt(statement))
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            out = model.generate(tok, max_new_tokens=n_new, do_sample=False, verbose=False)
    return model.to_string(out[0][tok.shape[1]:])

if LEGACY:
    print("Persistent-steer coherence check:")
    for s in [items[0]["statement"], items[20]["statement"]]:
        print("  ", repr(gen_under_multi(s, list(range(0, best_layer + 1)), A2, best_dir_t)))


# ===== PHASE 2c: component attribution (run AFTER 2b) =====

def make_zero_hook():
    def h(value, hook):
        return value * 0
    return h

def recovery_with_ablation(comp, R_layers):
    hooks  = [(f"blocks.{E}.hook_resid_post", make_sub_hook(A2, best_dir_t))]
    hooks += [(f"blocks.{L}.{comp}", make_zero_hook()) for L in R_layers]
    steered = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        steered.append(c["resid_post", best_layer][0, -1, :].cpu().numpy())
    return layer_probes[best_layer].predict(np.array(steered)).mean()

R = list(range(E + 1, best_layer + 1)) # Recovery band

if LEGACY:
    print("recovery, no ablation :", det_single_early[best_layer]) # From 2B
    print("recovery, MLP ablated :", recovery_with_ablation("hook_mlp_out", R))
    print("recovery, ATTN ablated:", recovery_with_ablation("hook_attn_out", R))


def gen_under_ablation(statement, comp, R_layers, n_new=20):
    hooks  = [(f"blocks.{E}.hook_resid_post", make_sub_hook(A2, best_dir_t))]
    hooks += [(f"blocks.{L}.{comp}", make_zero_hook()) for L in R_layers]
    tok = model.to_tokens(answer_prompt(statement))
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            out = model.generate(tok, max_new_tokens=n_new, do_sample=False, verbose=False)
    return model.to_string(out[0][tok.shape[1]:])

if LEGACY:
    for s in [items[0]["statement"], items[20]["statement"]]:
        print("MLP-ablated:", repr(gen_under_ablation(s, "hook_mlp_out", R)))

def boot_ci(vals, n=1000):
    vals = np.array(vals); rng = np.random.default_rng(0)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def acc_ci(pr, X, y, n=1000):
    correct = (pr.predict(X) == y).astype(float)
    rng = np.random.default_rng(0)
    b = [correct[grouped_ix(rng)].mean() for _ in range(n)]
    return correct.mean(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def recovery_preds(comp, R_layers):
    hooks = [(f"blocks.{E}.hook_resid_post", make_sub_hook(A2, best_dir_t))]
    if comp: hooks += [(f"blocks.{L}.{comp}", make_zero_hook()) for L in R_layers]
    preds = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        preds.append(layer_probes[best_layer].predict(c["resid_post", best_layer][0, -1, :].cpu().numpy().reshape(1,-1))[0])
    return np.array(preds)

def score_after_ablate(ablate_L, read_L, idxs):
    hooks = [(f"blocks.{ablate_L}.hook_mlp_out", make_zero_hook())]
    s = []
    for i in idxs:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        s.append(layer_probes[read_L].decision_function(c["resid_post", read_L][0, -1, :].cpu().numpy().reshape(1, -1))[0])
    return float(np.mean(s))

def ablate_pass(ablate_L, idxs, read_Ls):
    # One ablated forward per item - all layers and activations
    hooks = [(f"blocks.{ablate_L}.hook_mlp_out", make_zero_hook())]
    S = np.zeros((len(idxs), model.cfg.n_layers))
    Xs = {L: [] for L in read_Ls}
    for r, i in enumerate(idxs):
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        for L in range(model.cfg.n_layers):
            v = c["resid_post", L][0, -1, :].cpu().numpy()
            S[r,L] = layer_probes[L].decision_function(v.reshape(1, -1))[0]
            if L in read_Ls: Xs[L].append(v)
    return S, {L: np.array(v) for L, v in Xs.items()}

def score_clean(read_L, idxs):
    return float(np.mean([layer_probes[read_L].decision_function(acts[i, read_L].reshape(1, -1))[0] for i in idxs]))

# find the recomputing MLP dynamically (works for any model size)

sel_dec = [i for i in tr_idx if examples[i]["deceptive"] == 1][:120]
det_base = {L: det_clean(L, sel_dec).mean() for L in R}
cand = [L for L in R if det_base[L] > 0.70]
if not cand: cand = [max(det_base, key=det_base.get)]
local_drop = {L: score_clean(L, sel_dec) - score_after_ablate(L, L, sel_dec) for L in cand}
crit_L = max(local_drop, key=local_drop.get)
report(TAG, f"{SEC}/setup", "crit_L", int(crit_L)) # Savings Critical Layer

y_truth = np.array([items[i // 2]["label"] for i in range(len(examples))])
y_pol = np.array([1 if examples[i]["answer"] == "Yes" else 0 for i in range(len(examples))])
labs = {"deception": y_decep, "truth": y_truth, "polarity": y_pol}

S_by_ablate, X_by_ablate = {}, {}
for L in R:
    S_by_ablate[L], X_by_ablate[L] = ablate_pass(L, te_idx, {L, best_layer})
S_abl = S_by_ablate[crit_L]
S_clean = np.array([[layer_probes[L].decision_function(acts[i, L].reshape(1, -1))[0] for L in range(model.cfg.n_layers)] for i in te_idx])

ydv, ytv = y_decep[te_idx], y_truth[te_idx]
ypv = y_pol[te_idx]

te_groups = groups[te_idx]
uniq_g = np.unique(te_groups)
g_rows = {g: np.where(te_groups == g)[0] for g in uniq_g}

def grouped_ix(rng):
    # Resample Statements and take twins of each
    gs = rng.choice(uniq_g, len(uniq_g), replace=True)
    return np.concatenate([g_rows[g] for g in gs])

print(f"\nNecessity vs readout distance (ablate L{crit_L}) - ACC and AUC:")
for L in range(crit_L, best_layer + 1):
    ca = ((S_clean[:, L]>0).astype(int) == ydv).mean()
    aa = ((S_abl[:, L] >0).astype(int) == ydv).mean()
    cauc = roc_auc_score(ydv, S_clean[:, L])
    aauc = roc_auc_score(ydv, S_abl[:, L])
    print(f" read L{L:2d} (crit + {L - crit_L}): acc {ca:.2f} -> {aa:.2f} | AUC {cauc:.2f} -> {aauc:.2f}")
    # Record values
    report(TAG, f"{SEC}/distance_curve", f"L{L}",
           {"acc_clean": ca, "acc_abl": aa, "auc_clean": cauc, "auc_abl": aauc})

np.savez(f"{RESULTS}/distance_curve.npz", s_clean = S_clean, s_abl = S_abl, y=ydv, crit = crit_L, best = best_layer)


def auc_ci_from_scores(y, s, n=1000):
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = grouped_ix(rng)
        if len(np.unique(y[ix])) < 2: continue
        b.append(roc_auc_score(y[ix], s[ix]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

print(f"\nPer-Layer MLP ablation curve - AUC (clean -> ablated), local and deployed readout:")
curve = {}

for L in R:
    S_L = S_by_ablate[L]
    loc_c, loc_a = roc_auc_score(ydv, S_clean[:, L]), roc_auc_score(ydv, S_L[:, L])
    dep_c, dep_a = roc_auc_score(ydv, S_clean[:, best_layer]), roc_auc_score(ydv, S_L[:, best_layer])
    llo, lhi = auc_ci_from_scores(ydv, S_L[:, L])
    dlo, dhi = auc_ci_from_scores(ydv, S_L[:, best_layer])
    curve[L] = (loc_c, loc_a, dep_c, dep_a)
    star = " <- crit (selected)" if L == crit_L else (" [= best_layer]" if L == best_layer else "")
    print(f" ablate L{L:2d}: local {loc_c:.2f} -> {loc_a:.2f} [{llo:.2f}, {lhi:.2f}] |"
          f" deployed {dep_c:.2f} -> {dep_a:.2f} [{dlo:.2f}, {dhi:.2f}] {star}")
    report(TAG, f"{SEC}/ablation_curve", f"L{L}",
           {"local_clean": loc_c, "local_abl": loc_a, "deployed_clean": dep_c, "deployed_abl": dep_a},
           ci={"local": [llo, lhi], "deployed": [dlo, dhi]})

np.savez(f"{RESULTS}/ablation_curve.npz",
         layers=np.array(R), curve=np.array([curve[L] for L in R]),
         crit=crit_L, best = best_layer, y=ydv)

# Necessity on hold out test at both the local readout
def spec_at(L_read, Xa, n=1000):
    P = {nm: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, L_read, :], lab[tr_idx]) for nm, lab in [("deception", y_decep), ("truth", y_truth), ("polarity", y_pol)]}
    Xc = acts[te_idx, L_read, :]
    s = {nm: (P[nm].decision_function(Xc), P[nm].decision_function(Xa)) for nm in P}
    Y = {"deception": ydv, "truth": ytv, "polarity": ypv}
    auc = {nm: (roc_auc_score(Y[nm], s[nm][0]), roc_auc_score(Y[nm], s[nm][1])) for nm in P}
    point = (auc["truth"][0] - auc["truth"][1]) - (auc["deception"][0] - auc["deception"][1])
    rng = np.random.default_rng(0); ds=[]
    for _ in range(n):
        ix = grouped_ix(rng)
        if len(np.unique(ydv[ix])) < 2 or len(np.unique(ytv[ix])) < 2: continue
        ds.append((roc_auc_score(ytv[ix], s["truth"][0][ix]) - roc_auc_score(ytv[ix], s["truth"][1][ix])) - (roc_auc_score(ydv[ix], s["deception"][0][ix]) - roc_auc_score(ydv[ix], s["deception"][1][ix])))
    return auc, point, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

#Specificity Loop
print(f"\nSpecificity curve - every layer in R, local readout (AUC clean -> ablated):")
spec = {}
for L in R:
    auc, pt, lo, hi = spec_at(L, X_by_ablate[L][L])
    spec[L] = {"dec": list(auc["deception"]), "tru": list(auc["truth"]), "pol": list(auc["polarity"]), "asym": pt, "ci": [lo, hi]}
    report(TAG, f"{SEC}/specificity_curve", f"L{L}", spec[L], ci=[lo,hi])
    tag = " <- crit" if L == crit_L else (" [=best]" if L == best_layer else "")
    print(f" ablate L{L:2d}: dec {auc['deception'][0]:.2f} -> {auc['deception'][1]:.2f} | "
          f" tru {auc['truth'][0]:.2f} -> {auc['truth'][1]:.2f} | "
          f" pol {auc['polarity'][0]:.2f} -> {auc['polarity'][1]:.2f} | "
          f" asym {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] {tag}")

np.savez(f"{RESULTS}/specificity_curve.npz",
         layers=np.array(R),
         dec=np.array([spec[L]["dec"] for L in R]),
         tru=np.array([spec[L]["tru"] for L in R]),
         pol=np.array([spec[L]["pol"] for L in R]),
         asym=np.array([spec[L]["asym"] for L in R]),
         asym_ci=np.array([spec[L]["ci"] for L in R]),
         crit=crit_L, best=best_layer)


print(f"crit MLP L{crit_L} (no_steer necessity, chosen on train-fold items)")
if LEGACY:
    for name, comp in [("no ablation", None), (f"MLP L{crit_L}", "hook_mlp_out")]:
        m, lo, hi = boot_ci(recovery_preds(comp, [crit_L]))
        print(f"recovery {name}: {m:.2f} [{lo:.2f}, {hi:.2f}]")

    print(f"coherence under L{crit_L} ablation:",
        repr(gen_under_ablation(items[0]["statement"], "hook_mlp_out", [crit_L])))

def recovery_patch_clean_mlp(L):
    clean = {}
    def save(v, hook): clean["v"] = v.clone(); return v
    def steer(v, hook): v[:, :, :] = v[:, :, :] - A2 * best_dir_t; return v
    def paste(v, hook): return clean["v"]
    s = []
    for i in DECEP:
        tok = model.to_tokens(examples[i]["prompt"])
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", save)]):
                model.run_with_cache(tok, names_filter=RESID_ONLY)
            with model.hooks(fwd_hooks=[(f"blocks.{E}.hook_resid_post", steer), (f"blocks.{L}.hook_mlp_out", paste)]):
                _, c = model.run_with_cache(tok, names_filter=RESID_ONLY)
        s.append(layer_probes[best_layer].decision_function(c["resid_post", best_layer][0,-1,:].cpu().numpy().reshape(1,-1))[0])
    return np.array(s)

patched_scores = recovery_patch_clean_mlp(crit_L)
m, lo, hi = boot_ci((patched_scores > 0).astype(int))
print(f"recovery, clean {crit_L} MLP patched into steered run: {m:.2f} [{lo:.2f}, {hi:.2f}]")


h_steer = (f"blocks.{E}.hook_resid_post", make_sub_hook(A2, best_dir_t))
h_zero = (f"blocks.{crit_L}.hook_mlp_out", make_zero_hook())

def scores_cond(hooks):
    s = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        s.append(layer_probes[best_layer].decision_function(c["resid_post", best_layer][0, -1, :].cpu().numpy().reshape(1, -1))[0])
    return np.array(s)

s_clean_i = np.array([layer_probes[best_layer].decision_function(acts[i, best_layer].reshape(1, -1))[0] for i in DECEP])
s_steer_i = scores_cond([h_steer])
s_steerabl_i = scores_cond([h_steer, h_zero])
eps = 0.25 * np.std(s_clean_i)

d_abl = np.abs(s_steerabl_i - s_steer_i)
print(f"ablation effect on steered run: mean |delta| {d_abl.mean():.3f} | "
      f"frac items |delta| > {eps:.2f}: {(d_abl > eps).mean():.2f}")

s_patch_i = patched_scores

# Save the steered ablation diagnosis
report(TAG, f"{SEC}/sufficiency", "ablation_effect_steered",
       {"mean_abs_delta": float(d_abl.mean()), "frac_above_eps": float((d_abl > eps).mean()), "eps": float(eps)})
report(TAG, f"{SEC}/sufficiency", "cond_means",
       {"clean": float(s_clean_i.mean()), "steered": float(s_steer_i.mean()),
        "steered_ablated": float(s_steerabl_i.mean()), "patched": float(s_patch_i.mean())})

def med_frac(num, den, eps, n=1000):
    m = np.abs(den) > eps
    r = num[m] / den[m]
    rng = np.random.default_rng(0)
    b = [np.median(r[rng.integers(0, len(r), len(r))]) for _ in range(n)]
    return float(np.median(r)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(m.sum())

A, Alo, Ahi, nA = med_frac(s_patch_i - s_steerabl_i, s_steer_i - s_steerabl_i, eps)
B, Blo, Bhi, nB = med_frac(s_patch_i - s_steer_i, s_clean_i - s_steer_i, eps)

print(f"Restoration A (conduit): {A:+.2f} [{Alo:+.2f}, {Ahi:+.2f}] n={nA}/{len(DECEP)}")
print(f"Restoration B (undo steering): {B:+.2f} [{Blo:+.2f}, {Bhi:+.2f}] n={nB}/{len(DECEP)}")

# Saving the restoration
report(TAG, f"{SEC}/sufficiency", "restoration_A_conduit", A, ci = [Alo, Ahi], n=nA)
report(TAG, f"{SEC}/sufficiency", "restoration_B_undo_steering", B, ci = [Blo, Bhi], n=nB)

Ploc = {nm: LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, crit_L, :], lab[tr_idx]) for nm, lab in [("deception", y_decep), ("truth", y_truth), ("polarity", y_pol)]}
Xabl_loc = X_by_ablate[crit_L][crit_L]
Xabl_dep = X_by_ablate[crit_L][best_layer]

dc_m, dc_lo, dc_hi = acc_ci(layer_probes[best_layer], acts[te_idx, best_layer, :], labs["deception"][te_idx])
da_m, da_lo, da_hi = acc_ci(layer_probes[best_layer], Xabl_dep, labs["deception"][te_idx])
print(f"\nDEPLOYED necessity as ACC (read L{best_layer})"
      f" {dc_m:.2f} [{dc_lo:.2f}, {dc_hi:.2f}] -> {da_m:.2f} [{da_lo:.2f}, {da_hi:.2f}]")

# Saving deployed necessity:
report(TAG, f"{SEC}/necessity", "deployed_acc",
       {"clean": dc_m, "abl": da_m}, ci={"clean": [dc_lo, dc_hi], "abl": [da_lo, da_hi]})

print(f"\nSpecificity at LOCAL readout L{crit_L} (clean -> ablated), held-out:")
for nm in ["deception", "truth", "polarity"]:
    cm, clo, chi = acc_ci(Ploc[nm], acts[te_idx, crit_L, :], labs[nm][te_idx])
    am, alo, ahi = acc_ci(Ploc[nm], Xabl_loc, labs[nm][te_idx])
    print(f" {nm:9}: {cm:.2f} [{clo:.2f}, {chi:.2f}] -> {am:.2f} [{alo:.2f}, {ahi:.2f}]")
    # Saving the crit-specificity accuracy
    report(TAG, f"{SEC}/specificity_crit_acc", nm, {"clean": cm, "abl": am}, ci={"clean": [clo, chi], "abl": [alo, ahi]})

Xc_loc = acts[te_idx, crit_L, :]
d_dec = ((Ploc["deception"].predict(Xc_loc)  == labs["deception"][te_idx]).astype(float) - (Ploc["deception"].predict(Xabl_loc) == labs["deception"][te_idx]).astype(float))
d_tru = ((Ploc["truth"].predict(Xc_loc)  == labs["truth"][te_idx]).astype(float) - (Ploc["truth"].predict(Xabl_loc) == labs["truth"][te_idx]).astype(float))

diff = d_tru - d_dec
rng = np.random.default_rng(0)
b = [diff[grouped_ix(rng)].mean() for _ in range(1000)]

print(f" paired (truth drop - deception drop) at L{crit_L}: {diff.mean():+.3f} "
      f"[{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] (asymmetry is real only if CI excludes 0)")
#Savings the asymmetry in paired accuracy
report(TAG, f"{SEC}/specificity_crit_acc", "paired_asym", float(diff.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])

def auc_pair(pr, Xc, Xa, y, n=1000):
    sc, sa = pr.decision_function(Xc), pr.decision_function(Xa)
    rng = np.random.default_rng(0); b = []
    for _ in range(n):
        ix = grouped_ix(rng)
        if len(np.unique(y[ix])) < 2: continue
        b.append(roc_auc_score(y[ix], sa[ix]))
    return roc_auc_score(y, sc), roc_auc_score(y, sa), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

print(f"\nSpecificity at LOCAL readout L{crit_L} - AUC (clean -> ablated):")
for nm in ["deception", "truth", "polarity"]:
    ac, aa, lo, hi = auc_pair(Ploc[nm], acts[te_idx, crit_L, :], Xabl_loc, labs[nm][te_idx])
    print(f" {nm:9}: AUC {ac:.2f} -> {aa:.2f} [{lo:.2f}, {hi:.2f}]")
    # saving the crit-specificity data
    report(TAG, f"{SEC}/specificity_crit_auc", nm, {"clean": ac, "abl": aa}, ci=[lo, hi])
