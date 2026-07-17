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
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float16

device = DEVICE

SIZE = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
TAG = SIZE
model_name = MODELS[SIZE]
model = HookedTransformer.from_pretrained(model_name, device=DEVICE, dtype=DTYPE)

RESULTS, FIGS = f"results/{TAG}", f"figures/{TAG}"

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

RESID_ONLY = lambda name: name.endswith("hook_resid_post")

print("Model Loaded")

# Nested layer section
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

# ---- prompts (identical) ----
def build_example(statement, forced_answer):
    messages = [{"role": "user", "content": f"Statement: {statement}\nIs this statement true? Answer Yes or No."}]
    chat = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat + forced_answer

def answer_prompt(statement):
    messages = [{"role": "user", "content": f'Statement: {statement}\nIs this statement true? Reply with exactly one word: "Yes" or "No".'}]
    return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
deceptive_idx = [i for i, e in enumerate(examples) if e["deceptive"] == 1]

#----------------------- Grouped Split function --------------------
groups = np.array([i // 2 for i in range(len(examples))])
def gsplit(X, y, seed=0, test_size=0.25):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    return X[tr], X[te], y[tr], y[te], tr, te

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



# ========================================================
#                       PHASE 2 ANALYSIS
# ========================================================
# 2A & 2B - Persistent vs single-layer steering

def make_sub_hook(alpha, d_t):
    def h(value, hook):
        value[:, :, :] = value[:, :, :] - alpha * d_t
        return value
    return h

DECEP = deceptive_eval
A2 = float(min_alpha)

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

for s in [items[0]["statement"], items[20]["statement"]]:
    print("MLP-ablated:", repr(gen_under_ablation(s, "hook_mlp_out", R)))


# print(f"\nPer-layer MLP ablation (recover at {best_layer}):")
# for L in R:
#     print(f" ablate MLP L{L:2d}: {recovery_with_ablation('hook_mlp_out', [L]):.2f}")

def boot_ci(vals, n=1000):
    vals = np.array(vals); rng = np.random.default_rng(0)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

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

# find the recomputing MLP dynamically (works for any model size)

per_layer = {L: recovery_with_ablation("hook_mlp_out", [L]) for L in R}
crit_L = min(per_layer, key=per_layer.get)
print("per-layer MLP ablation:", {L: round(v, 2) for L, v in per_layer.items()})
print(f"Critical recomputing MLP: L{crit_L} (recovery {per_layer[crit_L]:.2f})")




def detection_ablate_only(comp, R_layers):
    hooks = [(f"blocks.{L}.{comp}", make_zero_hook()) for L in R_layers]
    preds = []
    for i in DECEP:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        preds.append(layer_probes[best_layer].predict(c["resid_post", best_layer][0,-1,:].cpu().numpy().reshape(1,-1))[0])
    return boot_ci(np.array(preds))

base = boot_ci(np.array([layer_probes[best_layer].predict(acts[i, best_layer].reshape(1,-1))[0] for i in DECEP]))
print("baseline detection (no steer, no ablate):", base)
print(f"detection, ablate L{crit_L} MLP, NO steer :", detection_ablate_only('hook_mlp_out', [crit_L]))

# -----no-steer ablation curve across the band + matched non-critical control
noste = {L: detection_ablate_only('hook_mlp_out', [L])[0] for L in R}
print("no-steer ablation, detection by ablated layer:", {L: round(v, 2) for L, v in noste.items()})
ctrl_L = max(noste, key=noste.get)
print(f"NON-critical control L{ctrl_L}: {noste[ctrl_L]:.2f} | critical L{crit_L}: {noste[crit_L]:.2f}")


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
    preds = []
    for i in DECEP:
        tok = model.to_tokens(examples[i]["prompt"])
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", save)]):
                model.run_with_cache(tok, names_filter=RESID_ONLY)
            with model.hooks(fwd_hooks=[(f"blocks.{E}.hook_resid_post", steer), (f"blocks.{L}.hook_mlp_out", paste)]):
                _, c = model.run_with_cache(tok, names_filter=RESID_ONLY)
        preds.append(layer_probes[best_layer].predict(c["resid_post", best_layer][0,-1,:].cpu().numpy().reshape(1,-1))[0])
    return np.array(preds)

m, lo, hi = boot_ci(recovery_patch_clean_mlp(crit_L))
print(f"recovery, clean {crit_L} MLP patched into steered run: {m:.2f} [{lo:.2f}, {hi:.2f}]")

y_truth = np.array([items[i // 2]["label"] for i in range(len(examples))])
y_pol = np.array([1 if examples[i]["answer"] == "Yes" else 0 for i in range(len(examples))])

def train_probe(lab):
    return LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, best_layer, :], lab[tr_idx])

P = {"deception": train_probe(y_decep), "truth": train_probe(y_truth), "polarity": train_probe(y_pol)}
labs = {"deception": y_decep, "truth": y_truth, "polarity": y_pol}

def acts_ablated(idxs, L):
    hooks = [(f"blocks.{L}.hook_mlp_out", make_zero_hook())]
    out = []
    for i in idxs:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                _, c = model.run_with_cache(model.to_tokens(examples[i]["prompt"]), names_filter=RESID_ONLY)
        out.append(c["resid_post", best_layer][0, -1, :].cpu().numpy())
    return np.array(out)

Xabl = acts_ablated(te_idx, crit_L)
print(f"\nSpecificity of MLP L{crit_L} (clean -> ablated), held-out:")
for nm in ["deception", "truth", "polarity"]:
    clean = P[nm].score(acts[te_idx, best_layer, :], labs[nm][te_idx])
    abl = P[nm].score(Xabl, labs[nm][te_idx])
    print(f" {nm:9}: {clean:.2f} -> {abl:.2f}")




# ===== PHASE 3: probe-architecture robustness =====

def pooled_acts(prompt, fwd_hooks=None):
    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks or []):
            _, c = model.run_with_cache(model.to_tokens(prompt), names_filter=RESID_ONLY)
    L = best_layer
    r = c["resid_post", L][0]
    return (r[-1].cpu().numpy(), r.mean(0).cpu().numpy(), r.max(0).values.cpu().numpy())

# 1) clean pooled features (cache these — new extraction, ~5-10 min)
POOL = f"{RESULTS}/pooled_bestlayer.npz"
if os.path.exists(POOL):
    z = np.load(POOL); Xlast, Xmean, Xmax = z["last"], z["mean"], z["max"]
else:
    feats = [pooled_acts(e["prompt"]) for e in examples]
    Xlast = np.array([f[0] for f in feats]); Xmean = np.array([f[1] for f in feats]); Xmax = np.array([f[2] for f in feats])
    np.savez(POOL, last=Xlast, mean=Xmean, max=Xmax)

# 2) train each architecture, report clean accuracy
def clean_acc(X):
    Xtr, Xte, ytr, yte, _, _ = gsplit(X, y_decep)
    return LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr).score(Xte, yte)
probes = {
    "linear-last": LogisticRegression(max_iter=2000, C=0.1).fit(Xlast, y_decep),
    "linear-mean": LogisticRegression(max_iter=2000, C=0.1).fit(Xmean, y_decep),
    "linear-max":  LogisticRegression(max_iter=2000, C=0.1).fit(Xmax,  y_decep),
    "mlp-last":    MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(Xlast, y_decep),
}

for name, X in [("linear-last", Xlast),("linear-mean",Xmean),("linear-max", Xmax),("mlp-last",Xlast)]:
    print(f"{name:12s} clean acc: {clean_acc(X):.3f}")

# 3) robustness: SAME residual attack, measure each probe's detection on steered deceptive items
best_dir_t = best_dir_t
def steer_hook(a):
    def h(v, hook):
        v[:, :, :] = v[:,:,:] - a * best_dir_t
        return v
    return [(f"blocks.{best_layer}.hook_resid_post", h)]

for a in [0, min_alpha, 2 * min_alpha]:
    feats = [pooled_acts(examples[i]["prompt"], steer_hook(a)) for i in DECEP]
    Xl = np.array([f[0] for f in feats]); Xm = np.array([f[1] for f in feats]); Xx = np.array([f[2] for f in feats])
    print(f"\nalpha={a}")
    print(f"  linear-last det: {probes['linear-last'].predict(Xl).mean():.2f}")
    print(f"  linear-mean det: {probes['linear-mean'].predict(Xm).mean():.2f}")
    print(f"  linear-max  det: {probes['linear-max'].predict(Xx).mean():.2f}")
    print(f"  mlp-last    det: {probes['mlp-last'].predict(Xl).mean():.2f}")
