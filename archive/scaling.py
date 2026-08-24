import json, os
import numpy as np, torch

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt

device = "cude" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# name -> (HF id, param count for the scaling axis)
MODELS = {
    "pythia-410m": ("EleutherAI/pythia-410m", 0.41e9),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b",  1.4e9),
    # "pythia-2.8b": ("EleutherAI/pythia-2.8b",  2.8e9),
    # "pythia-6.9b": ("EleutherAI/pythia-6.9b",  6.9e9),
    "qwen-0.5b":   ("Qwen/Qwen2.5-0.5B", 0.5e9),
    "qwen-1.5b":   ("Qwen/Qwen2.5-1.5B", 1.5e9),
    "qwen-3b":     ("Qwen/Qwen2.5-3B",   3.0e9),
    # "qwen-7b":     ("Qwen/Qwen2.5-7B",   7.0e9),
}

def build_example(stmt, ans):
    return f"Statement: {stmt}\nIs this statement true? Answer Yes or No.\nAnswer: {ans}"

items = json.load(open("data/mixed.json"))
examples, y = [], []
for d in items:
    t = d["label"]
    examples.append(build_example(d["statement"], "Yes" if t else "No")); y.append(0)
    examples.append(build_example(d["statement"], "No" if t else "Yes")); y.append(1)
y = np.array(y); groups = np.arange(len(examples)) // 2

def gsplit(X, yy, seed):
    return next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, yy, groups))

def get_acts(model, prompt):
    with torch.no_grad():
        _, c = model.run_with_cache(model.to_tokens(prompt))
    return np.array([c["resid_post", L][0, -1, :].float().cpu().numpy() for L in range(model.cfg.n_layers)])

def best_layer(acts):
    b, ba = 0, 0.0
    for L in range(acts.shape[1]):
        tr, te = gsplit(acts[:, L, :], y, 0)
        a = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], y[tr]).score(acts[te, L, :], y[te])
        if a > ba: b, ba = L, a
    return b, ba

def inlp_dim(X, yy, seed, n_dirs=60, margin=0.03):
    tr, te = gsplit(X, yy, seed)
    Xtr, Xte, ytr, yte = X[tr].astype(np.float64).copy(), X[te].astype(np.float64).copy(), yy[tr], yy[te]
    removed = []
    for _ in range(n_dirs):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
        if clf.score(Xte, yte) <=0.5 + margin: break
        w = clf.coef_[0].astype(np.float64)
        for r in removed: w -= (w @ r) * r
        nrm = np.linalg.norm(w)
        if nrm < 1e-9: break
        w /= nrm; removed.append(w)
        Xtr -= np.outer(Xtr @ w, w); Xte -= np.outer(Xte @ w, w)
    return len(removed)

os.makedirs("results/scaling", exist_ok=True)
xs, ys, es = [], [], []
print(f"{'model':12} {'d_model':7} {'L*':3} {'acc':5} {'dim (mean +/- sd)':16} {'shuf':5} {'dim/d':7}")
for name, (hf, params) in MODELS.items():
    cache = f"results/scaling/{name}.npy"
    if os.path.exists(cache):
        acts = np.load(cache)
    else:
        try:
            model = HookedTransformer.from_pretrained(hf, device=device, dtype=torch.float16)
        except Exception as e:
            print(f"{name}: load failed ({e}) - skipping"); continue
        acts = np.array([get_acts(model, e) for e in examples])
        np.save(cache, acts)
        del model
        if device == "cude": torch.cuda.empty_cache()
    L, acc = best_layer(acts); d = acts.shape[2]
    dims = [inlp_dim(acts[:, L, :], y, s) for s in range(5)]
    shuf = np.mean([inlp_dim(acts[:, L, :], np.random.default_rng(s).permutation(y), s) for s in range(5)])
    m, sd = float(np.mean(dims)), float(np.std(dims))
    print(f"{name:12} {d:7} {L:3} {acc:.2f} {m:6.1f} +/- {sd:4.1f} {shuf:5.1f} {m/d:7.4f}")
    xs.append(params); ys.append(m); es.append(sd)

# Scaling-law plot
plt.figure(figsize=(7, 5))
plt.errorbar(xs, ys, yerr=es, fmt="o", capsize=3)
plt.xscale("log"); plt.xlabel("parameters"); plt.ylabel("INLP effective dimensionality of deception signal")
plt.title("Deception-representation dimensionality vs model scale"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/scaling_dim.png", dpi=150)
print("Saved figures/scaling_dim.png")
