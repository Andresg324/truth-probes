import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

def gsplit(X, y, g, seed):
    return next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, g))

def inlp_dim(X, y, g, seed, n_dirs=60):
    tr, te = gsplit(X, y, g, seed)
    Xtr, Xte, ytr, yte = X[tr].astype(np.float64).copy(), X[te].astype(np.float64).copy(), y[tr], y[te]
    thresh = 0.5 + 2*np.sqrt(0.25/len(yte))
    rem = []
    for _ in range(n_dirs):
        clf = LogisticRegression(max_iter=5000, C=0.1).fit(Xtr, ytr)
        if clf.score(Xte, yte) <= thresh: 
            break
        w = clf.coef_[0].astype(np.float64)
        for r in rem:
            w -= (w @ r) * r
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            break
        w /= nrm; rem.append(w)
        Xtr -= np.outer(Xtr @ w, w); Xte -= np.outer(Xte @ w, w)
    return len(rem)

n_syn = 800; y_syn = np.arange(n_syn) % 2; g_syn = np.arange(n_syn) // 2

COMMON = 512
BEST = {"0.5B": 22, "1.5B": 18, "3B": 27, "gemma-2b": 15}

#(1) Capacity Control
for s, L in BEST.items():
    A = np.load(f"results/{s}/acts_deception_balanced.npy")[:, L, :]
    lab = np.load(f"results/{s}/labels.npz"); y, g = lab["y_decep"], lab["groups"]
    P  = np.random.default_rng(0).standard_normal((A.shape[1], COMMON)) / np.sqrt(COMMON)
    Xp = A @ P
    dims = [inlp_dim(Xp, y, g, i) for i in range(5)]
    null = [inlp_dim(Xp, np.random.default_rng(i).permutation(y), g, i) for i in range(5)]
    print(f" {s}: INLP {np.mean(dims):.1f} +/- {np.std(dims):.1f} | null {np.mean(null):.1f}")


def synth(d, k, gap = 1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0][:, :k]
    sub = rng.integers(0, k, size=n_syn)
    coords = rng.standard_normal((n_syn, k))
    coords[np.arange(n_syn), sub] += (y_syn * 2 -1) * gap
    return (coords @ Q.T + rng.standard_normal((n_syn, d)) * 0.3).astype(np.float32)

print("\nINLP recovery of planted rank (should track k, roughly d-independent):")
for d in [896, 2048]:
    for k in [3, 8]:
        X = synth(d, k)
        print(f" d={d}, planted k={k}: recovered {np.mean([inlp_dim(X, y_syn, g_syn, i) for i in range(3)]):.1f}")



from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
for m in ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
          "google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct",
          "meta-llama/Llama-3.2-3B-Instruct"]:
    print(m, m in OFFICIAL_MODEL_NAMES)
    