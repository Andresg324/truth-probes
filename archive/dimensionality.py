import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

SIZES = ["0.5B", "1.5B", "3B"]
SEEDS = range(5)

def gsplit(X, y, g, seed, ts=0.25):
    return next(GroupShuffleSplit(1, test_size=ts, random_state=seed).split(X, y, g))

def best_layer(acts, y, g):
    tr, te = gsplit(acts[:, 0, :], y, g, 0)
    best, bacc = 0, 0.0
    for L in range(acts.shape[1]):
        acc = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr, L, :], y[tr]).score(acts[te, L, :], y[te])
        if acc > bacc: best, bacc = L, acc
    return best, bacc

# INLP: Iteratively strp probe directions; count how many to drive held-out acc
def inlp_dim(X, y, g, seed, n_dirs=40, margin=0.03, verbose=False):
    tr, te = gsplit(X, y, g, seed)
    Xtr, Xte = X[tr].astype(np.float64).copy(), X[te].astype(np.float64).copy()
    ytr, yte = y[tr], y[te]
    removed, curve = [], []
    for _ in range(n_dirs):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
        acc = clf.score(Xte, yte); curve.append(round(acc, 2))
        if acc <= 0.5 + margin:
            break
        w = clf.coef_[0].astype(np.float64)
        for r in removed: w -= (w @ r) * r
        nrm = np.linalg.norm(w)
        if nrm < 1e-9: break
        w /= nrm; removed.append(w)
        Xtr -= np.outer(Xtr @ w, w); Xte -= np.outer(Xte @ w, w)
    return len(removed)

# Evasion cost. 

def dirs_to_blind_independent(X, y, g, seed, n_dirs=40, margin=0.03):
    uniq = np.random.default_rng(seed).permutation(np.unique(g))
    nA = len(uniq) // 3
    A, B, T = set(uniq[:nA]), set(uniq[nA:2*nA]), set(uniq[2*nA:])
    iA = np.isin(g, list(A)); iB = np.isin(g, list(B)); iT=np.isin(g, list(T))
    probeB = LogisticRegression(max_iter=2000, C=0.1).fit(X[iB], y[iB])
    XA, yA = X[iA].astype(np.float64).copy(), y[iA]
    XT, yT = X[iT].astype(np.float64).copy(), y[iT]
    removed = []
    for _ in range(n_dirs):
        if probeB.score(XT, yT) <= 0.5 + margin: break
        w = LogisticRegression(max_iter=2000, C=0.1).fit(XA, yA).coef_[0].astype(np.float64)
        for r in removed: w -= (w @ r) * r
        nrm = np.linalg.norm(w)
        if nrm < 1e-9: break
        w /= nrm; removed.append(w)
        XA -= np.outer(XA @ w, w); XT -= np.outer(XT @ w, w)
    return len(removed)

print(f"{'size':5} {'d_model':7} {'L*':3} {'acc':5} {'INLP_dim':9} {'shuffle':7} {'blind_B':7}")
for s in SIZES:
    acts = np.load(f"results/{s}/acts_deception_balanced.npy").astype(np.float32)
    n, _, d = acts.shape
    y, g = np.arange(n) % 2, np.arange(n) // 2
    L, acc = best_layer(acts, y, g)
    X = acts[:, L, :]
    dim = np.mean([inlp_dim(X, y, g, i) for i in SEEDS])
    shuf = np.mean([inlp_dim(X, np.random.default_rng(i).permutation(y), g, i) for i in SEEDS])
    blind = np.mean([dirs_to_blind_independent(X, y, g, i) for i in SEEDS])
    print(f"{s:5} {d:7} {L:3} {acc:.2f} {dim:9.1f} {shuf:7.1f} {blind:7.1f}")

