import json, os, re, warnings
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = os.path.expanduser("~/Desktop/truth-probes/results_gpu")
TAGS = ["0.5B", "1.5B", "3B", "7B", "14B", "gemma-2b", "gemma-9b", "llama-3b", "llama-8b"]

# Parameter counts in billions - the x axis for "the is it size?" test
PARAMS = {"0.5B": 0.494, "1.5B": 1.54, "3B": 3.09, "7B":7.62, "14B": 14.8, "gemma-2b": 2.61, "gemma-9b": 9.24, "llama-3b": 3.21, "llama-8b": 8.03}

COMMON = 512 # project every model to same width - ask about this
N_DIRS = 200 # this was 60 but GEMMA hit the ceiling
N_SEEDS = 3

def gsplit(y, g, seed):
    """Statement-grouped train/test split: both twins of a statement stay on the same side."""
    return next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(np.zeros(len(y)), y, g))
                
def best_layer(tag):
    """Read the per-layer probe accuracies and return (argmax layer, that accuracy)"""
    sw = json.load(open(f"{ROOT}/{tag}/sweep_stats.json"))
    L = int(max(sw, key=lambda k: sw[k]["mean"]))
    return L, max(v["mean"] for v in sw.values())

def load_best(tag):
    """Activations at the best layer only -> (800, d_model), plus labels and groups"""
    L, _ = best_layer(tag)
    # mmap_mode = 'r' streams from disk - ask
    A = np.load(f"{ROOT}/{tag}/acts_deception_balanced.npy", mmap_mode="r")[:, L, :]
    lab = np.load(f"{ROOT}/{tag}/labels.npz")
    return np.asarray(A, dtype=np.float32), lab["y_decep"], lab["groups"]

def task_accuracy(tag):
    """Scrape the model's OWN Yes/No correctness out of the run log (our capability proxy)"""
    try:
        txt = open(f"{ROOT}/{tag}/run_probe.txt", errors="ignore").read()
        hits = re.findall(r"statements: model correct (\d+)%", txt) # Matches true and False Lines
        return np.mean([int(h) for h in hits]) / 100 if hits else np.nan
    except Exception:
        return np.nan
    
# ---------------- Analysis A: INLP, un-capped ---------------

def inlp_dim(X, y, g, seed, n_dirs=N_DIRS):
    """
    Iterative Nullspace Projection: how many directions must we delete before a
    linear probe can no longer beat chance on held-out data?
    Each loop: fit a probe, take its direction, project that direction out of the data, repeat.
    """
    
    tr, te = gsplit(y, g, seed)

    # Standardise - new aspect of experiment
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr]).astype(np.float64)
    Xte = sc.transform(X[te]).astype(np.float64)
    ytr, yte = y[tr], y[te]

    # Stop when held out accuracy is wihtin 2 SD of chance
    thresh = 0.5 + 2* np.sqrt(0.25/len(yte))

    removed = []
    for _ in range(n_dirs):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, ytr)
        if clf.score(Xte, yte) <= thresh:
            break # Signal is gone and we're done

        w = clf.coef_[0].astype(np.float64)
        for r in removed:
            w -= (w @ r) * r
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w /= n
        removed.append(w)

        # Project w out of the data: x <- x - (x*w)w
        Xtr -= np.outer(Xtr @ w, w)
        Xte -= np.outer(Xte @ w, w)

    return len(removed)

print("A. INLP directions-to-erase (n_dirs=%d, projected to %dd)\n" % (N_DIRS, COMMON))
rows = {}
for t in TAGS:
    A, y, g = load_best(t)

    # Random projection to common width so a wide model doesn't win by width
    P = np.random.default_rng(0).standard_normal((A.shape[1], COMMON)) / np.sqrt(COMMON)
    Xp = (A @ P).astype(np.float32)

    dims = [inlp_dim(Xp, y, g, s) for s in range(N_SEEDS)]
    #null: same pipeline, labels shuffled shoudl be ~0 if method isn't hallucinating structure/
    null = [inlp_dim(Xp, np.random.default_rng(s).permutation(y), g, s) for s in range(N_SEEDS)]

    L, acc = best_layer(t)
    capped = np.mean(dims) >= N_DIRS - 1 # Not a measurement in case we hit loop limit
    rows[t] = {"inlp": float(np.mean(dims)), "null": float(np.mean(null)), "probe_acc": acc, "task_acc": task_accuracy(t), "params": PARAMS[t], "capped": capped}

    flag = " <-- STILL CAPPED exclude" if capped else ""
    print(f" {t:10s} L{L:2d} INLP {np.mean(dims):6.1f} +/- {np.std(dims):4.1f} | "
          f"null {np.mean(null):4.1f} | probe {acc:.3f} | task {rows[t]['task_acc']:.2f}{flag}")
    
# ------------- B. Deos redundancy track size or capability? ---------
print("\nB. What does redundancy track\n")

ok = [t for t in TAGS if not rows[t]["capped"]] # a capped value is fake
inlp = np.array([rows[t]["inlp"] for t in ok])

for name, x in [("log10(params)",  np.log10([rows[t]["params"] for t in ok])),
                ("probe accuracy", np.array([rows[t]["probe_acc"] for t in ok])),
                ("task accuracy",  np.array([rows[t]["task_acc"] for t in ok]))]:
    m = ~np.isnan(x)
    # Spearman = rank correlation
    r, p = spearmanr(x[m], inlp[m])
    print(f" INLP vs {name:16s}: rho = {r:+.3f} (p = {p:.3f}, n = {m.sum()})")

# --------------- C. Free-form --------------

print("C. Frozen probe --> free-form lies, AUC at four readout positions\n")
print(f" {'model':10s} {'n_lies':>6s} {'prompt':>8s} {'first':>8s} {'last':>8s} {'mean':>8s}")

for t in TAGS:
    f = f"{ROOT}/{t}/phase4b.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    y4 = z["y"]
    if len(set(y4.tolist())) < 2 or min(np.bincount(y4)) < 5:
        print(f" {t:10s} too few lies ({int((y4 == 1).sum())})")
        continue

    # Rebuild the frozen teacher-forced probe - same construction as phase2.py

    A, y, g = load_best(t)
    tr, _ = gsplit(y, g, 0)
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(A[tr], y[tr])

    # p = last prompt token
    aucs = [roc_auc_score(y4, probe.decision_function(z[k].astype(np.float32))) for k in ("p", "f", "l", "m")]

    print(f" {t:10s} {int((y4 == 1).sum()):6d} " + " ".join(f"{a:8.3f}" for a in aucs))

print("""
 How to read C:
   ~0.5 everywhere            -> genuine NON-TRANSFER (your strongest claim)
   low at 'last', ok at 'mean'-> POSITION FRAGILITY (smaller claim, still real)
   < 0.3 anywhere             -> anti-predictive: the direction inverts out of format
""")

# ----- D. Instructed Yes/No transfer - the Refusal - Filtered version of C ------
print("\nD. Frozen probe -> instructed lies (refusalss excluded by constructions)\n")
print(f" {'model':10s} {'n_lies':>6s} {'refusal':>8s} {'AUC':>7s} {'1-AUC': >7s}")

for t in TAGS:
    f = f"{ROOT}/{t}/phase4.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    X4, y4 = z["X"], z["y"] # X4 is (n, n_layers, d_model)
    if len(set(y4.tolist())) < 2 or min(np.bincount(y4)) < 5:
        print(f" {t:10s} too few lies ({int((y4 == 1).sum())})")
        continue

    # frozen teahcer-forced probe
    A, y, g = load_best(t)
    tr, _ = gsplit(y, g, 0)
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(A[tr], y[tr])

    L, _ = best_layer(t)
    auc = roc_auc_score(y4, probe.decision_function(X4[:, L, :].astype(np.float32)))
    n_lies = int((y4 == 1).sum())
    refusal = 1 - n_lies / 400 # 400 statements

    # 1 - AUC tells whether its a clean sign flip rather than noise
    print(f" {t:10s} {n_lies:6d} {refusal:7.0%} {auc:7.3f} {1-auc:7.3f}")