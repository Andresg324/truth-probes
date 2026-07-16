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
TAG = sys.argv[1] if len(sys.argv) > 1 else "3B"
model = HookedTransformer.from_pretrained(MODELS[TAG], device=DEVICE, dtype=torch.float16)
RESID = lambda n: n.endswith("hook_resid_post")
items = json.load(open("data/mixed.json"))

# Three paraphrases with the same yes/no task
TEMPLATES = [
    ("orig",  lambda s: f"Statement: {s}\nIs this statement true? Answer Yes or No."),
    ("claim", lambda s: f"Consider the claim: {s}\nIs it correct? Respond Yes or No."),
    ("tf",    lambda s: f'"{s}"\nTrue or false? Reply Yes or No.'),
]

def zero_hook(v, hook): return v * 0

def collect(tmpl):
    prompts, y = [], []
    for d in items:
        for forced, lab in [("Yes" if d["label"] else "No", 0), ("No" if d["label"] else "Yes", 1)]:
            chat = model.tokenizer.apply_chat_template(
                [{"role":"user", "content":tmpl(d["statement"])}], tokenize=False, add_generation_prompt=True)
            prompts.append(chat + forced); y.append(lab)
    y = np.array(y); g = np.arange(len(prompts)) // 2
    A = []
    for p in prompts:
        with torch.no_grad():
            _, c = model.run_with_cache(model.to_tokens(p), names_filter=RESID)
        A.append(np.array([c["resid_post", L][0,-1,:].cpu().numpy() for L in range(model.cfg.n_layers)]))
    return np.array(A), y, g, prompts

def analyze(A, y, g, prompts):
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(A[:, 0, :], y, g))
    sw = [LogisticRegression(max_iter=2000, C=0.1).fit(A[tr, L, :], y[tr]).score(A[te,L,:], y[te]) for L in range(model.cfg.n_layers)]
    bl = int(np.argmax(sw))
    probe = LogisticRegression(max_iter=2000, C=0.1).fit(A[tr, bl,:], y[tr])
    decep = [i for i in te if y[i]==1]
    base = probe.predict(A[decep, bl, :]).mean()
    def det_ablate(L):
        preds = []
        for i in decep:
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(f"blocks.{L}.hook_mlp_out", zero_hook)]):
                    _, c = model.run_with_cache(model.to_tokens(prompts[i]), names_filter=RESID)
            preds.append(probe.predict(c["resid_post",bl][0,-1,:].cpu().numpy().reshape(1,-1))[0])
        return float(np.mean(preds))
    band = list(range(bl//2 + 1, bl + 1))
    curve = {L: det_ablate(L) for L in band}
    crit = min(curve, key=curve.get)
    return bl, crit, base, curve[crit]

print(f"\n{TAG}: is the critical MLP stable across prompt paraphrases?\n")
for name, tmpl in TEMPLATES:
    A, y, g, prompts = collect(tmpl)
    bl, crit, base, cval = analyze(A, y, g, prompts)
    print(f" [{name:5s}] best_layer=L{bl}/{model.cfg.n_layers} critical_MLP=L{crit} "
          f"(no-steer ablation: detection {base:.2f} -> {cval:.2f})")
print("\n-> if critical_MLP is the same (or +/-1) across all three, localization is not a template artifact")
