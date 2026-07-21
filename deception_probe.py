import json, os
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
import sys

from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


#--------------- Setting up the model -------------------------
MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "gemma-2b": "google/gemma-2-2b-it",
    "gemma-9b": "google/gemma-2-9b-it",
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

#--------------- Building Examples -------------------------
def build_example(statement, forced_answer):
    #Forced answer is either yes or no and we append it
    messages = [{"role": "user", "content": f"Statement: {statement}\nIs this statement true? Answer Yes or No."}]
    chat = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat + forced_answer

try:
    toks = model.to_str_tokens(build_example("The sky is blue", "Yes"))
    print("First:", toks[0], "| Last 3:", toks[-3:])
except Exception as e:
    print("token preview skipped:", e)

with open("data/mixed.json") as f:
    items = json.load(f)

print(f"Loaded {len(items)} statements:", Counter(d['label'] for d in items))

#Building the cells
examples = []
for d in items:
    stmt, is_true = d["statement"],d["label"]
    honest_ans = "Yes" if is_true else "No"
    lying_ans = "No" if is_true else "Yes"
    examples.append({"prompt": build_example(stmt, honest_ans), "deceptive": 0, "answer": honest_ans})
    examples.append({"prompt": build_example(stmt, lying_ans), "deceptive": 1, "answer": lying_ans})
print(f"Built {len(examples)} examples")

honest_ans = Counter(e["answer"] for e in examples if e["deceptive"] == 0)
deceptive_ans = Counter(e["answer"] for e in examples if e["deceptive"] == 1)
print("Honest answers:", honest_ans)
print("Deceptive answers:", deceptive_ans)

#--------------------- Extract activations -----------------------------
RESID_ONLY = lambda name: name.endswith("hook_resid_post")

def get_answer_acts(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=RESID_ONLY)
    return np.array([cache["resid_post", L][0,-1,:].cpu().numpy() 
                     for L in range(model.cfg.n_layers)])
    
CACHE = f"{RESULTS}/acts_deception_balanced.npy"
if os.path.exists(CACHE):
    acts = np.load(CACHE)
    print("Loaded cached activations")
else:
    print("Extracting (fast - no generation)...")
    acts = np.array([get_answer_acts(e["prompt"]) for e in examples])
    os.makedirs("results", exist_ok=True)
    np.save(CACHE, acts)
    print("Cached")

y_decep = np.array([e["deceptive"] for e in examples])
print("Shape:", acts.shape)
groups = np.arange(len(examples)) // 2
np.savez(f"results/{TAG}/labels.npz", y_decep=np.asarray(y_decep), groups=np.asarray(groups))



#----------------------- Grouped Split function --------------------
groups = np.array([i // 2 for i in range(len(examples))])
def gsplit(X, y, seed=0, test_size=0.25):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    return X[tr], X[te], y[tr], y[te], tr, te

#----------------------- Probe Layer Sweep and finding the best layer --------------------
best_layer, tr_idx, va_idx, te_idx = nested_layer(acts, y_decep, groups)
trva = np.concatenate([tr_idx, va_idx])
print(f"Best layer (chosen on val fold): {best_layer}")
for L in range(model.cfg.n_layers):
    va_acc = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx, L, :], y_decep[tr_idx]).score(acts[va_idx, L, :], y_decep[va_idx])
    print(f"Layer {L:2d}: val {va_acc:.3f}")

#------------------------ Leak Control ---------------------------------------------------
#Verification step
#Control: Train a probe on the answer token string only, no model
#This should score around 0.5, else the design leaks
answer_text = [e["answer"] for e in examples]
Xc = CountVectorizer().fit_transform(answer_text)
Xtr, Xte, ytr, yte, _, _ = gsplit(Xc, y_decep)
ctrl = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte,yte)
print(f"\nLEAK CONTROL (answer word only): {ctrl:.3f} <- must be near 0.5")

accs = []
for s in range(10):
    Xtr, Xte, ytr, yte, _, _ = gsplit(Xc, y_decep, seed = s)
    accs.append(LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte))
print(f"leak control over 10 seeds: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

#----------------------- Main Probe and Deception Direction -------------------------------
probe = LogisticRegression(max_iter=2000, C=0.1).fit(acts[trva, best_layer, :], y_decep[trva])
deceptive_eval = [i for i in te_idx if examples[i]["deceptive"] == 1]
print(f"Probe accuracy at {best_layer} (held-out TEST fold): {probe.score(acts[te_idx, best_layer, :], y_decep[te_idx]):.3f}")

probe_dir = probe.coef_[0]
probe_dir = probe_dir / np.linalg.norm(probe_dir) #Normalized to a unit length
probe_dir_t = torch.tensor(probe_dir, dtype=DTYPE, device=device)

def make_evasion_hook(alpha):
    def hook(value, hook):
        #Value shape [batch, seq, d_model]; push away from the deception direction
        value[:, :, :] = value[:,:,:] - alpha * probe_dir_t
        return value
    return hook

def get_acts_with_evasion(prompt, alpha, layer=best_layer):
    tokens = model.to_tokens(prompt)
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, make_evasion_hook(alpha))]):
            _, cache = model.run_with_cache(tokens, names_filter = RESID_ONLY)
        return cache["resid_post", layer][0,-1,:].cpu().numpy()
    
deceptive_idx = [i for i, e in enumerate(examples) if e["deceptive"] == 1]

#-------------------------- Collapse the Curve at the best layer ---------------------------
CACHE_Collapse = f"{RESULTS}/collapse_main.json"
if os.path.exists(CACHE_Collapse):
    with open(CACHE_Collapse) as f:
        collapse = {int(k): v for k, v in json.load(f).items()}
    print("Loaded cached collapse curve")
else:
    collapse = {}
    for alpha in [0,2,4,6,8,10]:
        steered = np.array([get_acts_with_evasion(examples[i]["prompt"], alpha) for i in deceptive_eval])
        collapse[alpha] = float(probe.predict(steered).mean())
        print(f"alpha: {alpha:2d}: probe detects {collapse[alpha]:.2%} of lies")
    os.makedirs("results", exist_ok=True)
    with open(CACHE_Collapse, "w") as f:
        json.dump(collapse, f, indent=2)
    print("Cached collapse curve")

min_alpha = next((a for a in sorted(collapse) if collapse[a] < 0.10), max(collapse))
print("Chosen alpha:", min_alpha)

resid_norm = float(np.linalg.norm(acts[:, best_layer, :], axis=1).mean())
print(f"alpha relative units (min_alpha/||resid||): {min_alpha / resid_norm:.4f}")

#--------------------- Per Layer Probes ------------------------------------------------
layer_probes = {}
for L in range(model.cfg.n_layers):
    layer_probes[L] = LogisticRegression(max_iter=2000, C=0.1).fit(acts[tr_idx,L,:], y_decep[tr_idx])

#------------------------ Steer at best layer, then measure all layers -----------------
#Find an alpha that fools layer 19 and get all layers below that
def get_all_acts_with_evasion(prompt, alpha, steer_layer=best_layer):
    tokens = model.to_tokens(prompt)
    hook_name = f"blocks.{steer_layer}.hook_resid_post"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, make_evasion_hook(alpha))]):
            _, cache = model.run_with_cache(tokens, names_filter=RESID_ONLY)
    return np.array([cache["resid_post", L][0,-1,:].cpu().numpy() for L in range(model.cfg.n_layers)])

ALPHA = min_alpha

test_prompt = examples[deceptive_idx[0]]["prompt"]
tokens = model.to_tokens(test_prompt)
hook_name = f"blocks.{best_layer}.hook_resid_post"
with torch.no_grad():
    with model.hooks(fwd_hooks=[(hook_name, make_evasion_hook(ALPHA))]):
        out = model.generate(tokens, max_new_tokens=15, do_sample=False, verbose=False)
print(f"Coherence check at alpha {ALPHA}")
print(model.to_string(out[0]))

steered_all = np.array([get_all_acts_with_evasion(examples[i]["prompt"], ALPHA) for i in deceptive_eval])
print(f"\nSteering applied at layer {best_layer}, alpha={ALPHA} - detection by layer:")
for L in range(model.cfg.n_layers):
    preds = layer_probes[L].predict(steered_all[:, L,:])
    print(f"layer {L:2d}: {preds.mean():.2%}")


#------------------------- Behavioral effect of steering ---------------------------
def answer_prompt(statement):
    messages = [{"role": "user", "content": f'Statement: {statement}\nIs this statement true? Reply with exactly one word: "Yes" or "No".'}]
    return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)    


def parse_answer(raw):
    t = raw.lower().replace("<|endoftext|>", "").strip()
    yes_pos, no_pos = t.find("yes"), t.find("no")
    if yes_pos == -1 and no_pos == -1:
        return "other"
    if no_pos == -1: return "yes"
    if yes_pos == -1: return "no"
    return "yes" if yes_pos < no_pos else "no"

def get_answer_under_steering(statement, alpha, layer = None):
    layer = layer if layer is not None else best_layer
    tokens = model.to_tokens(answer_prompt(statement))
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, make_evasion_hook(alpha))]):
            out = model.generate(tokens, max_new_tokens=3, do_sample=False, verbose = False)
    return parse_answer(model.to_string(out[0][tokens.shape[1]:]))

CACHE_behavior = f"{RESULTS}/alpha_behavior.json"
if os.path.exists(CACHE_behavior):
    with open(CACHE_behavior) as f:
        behavior = json.load(f)
    print("\nLoaded cached behavior sweep")
else:
    print("Running behavioral sweep (slow - generates text)...")
    #Run across sample at increasing alpha
    sample = items[:60]
    behavior = {}
    for alpha in [0,2,4,6,10]:
        correct = wrong = other = 0
        for item in sample:
            ans = get_answer_under_steering(item["statement"], alpha)
            truth = "yes" if item["label"] == 1 else "no"
            if ans == "other": other += 1
            elif ans == truth: correct += 1
            else:              wrong += 1
        n = len(sample); answered = n - other
        behavior[str(alpha)] = {
            "correct": correct / answered if answered else 0,
            "wrong": wrong / answered if answered else 0,
            "answer_rate": answered / n,
        }
    os.makedirs("results", exist_ok=True)
    with open(CACHE_behavior, "w") as f:
        json.dump(behavior, f, indent=2)
    print("Cached behavioral sweep")
    
print("\nalpha | correct | wrong | answer_rate")
for a, r in behavior.items():
    print(f"{a:>5} | {r['correct']:.0%} | {r['wrong']:.0%} | {r['answer_rate']:.0%}")


# ------- Statistical Rigor ---------------------------
# ------- Multi-seed probe accuracy -------------------

def probe_accuracy_multiseed(X, y, n_seeds=10, C=0.1):
    accs = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte, _, _ = gsplit(X, y, seed=seed)
        acc = LogisticRegression(max_iter=2000, C=C).fit(Xtr, ytr).score(Xte,yte)
        accs.append(acc)
    accs = np.array(accs)
    return accs.mean(), accs.std()

mean_acc, std_acc = probe_accuracy_multiseed(acts[:, best_layer, :], y_decep)
print(f"Probe at layer {best_layer}: {mean_acc:.3f} +/- {std_acc:.3f} (10 seeds)")


# --------- Multi-seed layer sweep ----------------------
# Error bars for a paper

print("\nLayer sweep with error bars (10 seeds each)")
sweep_stats = {}
for L in range(model.cfg.n_layers):
    m, s = probe_accuracy_multiseed(acts[:, L, :], y_decep)
    sweep_stats[L] = {"mean": float(m), "std": float(s)}
    print(f" layer {L:2d}: {m:.3f} +/- {s:.3f}")

os.makedirs("results", exist_ok=True)
with open(f"{RESULTS}/sweep_stats.json", "w") as f:
    json.dump(sweep_stats, f, indent=2)
print("Saved -> results/sweep_stats.json (use for Figure 1 error bars)")

def bootstrap_ci(X, y, n_boot=1000, C=0.1):
    Xtr, Xte, ytr, yte, _, _ = gsplit(X, y)
    probe = LogisticRegression(max_iter=2000, C=C).fit(Xtr,ytr)
    preds = probe.predict(Xte)
    correct = (preds == yte).astype(int)
    boot_accs = []
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, len(correct), len(correct))
        boot_accs.append(correct[idx].mean())
    lo, hi = np.percentile(boot_accs, [2.5, 97.5])
    return np.mean(boot_accs), lo, hi

m, lo, hi = bootstrap_ci(acts[:, best_layer, :], y_decep)
print(f"\nBootstrap 95% CI at layer {best_layer}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

# ---------- Analysis B ---------

# Setup code
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

ALPHA = next((a for a in sorted({0,2,4,6,8,10})), 2) #Place holder, min_alpha below runs instead
ALPHA = min_alpha

late_steered = np.array([get_all_acts_with_evasion(examples[i]["prompt"], ALPHA) for i in deceptive_eval])

detect_late = [float(layer_probes[L].predict(late_steered[:,L,:]).mean()) for L in range(model.cfg.n_layers)]

with open(f"{RESULTS}/detection_by_layer.json", "w") as f:
    json.dump({"late": detect_late, "best layer": int(best_layer),
               "alpha": float(ALPHA)}, f, indent=2)
print("Savaed detection by layer arrays")

PAIR_CACHE = f"{RESULTS}/behavior_paired.json"
ALPHAS = [0,2,4,6,10]

if os.path.exists(PAIR_CACHE):
    with open(PAIR_CACHE) as f:
        per_item = json.load(f)
    print("Loaded cached paired data")
else:
    print("Running paired behavioral sweep (slow)...")
    per_item = []
    for item in items:
        truth = "yes" if item["label"] == 1 else "no"
        row = {"truth": truth, "answers": {}}
        for a in ALPHAS:
            row["answers"][str(a)] = get_answer_under_steering(item["statement"], a)
        per_item.append(row)
    with open(PAIR_CACHE, "w") as f:
        json.dump(per_item, f, indent=2)
    print(f"Cached paired data for {len(per_item)} items")

baseline = [r for r in per_item if r["answers"]["0"] in ("yes", "no") and r["answers"]["0"] == r["truth"]]
print(f"\n=== Analysis B: paired flip test (n={len(baseline)} correct at alpha 0 ===)")
print("alpha | still correct | flipped wrong | incoherent")
flip_rows = {}
for a in ALPHAS:
    if a == 0:
        continue
    still = flipped = incoh = 0
    for r in baseline:
        ans = r["answers"][str(a)]
        if ans not in ("yes", "no"): incoh += 1
        elif ans == r["truth"]: still += 1
        else: flipped += 1
    n = len(baseline)
    flip_rows[a] = {"still": still/n, "flipped": flipped/n, "incoh": incoh/n}
    print(f"{a:5d} | {still/n:.0%} | {flipped/n:.0%} | {incoh/n:.0%}")
print("High 'still correct' -> probe fooled but behavior intact (decoupled)")

def detection_with_ci(alpha, n_boot=1000):
    steered = np.array([get_acts_with_evasion(examples[i]["prompt"], alpha) for i in deceptive_eval])
    preds = probe.predict(steered).astype(float)
    rng = np.random.default_rng(0)
    boots = [preds[rng.integers(0, len(preds), len(preds))].mean() for _ in range(n_boot)]
    return preds.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

collapse_alphas = [0,2,4,6,8,10]
collapse_ci = {a: detection_with_ci(a) for a in collapse_alphas}
print("\nCollapse curve with 95% CI:")
for a in collapse_alphas:
    m,lo,hi = collapse_ci[a]
    print(f"alpha {a:2d}: {m:.2%} [{lo:.2%}, {hi:.2%}]")

# ------------------ Fine Alpha Sweep --------------------------
alpha_fine = np.linspace(0.0, float(min_alpha), 9)
CACHE_fine = f"{RESULTS}/fine_alpha_sweep.json"

if os.path.exists(CACHE_fine):
    with open(CACHE_fine) as f:
        fine = json.load(f)
    print("Loaded cached fine-alpha sweep")
else:
    print("Running fine-alpha sweep (slow - generates text for coherence)")
    sample = items[:60]
    truths = ["yes" if it["label"] == 1 else "no" for it in sample]
    fine = {}
    for a in alpha_fine:
        a = float(a)
        det, lo, hi = detection_with_ci(a) # probe detection + 95% CI
        answers = [get_answer_under_steering(it["statement"], a) for it in sample]
        answered = [(ans, t) for ans, t in zip(answers, truths) if ans in ("yes", "no")]
        coherence = len(answered) / len(sample) # answer rate (parseable)
        correctness = np.mean([ans == t for ans, t in answered]) if answered else 0.0
        fine[f"{a:.2f}"] = {"detection": det, "lo": lo, "hi": hi, "coherence": float(coherence), "correctness": float(correctness)}
        
        print(f"alpha {a:.2f}: detection {det:.2%} [{lo:.2%}, {hi:.2%}] | "
              f"coherence {coherence:.2%} | correctness {correctness:.2%}")
    
    os.makedirs("results", exist_ok=True)
    with open(CACHE_fine, "w") as f:
        json.dump(fine, f, indent = 2)
    print("Cached fine-alpha sweep")


# --------------------- Human data validation ------------------------
# UPDATE if using new human inputs (e.g., Swapping out Dani's lines)

# Update file location here
with open("data/human_statements.json") as f:
    human_items = json.load(f)
print(f"\nLoaded {len(human_items)} human statments:", Counter(d['label'] for d in human_items))

human_examples = []
for d in human_items:
    stmt, is_true = d["statement"], d["label"]
    honest_ans = "Yes" if is_true else "No"
    lying_ans = "No" if is_true else "Yes"
    human_examples.append({"prompt": build_example(stmt, honest_ans), "deceptive": 0})
    human_examples.append({"prompt": build_example(stmt, lying_ans),  "deceptive": 1})

HUMAN_CACHE = f"{RESULTS}/acts_human.npy"
if os.path.exists(HUMAN_CACHE):
    human_acts = np.load(HUMAN_CACHE)
    print("Loaded cached human activations")
else:
    human_acts = np.array([get_answer_acts(e["prompt"]) for e in human_examples])
    np.save(HUMAN_CACHE, human_acts)
    print("Cached human activations", human_acts.shape)

y_human = np.array([e["deceptive"] for e in human_examples])

# Refit on all AI data provided then test it on the human statements
probe_full = LogisticRegression(max_iter=2000, C = 0.1).fit(acts[:, best_layer, :], y_decep)
X_human = human_acts[:, best_layer, :]
cross_acc = probe_full.score(X_human, y_human)

print(f"\n=== CROSS-DISTRIBUTION (Train AI --> test human) at layer {best_layer}: {cross_acc:.3f} ===")
print("  ~0.70 = signal is real & source-independent | ~0.50 = probe read AI artifacts")

# Layer-wise generalization curve - deos AI --> human transfer peak where the probe peaks?
print("\nAI --> human transfer by layer:")
for L in range(model.cfg.n_layers):
    p = LogisticRegression(max_iter=2000, C=0.1).fit(acts[:, L, :], y_decep)
    print(f" layer {L:2d}: {p.score(human_acts[:, L, :], y_human):.3f}")


# --------------------- Coherence / Correctness Test 1 - output -------------------

def generate_under_steer(statement, alpha, layer=None, n_new=20):
    layer = layer if layer is not None else best_layer
    tokens = model.to_tokens(answer_prompt(statement))
    hook = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook, make_evasion_hook(alpha))]):
            out = model.generate(tokens, max_new_tokens=n_new, do_sample=False, verbose=False)
    return model.to_string(out[0][tokens.shape[1]:])

print("\n=== Generations across alpha ===")
for stmt in [items[0]["statement"], items[10]["statement"], items[20]["statement"]]:
    print(f"\n{stmt}")
    for a in [0.0, 1.0, 2.0]:
        print(f" a={a}: {generate_under_steer(stmt, a)!r}")


# --------------------- Fluency under steering -------------------

def lm_ppl(text, alpha, layer = None):
    layer = layer if layer is not None else best_layer
    toks = model.to_tokens(text)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", make_evasion_hook(alpha))]):
            loss = model(toks, return_type="loss")
    return float(torch.exp(loss))

print("\n=== LM perplexity on statement text vs alpha (higher = steering degrades fluency) ===")
for a in alpha_fine:
    a = float(a)
    ppls = [lm_ppl(it["statement"], a) for it in items[:30]]
    print(f"alpha {a:.2f}: median LM perplexity {np.median(ppls):.2f}")

# ------------- Collection bias check --------------
from collections import Counter
rows = [(get_answer_under_steering(it["statement"], 0.0),
         "yes" if it["label"] == 1 else "no") for it in items]
print("Answer distribution:", Counter(a for a, _ in rows))
for name, lab in [("TRUE", "yes"), ("FALSE", "no")]:
    sub = [a == t for a, t in rows if t == lab]
    print(f"{name} statements: model correct {np.mean(sub):.0%} (n={len(sub)})")



# =====================================================================
# FIGURE 1 — probe accuracy by layer, with error bars
# =====================================================================
with open(f"{RESULTS}/sweep_stats.json") as f:
    sweep = json.load(f)
layers = sorted(int(k) for k in sweep)
means = [sweep[str(L)]["mean"] for L in layers]
stds = [sweep[str(L)]["std"] for L in layers]
 
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(layers, means, yerr=stds, marker='o', color='#2c6fbb', capsize=3, linewidth=2)
ax.axhline(0.5, ls='--', color='gray', label='chance')
peak = [L for L in layers if sweep[str(L)]["mean"] >= 0.95 * max(means)]
ax.axvspan(min(peak) - 0.5, max(peak) + 0.5, alpha=0.08, color='green', label='peak region')
ax.set_xlabel('Layer'); ax.set_ylabel('Deception-probe accuracy')
ax.set_title(f'Fig 1. Deception is linearly decodable ({model_name})')
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig1_layer_sweep.png', dpi=150); plt.close()
print("\nSaved figures/fig1_layer_sweep.png")
 
 
# =====================================================================
# FIGURE 2 — collapse curve (probe detection vs steering strength)
# =====================================================================
ms = [collapse_ci[a][0] for a in collapse_alphas]
los = [collapse_ci[a][1] for a in collapse_alphas]
his = [collapse_ci[a][2] for a in collapse_alphas]
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(collapse_alphas, ms, marker='o', color='#c0392b', linewidth=2)
ax.fill_between(collapse_alphas, los, his, alpha=0.2, color='#c0392b')
ax.axhline(0.5, ls='--', color='gray')
ax.set_xlabel('Steering strength (alpha)'); ax.set_ylabel('Probe detection of lies')
ax.set_title(f'Fig 2. Minimal steering collapses the probe (layer {best_layer})')
ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_collapse.png', dpi=150); plt.close()
print("Saved figures/fig2_collapse.png")
 
 
 
# =====================================================================
# FIGURE 4 — evasion vs behavior (probe collapses, behavior doesn't)
# =====================================================================
with open(f"{RESULTS}/behavior_paired.json") as f:
    per_item = json.load(f)
 
beh_alphas = [0, 2, 4, 6, 10]
# behavioral correctness on answered items, per alpha
beh_correct = []
for a in beh_alphas:
    answered = [r for r in per_item if r["answers"][str(a)] in ("yes", "no")]
    correct = [1 if r["answers"][str(a)] == r["truth"] else 0 for r in answered]
    beh_correct.append(np.mean(correct) if correct else 0)
 
# probe detection at those same alphas (from collapse_ci)
probe_detect = [collapse_ci.get(a, (np.nan,))[0] for a in beh_alphas]
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(beh_alphas, probe_detect, marker='o', color='#c0392b', label='probe detection of lies')
ax.plot(beh_alphas, beh_correct, marker='s', color='#27ae60', label='model behavioral correctness')
ax.axhline(0.5, ls='--', color='gray')
ax.set_xlabel('Steering strength (alpha)'); ax.set_ylabel('Rate')
ax.set_title('Fig 4. Steering collapses the probe but not behavior')
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig4_behavior.png', dpi=150); plt.close()
print("Saved figures/fig4_behavior.png")
print("\nAll figures saved to figures/. Open them to check the story holds visually.")

# =====================================================================
# FIGURE 5 — fine alpha sweep
# =====================================================================
xs   = [float(k) for k in fine]
det  = [fine[k]["detection"]   for k in fine]
lo   = [fine[k]["lo"]          for k in fine]
hi   = [fine[k]["hi"]          for k in fine]
coh  = [fine[k]["coherence"]   for k in fine]
corr = [fine[k]["correctness"] for k in fine]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, det,  marker='o', color='#c0392b', label='probe detection of lies')
ax.fill_between(xs, lo, hi, alpha=0.2, color='#c0392b')
ax.plot(xs, coh,  marker='s', color='#2c6fbb', label='output coherence (answer rate)')
ax.plot(xs, corr, marker='^', color='#27ae60', label='answer correctness')
ax.axhline(0.5, ls='--', color='gray')
ax.set_xlabel('Steering strength (alpha)'); ax.set_ylabel('Rate')
ax.set_title('Fig 5. Detection vs. coherence vs. correctness')
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig5_fine_alpha.png', dpi=150); plt.close()
print("Saved figures/fig5_fine_alpha.png")


# --- Figure 6: AI->human transfer by layer ---
transfer = [LogisticRegression(max_iter=2000, C=0.1).fit(acts[:, L, :], y_decep)
                                        .score(human_acts[:, L, :], y_human)
            for L in range(model.cfg.n_layers)]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(model.cfg.n_layers), transfer, marker='o', color='#8e44ad', linewidth=2)
ax.axhline(0.5, ls='--', color='gray', label='chance')
tpeak = [L for L in range(model.cfg.n_layers) if transfer[L] >= 0.95 * max(transfer)]
ax.axvspan(min(tpeak) - 0.5, max(tpeak) + 0.5, alpha=0.08, color='green', label='peak region')
ax.set_xlabel('Layer'); ax.set_ylabel('AI-probe accuracy on human data')
ax.set_title('Fig 6. Deception signal transfers AI to human, peaking mid-late layers')
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig6_transfer.png', dpi=150); plt.close()
print(f"Saved figures/fig6_transfer.png (peak {max(transfer):.3f} at layer {int(np.argmax(transfer))})")

# ===================== PHASE 1 ======================
# ----------- Test 1a - cross-training ---------------

# Split by STATEMENT, not example — else the honest & deceptive halves of one
# statement leak across A/B and the "independence" is fake.

Xbest, n_stmt = acts[:, best_layer, :], len(items)

def fit_dir(ex_idx):
    p = LogisticRegression(max_iter=2000, C=0.1).fit(Xbest[ex_idx], y_decep[ex_idx])
    d = p.coef_[0]; return p, d / norm(d)

def stmt_to_ex(ids): return np.array([i for i in range(len(examples)) if (i // 2) in ids])

# (A) cosine: disjoint split vs same-data noise floor (cheap, no model)
all_ex = np.arange(len(examples)); cos_dis, cos_floor = [], []
for s in range(50):
    rng = np.random.default_rng(s); perm = rng.permutation(n_stmt)
    _, dA = fit_dir(stmt_to_ex(set(perm[:n_stmt // 2])))
    _, dB = fit_dir(stmt_to_ex(set(perm[n_stmt // 2:])))
    cos_dis.append(float(dA @ dB))
    _, d1 = fit_dir(rng.choice(all_ex, len(all_ex), replace=True))
    _, d2 = fit_dir(rng.choice(all_ex, len(all_ex), replace=True))
    cos_floor.append(float(d1 @ d2))

print(f"cos disjoint   : {np.mean(cos_dis):.3f} +/- {np.std(cos_dis):.3f}")
print(f"cos noise-floor: {np.mean(cos_floor):.3f} +/- {np.std(cos_floor):.3f} (same data, resampled)")
print("disjoint << floor -> real probe-specific structure | ~equal -> mostly estimation noise")

# (B) detection transfer with seed variance (steering; ~10 min)
ALPHAS_X, K = [0,1,2,4,6], 10
def steered_best(prompt, alpha, d_t):
    def h(value, hook):
        value[:, :, :] = value[:, :, :] - alpha * d_t
        return value
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{best_layer}.hook_resid_post", h)]):
            _, c = model.run_with_cache(model.to_tokens(prompt), names_filter=RESID_ONLY)
    return c["resid_post", best_layer][0,-1,:].cpu().numpy()

detA = {a: [] for a in ALPHAS_X}; detB = {a: [] for a in ALPHAS_X}
for s in range(K):
    rng = np.random.default_rng(1000 + s); perm = rng.permutation(n_stmt)
    q = n_stmt // 2
    A, B, T = set(perm[:q-40]), set(perm[q-40:2 * q - 80]), set(perm[2 * q - 80:])
    pA, dA = fit_dir(stmt_to_ex(A)); pB, _ = fit_dir(stmt_to_ex(B))
    dA_t = torch.tensor(dA, dtype=DTYPE, device=device)
    tdec = [i for i in range(len(examples)) if examples[i]["deceptive"] == 1 and (i // 2) in T]
    for a in ALPHAS_X:
        Xs = np.array([steered_best(examples[i]["prompt"], a, dA_t) for i in tdec])
        detA[a].append(pA.predict(Xs).mean()); detB[a].append(pB.predict(Xs).mean())

print("\nalpha | detA (mean +/- sd) | detB (mean +/- sd)")
for a in ALPHAS_X:
    print(f"{a:5d} | {np.mean(detA[a]):.2f} +/- {np.std(detA[a]):.2f} | {np.mean(detB[a]):.2f} +/- {np.std(detB[a]):.2f}")

Xb = acts[:, best_layer, :]
print("\nDeception signal dimensionality (probe acc on top-k PCs):")
for k in [1, 2, 3, 5, 10, 20]:
    Xk = PCA(n_components=k, random_state=0).fit_transform(Xb)
    Xtr, Xte, ytr, yte, _, _ = gsplit(Xk, y_decep)
    acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
    print(f" top-{k:2d} PCs: {acc:.3f}")


# --------------------- Steering direction + helper (used by Fig 7 & Phase 2) -------------------

best_dir = probe.coef_[0] / norm(probe.coef_[0])
best_dir_t = torch.tensor(best_dir, dtype = DTYPE, device = device)

def hook_fixed(alpha):
    def h(value, hook):
        value[:,:,:] = value[:, :, :] - alpha * best_dir_t
        return value
    return h

def all_acts_steer_at(prompt, steer_layer, alpha):
    hn = f"blocks.{steer_layer}.hook_resid_post"
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hn, hook_fixed(alpha))]):
            _, cache = model.run_with_cache(model.to_tokens(prompt), names_filter=RESID_ONLY)
    return np.array([cache["resid_post", L][0, -1, :].cpu().numpy() for L in range(model.cfg.n_layers)])


# =======================================================
#                 FIGURE 7 
# =======================================================
fa    = sorted(float(k) for k in fine)
a50   = min(fa, key=lambda a: abs(fine[f"{a:.2f}"]["detection"] - 0.5))
acol  = float(min_alpha)
azero = float(next((a for a in collapse_alphas if collapse_ci[a][0] <= 0.01), collapse_alphas[-1]))

ALPHAS_H = sorted({round(a50, 2), acol, azero})
roles = {round(a50, 2): "Detection at 50%", acol: "collapse <10%", azero: "fully suppressed"}

nL = model.cfg.n_layers
CUT = next((L for L in range(nL) if sweep_stats[L]["mean"] > 0.55), nL // 2)
heats = {}

for A in ALPHAS_H:
    cache = f"{RESULTS}/heatmap_a{A}.npy"
    if os.path.exists(cache):
        heats[A] = np.load(cache)
    else:
        H = np.zeros((nL, nL))
        for Ls in range(nL):
            steered = np.array([all_acts_steer_at(examples[i]["prompt"], Ls, A) for i in deceptive_eval])
            for Lm in range(nL):
                H[Ls, Lm] = layer_probes[Lm].predict(steered[:, Lm, :]).mean()
        np.save(cache, H); heats[A] = H; print(f"alpha {A} done")

fig, axs = plt.subplots(1, len(ALPHAS_H), figsize=(6*len(ALPHAS_H), 6), sharey=True)
for ax, A in zip(axs, ALPHAS_H):
    im = ax.imshow(heats[A], origin="lower", aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.axvspan(-0.5, CUT-0.5, color="gray", alpha=0.45)
    ax.plot([-0.5, nL-0.5], [-0.5, nL-0.5], color="cyan", lw=1, ls="--")
    ax.set_title(f"alpha = {A:.2f} ({roles.get(A, '')})"); ax.set_xlabel("Measured at layer")
    ax.set_xlim(-0.5, nL-0.5); ax.set_ylim(-0.5, nL-0.5)
axs[0].set_ylabel("Steered at layer")
fig.colorbar(im, ax=axs, label="probe detection of lies", fraction=0.025)
fig.suptitle("Fig 7. Evasion attack window widens with steering strength")
plt.savefig(f"{FIGS}/fig7_heatmap_multi.png", dpi=150); plt.close()
print(f"Saved {FIGS}/fig7_heatmap_multi.png")



