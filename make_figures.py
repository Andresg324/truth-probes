# Recreates figures from paper using results/*/RESULTS.json
import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # save-only, no window
import matplotlib.pyplot as plt

# -------- setup -------------------------
CURVE  = ["0.5B", "1.5B", "3B", "gemma-2b"]                         # models with per-layer curves
CENSUS = ["0.5B","1.5B","3B","7B","14B","gemma-2b","gemma-9b","llama-3b","llama-8b"]
PARA   = ["1.5B", "gemma-2b"]
IDENT  = ["gemma-2b", "llama-8b"]
NICE   = {"0.5B":"Qwen-0.5B","1.5B":"Qwen-1.5B","3B":"Qwen-3B","7B":"Qwen-7B",
          "14B":"Qwen-14B","gemma-2b":"Gemma-2b","gemma-9b":"Gemma-9b",
          "llama-3b":"Llama-3.2-3B","llama-8b":"Llama-3.1-8B"}
os.makedirs("figures/paper", exist_ok=True)

# -------- helper functions --------------
_DB = {}
def db(m):
    if m not in _DB:
        _DB[m] = json.load(open(f"results/{m}/RESULTS.json"))
    return _DB[m]

def ent(m, sec, key):
    return db(m)[sec][key]

def val(e):                                # unwrap "value"
    return e["value"] if isinstance(e, dict) and "value" in e else e

def layers(secdict):
    return sorted(int(k[1:]) for k in secdict if k.startswith("L"))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/paper/{name}.svg")
    fig.savefig(f"figures/paper/{name}.png", dpi=300)
    plt.close(fig)
    print("  wrote", name)

# ------- pulls CI band from phase2 logs data ---------------
_ABL = re.compile(
    r'ablate L\s*(?P<layer>\d+):\s*'
    r'local\s+[0-9.]+\s*->\s*[0-9.]+\s*\[(?P<llo>[0-9.]+),\s*(?P<lhi>[0-9.]+)\]'
    r'.*?deployed\s+[0-9.]+\s*->\s*[0-9.]+\s*\[(?P<dlo>[0-9.]+),\s*(?P<dhi>[0-9.]+)\]'
)
def scrape_bands():
    bands = {}
    for path in glob.glob("logs/*phase2*.log"):        # top-level logs/ only, skips pod_final copies
        m = os.path.basename(path).split("_")[0]
        bands.setdefault(m, {})
        for line in open(path):
            g = _ABL.search(line)
            if g:
                bands[m][int(g["layer"])] = {"local":    [float(g["llo"]), float(g["lhi"])],
                                             "deployed": [float(g["dlo"]), float(g["dhi"])]}
    json.dump(bands, open("figures/ci_bands.json", "w"), indent=1)
    print("bands scraped:", {m: len(v) for m, v in bands.items()})

def load_bands():
    return json.load(open("figures/ci_bands.json")) if os.path.exists("figures/ci_bands.json") else {}

def diagnostic():
    d = db("1.5B")
    print("sections:", list(d.keys()))
    print("ablation L16 :", val(ent("1.5B","phase2/ablation_curve","L16")))
    print("specificity  :", val(ent("1.5B","phase2/specificity_curve","L16")))
    print("denial_bias  :", val(ent("gemma-2b","denial_bias","rates")))
    print("leak_census  :", val(ent("gemma-2b","leak_census","row")))

def fig_decalibration():
    acc  = val(ent("3B","phase2/specificity_crit_acc","deception"))
    aucv = val(ent("3B","phase2/specificity_crit_auc","deception"))
    fig, ax = plt.subplots(figsize=(5, 4.5)); x = np.arange(2)
    ax.bar(x-0.2, [acc["clean"], aucv["clean"]], 0.4, label="clean",   color="#4C72B0")
    ax.bar(x+0.2, [acc["abl"],   aucv["abl"]],   0.4, label="ablated", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "AUC"]); ax.axhline(0.5, ls=":", color="gray")
    ax.set_ylim(0.4, 1.02); ax.set_ylabel("score"); ax.legend()
    ax.set_title("Qwen-3B critical-MLP ablation: accuracy falls, AUC intact")
    save(fig, "decalibration_3B")

def fig_damage_grid():
    B = load_bands()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.flat, CURVE):
        Ls  = layers(db(m)["phase2/ablation_curve"])
        dep = [val(ent(m,"phase2/ablation_curve",f"L{L}"))["deployed_abl"] for L in Ls]
        loc = [val(ent(m,"phase2/ablation_curve",f"L{L}"))["local_abl"]    for L in Ls]
        clean = val(ent(m,"phase2/ablation_curve",f"L{Ls[0]}"))["deployed_clean"]
        ax.axhline(clean, ls="--", lw=1, color="gray", label="deployed clean")
        ax.axhline(0.5, ls=":", lw=1, color="red")
        dline, = ax.plot(Ls, dep, "-o", label="deployed (ablated)")
        lline, = ax.plot(Ls, loc, "-s", label="local (ablated)")
        if m in B:
            xs = [L for L in Ls if str(L) in B[m]]
            ax.fill_between(xs, [B[m][str(L)]["deployed"][0] for L in xs],
                                [B[m][str(L)]["deployed"][1] for L in xs],
                            alpha=0.15, color=dline.get_color())
            ax.fill_between(xs, [B[m][str(L)]["local"][0] for L in xs],
                                [B[m][str(L)]["local"][1] for L in xs],
                            alpha=0.12, color=lline.get_color())
        ax.set_title(NICE[m]); ax.set_xlabel("ablated layer"); ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.02); ax.legend(fontsize=7)
    save(fig, "damage_grid")

def fig_specificity_grid():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.flat, CURVE):
        Ls = layers(db(m)["phase2/specificity_curve"])
        A  = [val(ent(m,"phase2/specificity_curve",f"L{L}"))["asym"] for L in Ls]
        ci = [val(ent(m,"phase2/specificity_curve",f"L{L}"))["ci"]   for L in Ls]
        lo, hi = [c[0] for c in ci], [c[1] for c in ci]
        ax.axhline(0, color="black", lw=1); ax.fill_between(Ls, lo, hi, alpha=0.2)
        ax.plot(Ls, A, "-o")
        for L, h in zip(Ls, hi):
            if h < 0: ax.axvspan(L-0.4, L+0.4, color="red", alpha=0.12)
        ax.set_title(NICE[m]); ax.set_xlabel("ablated layer")
        ax.set_ylabel("asymmetry (truth drop − deception drop)")          # CHANGED: sign convention on the axis
    save(fig, "specificity_grid")

def fig_sufficiency():
    conds  = ["clean", "steered", "steered_ablated", "patched"]
    labels = ["clean", "steered", "steered + crit ablated", "steered + clean patch"]   # CHANGED: readable labels
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in CURVE:
        cm = val(ent(m,"phase2/sufficiency","cond_means"))
        ax.plot(range(4), [cm[c] for c in conds], "-o", label=NICE[m])
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15, ha="right")       # CHANGED
    ax.set_ylabel("probe decision score"); ax.legend()
    ax.set_title("Sufficiency: clean-patch restores the steered score toward clean")
    save(fig, "sufficiency")

def fig_layer_sweep():
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in CURVE:
        vs = val(ent(m,"probe/setup","val_sweep")); Ls = sorted(int(k) for k in vs)
        ax.plot(Ls, [vs[str(L)] for L in Ls], "-o", ms=3, label=NICE[m])
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy (val)")
    ax.legend(); ax.set_title("Deception is linearly decodable, rising to a mid-late plateau")
    save(fig, "layer_sweep_overlay")

def fig_transfer():
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in CURVE:
        tb = val(ent(m,"probe/transfer","ai_to_human_by_layer")); Ls = sorted(int(k) for k in tb)
        ax.plot(Ls, [tb[str(L)] for L in Ls], "-o", ms=3, label=NICE[m])
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xlabel("layer")
    ax.set_ylabel("cross-source transfer accuracy")                       # CHANGED: second-source language
    ax.legend(); ax.set_title("Deception signal transfers across statement sources")   # CHANGED
    save(fig, "transfer_overlay")

def fig_pca():
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in CURVE:
        pc = val(ent(m,"probe/phase1","pca_dims")); ks = sorted(int(k) for k in pc)
        ax.plot(ks, [pc[str(k)] for k in ks], "-o", label=NICE[m])
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xlabel("# PCA components"); ax.set_ylabel("cross-train acc")
    ax.legend(); ax.set_title("Deception signal is low-dimensional (saturates within a few components)")
    save(fig, "pca_dims")

def fig_collapse():
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in CURVE:
        cc = val(ent(m,"probe/steering","collapse_curve")); A = sorted(int(k) for k in cc)
        mean = [cc[str(a)]["mean"] for a in A]; lo = [cc[str(a)]["lo"] for a in A]; hi = [cc[str(a)]["hi"] for a in A]
        line, = ax.plot(A, mean, "-o", ms=3, label=NICE[m]); ax.fill_between(A, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, ls="--", color="gray"); ax.set_xlabel("steering strength (alpha)")
    ax.set_ylabel("monitor detection rate under attack"); ax.legend()
    ax.set_title("Minimal steering collapses probe detection"); save(fig, "collapse_overlay")

def fig_decoupling():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, m in zip(axes.flat, CURVE):
        cc = val(ent(m,"probe/steering","collapse_curve")); Ad = sorted(int(k) for k in cc)
        det = [cc[str(a)]["mean"] for a in Ad]
        bh = val(ent(m,"probe/behavior","alpha_sweep")); Ab = sorted(int(k) for k in bh)
        cor = [bh[str(a)]["correct"] for a in Ab]; coh = [bh[str(a)]["answer_rate"] for a in Ab]
        ax.plot(Ad, det, "-o", color="#C44E52", label="probe detection")
        ax.plot(Ab, coh, "-s", color="#4C72B0", label="output coherence")
        ax.plot(Ab, cor, "-^", color="#55A868", label="answer correctness")
        ax.axhline(0.5, ls="--", color="gray"); ax.set_title(NICE[m])
        ax.set_xlabel("steering strength (alpha)"); ax.set_ylabel("rate"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=7)
    save(fig, "decoupling_grid")

def fig_relalpha():
    vals = [val(ent(m,"probe/steering","min_alpha_relative")) for m in CURVE]
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.bar([NICE[m] for m in CURVE], vals, color="#8172B3")
    ax.set_ylabel("collapse-α / residual norm")
    ax.set_title("Steering resistance in residual-relative units")        # CHANGED: old title contradicted App. C3
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); save(fig, "relative_alpha")

def fig_nine_verdicts():
    x = np.arange(len(CENSUS))
    tdec = [val(ent(m,"phase4","transfer_aucs"))["deception"] for m in CENSUS]
    dcf  = [val(ent(m,"leak_census","row"))["deconfound"]     for m in CENSUS]
    pctl = [val(ent(m,"phase4","positive_control_auc"))       for m in CENSUS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, tdec, s=60, color="#C44E52", label="naive transfer AUC (deception)")
    ax.scatter(x, dcf,  s=60, marker="D", color="#4C72B0", label="naive de-confound")
    ax.scatter(x, pctl, s=40, marker="_", color="black", label="positive-control ceiling")
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xticks(x)
    ax.set_xticklabels([NICE[m] for m in CENSUS], rotation=30, ha="right")
    ax.set_ylabel("AUC"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8)
    ax.set_title("Nine models, nine uninterpretable verdicts"); save(fig, "nine_verdicts")

def fig_denial_bias():
    x = np.arange(len(CENSUS))
    dt = [val(ent(m,"denial_bias","rates"))["deny_true"]    for m in CENSUS]
    af = [val(ent(m,"denial_bias","rates"))["affirm_false"] for m in CENSUS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-0.2, dt, 0.4, label="deny a true statement", color="#C44E52")
    ax.bar(x+0.2, af, 0.4, label="affirm a false statement", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in CENSUS], rotation=30, ha="right")
    ax.set_ylabel("rate"); ax.legend(); ax.set_title("Denial bias: how each model lies")
    save(fig, "denial_bias")

def fig_regression():
    x = np.arange(len(IDENT))
    inst = [val(ent(m,"regress_census","coefficients"))["instruction"] for m in IDENT]
    dec  = [val(ent(m,"regress_census","coefficients"))["deception"]   for m in IDENT]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x-0.2, inst, 0.4, label="instruction", color="#4C72B0")
    ax.bar(x+0.2, dec,  0.4, label="deception",   color="#C44E52")
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in IDENT])
    ax.set_ylabel("standardized coefficient"); ax.legend()
    ax.set_title("Probe transfers as an instruction detector"); save(fig, "regression_coeffs")

def fig_paraphrase():
    rows = [(m, p) for m in PARA for p in ["tf", "claim"]]; x = np.arange(len(rows)); w = 0.2
    dec_c, dec_a, tru_c, tru_a = [], [], [], []
    for m, p in rows:
        sa = val(ent(m, f"paraphrase_{p}", "specificity_local_auc"))
        dec_c.append(sa["deception"]["clean"]); dec_a.append(sa["deception"]["abl"])
        tru_c.append(sa["truth"]["clean"]);     tru_a.append(sa["truth"]["abl"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-1.5*w, dec_c, w, label="deception clean",   color="#C44E52")
    ax.bar(x-0.5*w, dec_a, w, label="deception ablated", color="#E8A0A0")
    ax.bar(x+0.5*w, tru_c, w, label="truth clean",       color="#4C72B0")
    ax.bar(x+1.5*w, tru_a, w, label="truth ablated",     color="#A0B8E8")
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xticks(x)
    ax.set_xticklabels([f"{NICE[m]}\n{p}" for m, p in rows]); ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC"); ax.legend(fontsize=7)
    ax.set_title("Specificity fingerprint is format-stable: deception drops, truth holds"); save(fig, "paraphrase_fingerprint")

ALL = [fig_decalibration, fig_damage_grid, fig_specificity_grid, fig_sufficiency,
       fig_layer_sweep, fig_transfer, fig_pca,
       fig_collapse, fig_decoupling, fig_relalpha,
       fig_nine_verdicts, fig_denial_bias, fig_regression, fig_paraphrase]

if __name__ == "__main__":
    scrape_bands()                         # build figures/ci_bands.json from the logs
    for fn in ALL:
        try:
            fn()
        except Exception as e:
            print(f"  SKIPPED {fn.__name__}: {type(e).__name__}: {e}")
    print("\ndone -> figures/paper/")