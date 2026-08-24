# Recreates figures from paper using results/*/RESULTS.json
#
# Writes .pdf (for LaTeX), .png (for README/preview), and .svg to figures/paper/.

import json, os, re, glob, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # save-only, no window
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, embeds properly; Type 3 risks desk rejection
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                             # paths below are repo-root relative

# -------- setup -------------------------
CURVE  = ["0.5B", "1.5B", "3B", "gemma-2b"]                         # models with per-layer curves
CENSUS = ["0.5B","1.5B","3B","7B","14B","gemma-2b","gemma-9b","llama-3b","llama-8b"]
PARA   = ["1.5B", "gemma-2b"]
IDENT  = ["gemma-2b", "llama-8b"]
NICE   = {"0.5B":"Qwen-0.5B","1.5B":"Qwen-1.5B","3B":"Qwen-3B","7B":"Qwen-7B",
          "14B":"Qwen-14B","gemma-2b":"Gemma-2b","gemma-9b":"Gemma-9b",
          "llama-3b":"Llama-3.2-3B","llama-8b":"Llama-3.1-8B"}
MCOLOR = {"0.5B":"#8172B3","1.5B":"#4C72B0","3B":"#C44E52","gemma-2b":"#55A868"}
OUT    = "figures/paper"
os.makedirs(OUT, exist_ok=True)

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
    fig.savefig(f"{OUT}/{name}.pdf")                    # vector, for LaTeX
    fig.savefig(f"{OUT}/{name}.png", dpi=300)           # raster, for README
    fig.savefig(f"{OUT}/{name}.svg")
    plt.close(fig)
    print("  wrote", name, "(pdf/png/svg)")

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
    print("crit_L 3B    :", val(ent("3B","phase2/setup","crit_L")))
    print("crit acc 3B  :", ent("3B","phase2/specificity_crit_acc","paired_asym"))
    print("denial_bias  :", val(ent("gemma-2b","denial_bias","rates")))
    print("leak_census  :", val(ent("gemma-2b","leak_census","row")))

# ------- helpers for the two metric-comparison figures ------
def crit_layer(m):
    """Pre-registered critical MLP for this model."""
    return int(val(ent(m, "phase2/setup", "crit_L")))

def asym_from(entry_dec, entry_tru):
    """Paired asymmetry = truth drop - deception drop.
    Negative => ablation hurts deception readout more than truth readout."""
    return (entry_tru["clean"] - entry_tru["abl"]) - (entry_dec["clean"] - entry_dec["abl"])

def crit_asym_acc(m):
    """(asymmetry, ci_or_None) at the critical layer, balanced-accuracy metric."""
    dec = val(ent(m, "phase2/specificity_crit_acc", "deception"))
    tru = val(ent(m, "phase2/specificity_crit_acc", "truth"))
    a, ci = asym_from(dec, tru), None
    try:
        raw = ent(m, "phase2/specificity_crit_acc", "paired_asym")   # raw: val() drops "ci"
        if isinstance(raw, dict):
            ci = raw.get("ci")
            a  = raw.get("value", raw.get("asym", a))
        else:
            a = raw
    except Exception:
        pass
    return a, ci

def crit_asym_auc(m):
    """(asymmetry, ci) at the critical layer, AUC metric, from the per-layer curve."""
    e = val(ent(m, "phase2/specificity_curve", f"L{crit_layer(m)}"))
    return e["asym"], e.get("ci")

# ============================================================
# FIGURES
# ============================================================

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
        ax.set_ylabel("asymmetry (truth drop − deception drop)")
    save(fig, "specificity_grid")

def fig_sufficiency():
    conds  = ["clean", "steered", "steered_ablated", "patched"]
    labels = ["clean", "steered", "steered + crit ablated", "steered + clean patch"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in CURVE:
        cm = val(ent(m,"phase2/sufficiency","cond_means"))
        ax.plot(range(4), [cm[c] for c in conds], "-o", label=NICE[m])
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=15, ha="right")
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
    ax.set_ylabel("cross-source transfer accuracy")
    ax.legend(); ax.set_title("Deception signal transfers across statement sources")
    save(fig, "transfer_overlay")

def fig_pca():
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in CURVE:
        pc = val(ent(m,"probe/phase1","pca_dims")); ks = sorted(int(k) for k in pc)
        ax.plot(ks, [pc[str(k)] for k in ks], "-o", label=NICE[m])
    ax.axhline(0.5, ls=":", color="gray"); ax.set_xlabel("# PCA components"); ax.set_ylabel("cross-train acc")
    ax.legend(); ax.set_title("Components required for the signal fall with model scale")
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
    ax.set_title("Steering resistance in residual-relative units")
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

# ------- NEW: metric divergence ------------------------------
def fig_metric_divergence():
    """The thesis in one figure: same intervention, two metrics, two verdicts.
    A -- crit-layer paired asymmetry under balanced accuracy (with CI) vs AUC.
    B -- accuracy drop vs AUC drop for every cell where both metrics were recorded."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))

    x, w = np.arange(len(CURVE)), 0.35
    acc_a, acc_err, auc_a = [], [[], []], []
    for m in CURVE:
        a, ci = crit_asym_acc(m)
        acc_a.append(a)
        acc_err[0].append(max(0.0, a - ci[0]) if ci else 0.0)
        acc_err[1].append(max(0.0, ci[1] - a) if ci else 0.0)
        auc_a.append(crit_asym_auc(m)[0])

    axA.bar(x - w/2, acc_a, w, yerr=acc_err, capsize=4,
            label="balanced accuracy", color="#C44E52")
    axA.bar(x + w/2, auc_a, w, label="AUC (threshold-free)", color="#4C72B0")
    axA.axhline(0, color="black", lw=1)
    axA.set_xticks(x); axA.set_xticklabels([NICE[m] for m in CURVE], rotation=15, ha="right")
    axA.set_ylabel("paired asymmetry (truth drop − deception drop)")
    axA.set_title("A. Critical layer: the metric decides whether a mechanism exists")
    axA.legend(fontsize=8)
    if "3B" in CURVE:
        i3 = CURVE.index("3B")
        axA.annotate("significant under accuracy,\nabsent under AUC",
                     xy=(i3 - w/2, acc_a[i3]), xytext=(i3 - 0.15, acc_a[i3] - 0.10),
                     fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))

    lim, n_cells = 0.0, 0
    for m in CURVE:
        Ls = layers(db(m).get("phase2/distance_curve", {}))
        if not Ls:
            continue
        dacc, dauc = [], []
        for L in Ls:
            e = val(ent(m, "phase2/distance_curve", f"L{L}"))
            dacc.append(e["acc_clean"] - e["acc_abl"])
            dauc.append(e["auc_clean"] - e["auc_abl"])
        n_cells += len(Ls); lim = max(lim, max(dacc + dauc))
        axB.scatter(dauc, dacc, s=55, color=MCOLOR[m], label=NICE[m],
                    edgecolor="white", linewidth=0.6, zorder=3)
        if m == "3B" and crit_layer(m) in Ls:
            e = val(ent(m, "phase2/distance_curve", f"L{crit_layer(m)}"))
            axB.annotate(f"L{crit_layer(m)}",
                         (e["auc_clean"] - e["auc_abl"], e["acc_clean"] - e["acc_abl"]),
                         textcoords="offset points", xytext=(7, -3), fontsize=8)

    lim = max(lim, 0.05) * 1.12
    axB.plot([0, lim], [0, lim], ls="--", lw=1, color="gray", zorder=1)
    axB.text(lim*0.62, lim*0.70, "metrics agree", fontsize=7.5, color="gray",
             rotation=45, ha="center", va="center")
    axB.set_xlim(-0.01, lim); axB.set_ylim(-0.01, lim)
    axB.set_xlabel("AUC drop under ablation")
    axB.set_ylabel("balanced-accuracy drop under ablation")
    axB.set_title(f"B. De-calibration across {n_cells} per-layer cells")
    axB.legend(fontsize=8, loc="upper left")
    save(fig, "metric_divergence")

# ------- NEW: local vs deployed ------------------------------
def fig_local_vs_deployed(models=("1.5B", "3B")):
    """Does local ablation damage reach the deployed monitor? At 1.5B yes; at 3B no,
    including at layers where the specificity asymmetry is significant."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.6*len(models), 4.6), squeeze=False)
    for ax, m in zip(axes.flat, models):
        Ls = layers(db(m)["phase2/ablation_curve"])
        loc_d, dep_d = [], []
        for L in Ls:
            e = val(ent(m, "phase2/ablation_curve", f"L{L}"))
            loc_d.append(e["local_clean"] - e["local_abl"])
            dep_d.append(e["deployed_clean"] - e["deployed_abl"])
        for L in layers(db(m)["phase2/specificity_curve"]):
            ci = val(ent(m, "phase2/specificity_curve", f"L{L}")).get("ci")
            if ci and ci[1] < 0:
                ax.axvspan(L-0.45, L+0.45, color="red", alpha=0.11, zorder=0)
        ax.axhline(0, color="black", lw=1)
        ax.plot(Ls, loc_d, "-s", color="#4C72B0", label="local (at ablated layer)")
        ax.plot(Ls, dep_d, "-o", color="#C44E52", label="deployed (at the monitor)")
        ax.axvline(crit_layer(m), ls=":", lw=1, color="gray")
        ax.set_title(f"{NICE[m]}  (shaded = significant deception-specificity)", fontsize=10)
        ax.set_xlabel("ablated layer"); ax.set_ylabel("AUC drop under ablation")
        ax.legend(fontsize=8)
    ymax = max(a.get_ylim()[1] for a in axes.flat); ymin = min(a.get_ylim()[0] for a in axes.flat)
    for a in axes.flat:
        a.set_ylim(ymin, ymax)
    save(fig, "local_vs_deployed")

# ============================================================
# registry + CLI
# ============================================================
FIGS = {
    "decalibration_3B":       fig_decalibration,
    "damage_grid":            fig_damage_grid,
    "specificity_grid":       fig_specificity_grid,
    "sufficiency":            fig_sufficiency,
    "layer_sweep_overlay":    fig_layer_sweep,
    "transfer_overlay":       fig_transfer,
    "pca_dims":               fig_pca,
    "collapse_overlay":       fig_collapse,
    "decoupling_grid":        fig_decoupling,
    "relative_alpha":         fig_relalpha,
    "nine_verdicts":          fig_nine_verdicts,
    "denial_bias":            fig_denial_bias,
    "regression_coeffs":      fig_regression,
    "paraphrase_fingerprint": fig_paraphrase,
    "metric_divergence":      fig_metric_divergence,
    "local_vs_deployed":      fig_local_vs_deployed,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Regenerate paper figures from results/*/RESULTS.json")
    p.add_argument("names", nargs="*", help="figure names to build (default: all)")
    p.add_argument("--list", action="store_true", help="list figure names and exit")
    p.add_argument("--no-scrape", action="store_true", help="skip rebuilding figures/ci_bands.json")
    p.add_argument("--diagnostic", action="store_true", help="print a few raw entries and exit")
    args = p.parse_args()

    if args.list:
        for k in FIGS: print(" ", k)
        sys.exit(0)
    if args.diagnostic:
        diagnostic(); sys.exit(0)

    unknown = [n for n in args.names if n not in FIGS]
    if unknown:
        print("unknown figure(s):", ", ".join(unknown))
        print("available:", ", ".join(FIGS)); sys.exit(1)

    todo = args.names or list(FIGS)
    if not args.no_scrape and ("damage_grid" in todo or not args.names):
        scrape_bands()

    for name in todo:
        try:
            FIGS[name]()
        except Exception as e:
            print(f"  SKIPPED {name}: {type(e).__name__}: {e}")
    print(f"\ndone -> {OUT}/")