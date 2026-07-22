import json, os, glob, datetime

TAG_ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "gemma-2b", "gemma-9b", "llama-3b", "llama-8b"]

def load_all():
    out, man = {}, {}
    for path in glob.glob("results/*/RESULTS.json"):
        tag = os.path.basename(os.path.dirname(path))
        try:
            out[tag] = json.load(open(path))
        except Exception as e:
            out[tag] = {"_error": str(e)}
    for path in glob.glob("results/*/MANIFEST.json"):
        tag = os.path.basename(os.path.dirname(path))
        try:
            man[tag] = json.load(open(path))
        except Exception:
            pass
    return out, man

def order(tags):
    known = [t for t in TAG_ORDER if t in tags]
    rest = sorted(t for t in tags if t not in TAG_ORDER)
    return known + rest

def fmt(v):
    # Render a reported entry
    if not isinstance(v, dict) or "value" not in v:
        return f"`{v}`"
    val, ci, n = v["value"], v.get("ci"), v.get("n")
    if isinstance(val, float):
        s = f"{val:.3f}"
    elif isinstance(val, dict):
        s = ", ".join(f"{k}={val[k]:.3f}" if isinstance(val[k], float) else f"{k} = {val[k]}" for k in val)
    else:
        s = str(val)
    if isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
        s += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    if n is not None:
        s += f" (n={n})"
    return s

def main():
    data, man = load_all()
    tags = order(list(data))
    L = ["# MEASURED.md - generated source of truth (DO NOT MANUALLY EDIT)\n",
         f"*Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from results/*/RESULTS.json "
         f"If a number is not here, it was not measured. Regenerate - do not edit.*\n",
         "## Provenance\n",
         "| model | script | date | noproc |", "|---|---|---|---|"]
    for t in tags:
        m=man.get(t,{})
        L.append(f"| {t} | {m.get('script', '?')} | {str(m.get('date', '?'))[:16]} | {m.get('noproc','-')} |")
    L.append("")

    all_sections = sorted({s for t in tags for s in data[t] if not s.startswith("_")})
    for sec in all_sections:
        L.append(f"## {sec}\n")
        keys = []
        for t in tags:
            for k in data[t].get(sec, {}):
                if k not in keys:
                    keys.append(k)
        L.append("| model | " + " | ".join(keys) + " |")
        L.append("|" + "---|" * (len(keys) + 1))
        for t in tags:
            sd = data[t].get(sec)
            if not sd:
                continue
            L.append("| " + " | ".join([t] + [fmt(sd[k]) if k in sd else "-" for k in keys]) + " |")
        L.append("")

    open("MEASURED.md", "w").write("\n".join(L))
    print(f"wrote MEASURED.md ({len(all_sections)} sections, {len(tags)} models)")

    print("\nsection names seen (check for casing/prefic drift):")
    for s in all_sections:
        print(f" {s:35s} {[t for t in tags if s in data[t]]}")
    
if __name__ == "__main__":
    main()