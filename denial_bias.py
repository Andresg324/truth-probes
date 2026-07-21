import json, os

TAGS = ["0.5B", "1.5B", "3B", "7B", "14B", "gemma-2b", "gemma-9b", "llama-3b", "llama-8b"]

def root(tag):
    a = f"results/{tag}"
    return a if os.path.exists(f"{a}/p4_rows.json") else f"results_census/{tag}"

print(f"{'model':9s} {'deny_true%':>11s} {'affirm_false%':>14s} {'asymmetry':>10s}")
print("-" * 48)
for tag in TAGS:
    R = root(tag)
    if not os.path.exists(f"{R}/p4_rows.json"): print(f"{tag:9s} (no data)"); continue
    rows = json.load(open(f"{R}/p4_rows.json"))
    lie = [r for r in rows if r["cond"] == "lie"]
    true_s = [r for r in lie if r["stmt_true"] == 1]
    false_s = [r for r in lie if r["stmt_true"] == 0]
    deny = sum(r["ans"] == "no" for r in true_s) / max(len(true_s), 1)
    affirm = sum(r["ans"] == "yes" for r in false_s) / max(len(false_s), 1)
    print(f"{tag:9s} {deny*100:>10.0f}% {affirm*100:>13.0f}% {deny-affirm:>+9.2f}")
    