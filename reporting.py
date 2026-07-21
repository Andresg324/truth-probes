import json, os, subprocess, sys, datetime

def report(tag, section, key, value, ci=None, n=None):
    # Records all printed numbers, overwriting the same key - to get consistent, updated results
    os.makedirs(f"results/{tag}", exist_ok=True)
    path = f"results/{tag}/RESULTS.json"
    db = json.load(open(path)) if os.path.exists(path) else {}
    db.setdefault(section, {})[key] = {"value": value, "ci": ci, "n":n}
    json.dump(db, open(path, "w"), indent = 2, default = float)

def manifest(tag, **extra):
    os.makedirs(f"results/{tag}", exist_ok=True)
    m = {"date": str(datetime.datetime.now()), "argv": sys.argv, "pip": subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.split("\n"), **extra}
    json.dump(m, open(f"results/{tag}/MANIFEST.json", "w"), indent = 2)
