"""v6: every learned method retrained with ONE common epoch grid.

The v4/v5 runs used different grids per method (AdaptiveFusion up to 150,
DCCA-style and DIMvLN up to 200, BML up to 400). Here every learned method selects
its budget from the same grid by the same 3-fold inner cross-validation:

    EPOCH GRID = 25, 50, 100, 200, 400

Everything else (splits, seeds, training augmentation, the 12 test sets,
protocols A and B, the methods) is identical to v4/v5. Old results are not touched:
all output goes to results/v6 (results_quick/v6 with --quick).

Order (fast parts first, so the main tables exist early if the run is interrupted):
  1. AdaptiveFusion (+ SP-RR / Uniform / Const-beta), Concat-SVM, MAXVAR-GCCA, DCCA-style
  2. BML: full and capacity-matched, each with and without the controller
  3. DIMvLN (slowest)
  4. AUROC / localization from the saved AdaptiveFusion models
  5. reports, then one zip to send back

Usage (normally through run1.bat):
    python run1.py           full run
    python run1.py --quick   2 seeds, tiny grid: checks that everything works
Interrupted runs resume: finished (method, seed) pairs are skipped.
"""
import argparse, datetime, importlib, json, os, platform, subprocess, sys, time, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["MFeat", "Caltech101-20", "Scene15", "BDGP", "LandUse21", "Reuters1200"]
SEEDS = "3,7,11,17,23,29,31,37,41,43"
GRID = "25,50,100,200,400"


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s); sys.__stdout__.flush()
        self.f.write(s); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    res = os.path.join(ROOT, "results_quick" if a.quick else "results", "v6")
    logs = os.path.join(ROOT, "logs")
    os.makedirs(res, exist_ok=True); os.makedirs(logs, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Tee(os.path.join(logs, f"run1_{stamp}.log"))
    status_path = os.path.join(res, "status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}

    def say(msg):
        log.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")

    def run(cmd, name):
        say(f"START {name}: {' '.join(cmd)}")
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            log.write("    " + line)
        p.wait()
        ok = p.returncode == 0
        say(f"{'DONE' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)")
        status[name] = "done" if ok else f"failed (exit {p.returncode})"
        json.dump(status, open(status_path, "w"), indent=2)
        return ok

    # ---- environment record ----
    env_lines = [f"date: {stamp}", f"epoch grid: {GRID}", f"python: {sys.version}",
                 f"platform: {platform.platform()}"]
    for mod in ["torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib"]:
        try: env_lines.append(f"{mod}: {importlib.import_module(mod).__version__}")
        except Exception as e: env_lines.append(f"{mod}: not available ({e})")
    try:
        import torch
        env_lines.append(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available(): env_lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception: pass
    open(os.path.join(res, "environment.txt"), "w").write("\n".join(env_lines) + "\n")
    say("Environment:\n  " + "\n  ".join(env_lines))

    py = sys.executable
    seeds = "3,7" if a.quick else SEEDS
    grid = ["--quick"] if a.quick else ["--grid", GRID]
    udir = os.path.join(res, "unified"); bdir = os.path.join(res, "v5")

    # ---- 1. AdaptiveFusion family and feature-level baselines ----
    for ds in DATASETS:
        run([py, "-m", "benchmark.unified", "--dataset", ds, "--seeds", seeds,
             "--groups", "AF,CONCAT,GCCA,DCCA", "--out-dir", udir] + grid, f"v6_fast_{ds}")
    # ---- 2. BML: capacity-matched first (fast), then full width ----
    for models, tag in [("SMALL_FIXED,SMALL_CTRL", "bml_small"), ("FULL_FIXED,FULL_CTRL", "bml_full")]:
        for ds in DATASETS:
            run([py, "-m", "benchmark.v5_bml", "--dataset", ds, "--seeds", seeds,
                 "--models", models, "--out-dir", bdir] + grid, f"v6_{tag}_{ds}")
    # ---- 3. DIMvLN (slowest) ----
    for ds in DATASETS:
        run([py, "-m", "benchmark.unified", "--dataset", ds, "--seeds", seeds,
             "--groups", "DIMvLN", "--out-dir", udir] + grid, f"v6_dimvln_{ds}")
    # ---- 4. corruption detection from the saved AdaptiveFusion models ----
    run([py, "-m", "benchmark.auroc_from_models", "--models", os.path.join(udir, "models"),
         "--out", os.path.join(res, "auroc.csv")], "v6_auroc")
    # ---- 5. reports ----
    run([py, "-m", "benchmark.report", "--results", res], "v6_report")
    run([py, "-m", "benchmark.v5_report", "--results", res], "v6_bml_report")

    # ---- pack ----
    zname = os.path.join(ROOT, f"{'QUICK_' if a.quick else ''}results_v6_for_review_{stamp}.zip")
    with zipfile.ZipFile(zname, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(res):
            for fn in files:
                p = os.path.join(base, fn)
                z.write(p, os.path.relpath(p, ROOT))
        for fn in os.listdir(logs):
            if fn.startswith("run1_"):
                z.write(os.path.join(logs, fn), os.path.join("logs", fn))
    say(f"All results packed into: {zname}")
    failed = [k for k, v in status.items() if v != "done"]
    if failed:
        say("Some steps failed: " + ", ".join(failed) + ". Run run1.bat again to retry them.")
    else:
        say("Everything finished successfully.")


if __name__ == "__main__":
    main()
