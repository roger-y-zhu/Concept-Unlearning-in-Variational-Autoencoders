"""
scripts/run_ulira_shadows.py
============================
Run the SAME unlearning configuration on the replicate baselines (baseline_ts1..3) so that
U-LiRA (Hayes et al. 2024) gets its proper IN distribution Q1 = "trained WITH the concept, then
unlearned by the algorithm under test" (ledger D4). Without these, run_ulira.py can only
compare a model to UNTOUCHED baselines, which measures baseline-likeness, not whether an
unlearning process is distinguishable from a retrain (ledger E29).

Operating points come from results/<ds>/dense_recovery.json (method + run name). Each run name
is inverted to the CLI flags that produced it; the inversion is VERIFIED by checking the run
directory the unlearn runner creates is exactly `<run>_bl<tag>` (the runner derives that name
from the flags we pass, so any mis-parse changes the name and is caught here).

Resumable: the runner skips a run whose final.pt exists.
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPOCH = {"dsprites": "epoch_0150.pt", "celeba": "epoch_0250.pt", "mpi3d": "epoch_0120.pt"}
EPOCH.update({f"{d}_b4": e for d, e in list(EPOCH.items())})   # beta=4 variants: same epoch counts
NUM = r"[\d.]+(?:e-?\d+)?"
MIN_FREE_GB = 3.0

# run-name -> CLI flags (inverse of each UnlearnMethod.run_name; only the plain families)
PARSERS = {
    "neg_grad":        (rf"^lam({NUM})_ep(\d+)$",                 lambda m: ["--lambda-forget", m[0], "--epochs", m[1]]),
    "finetune":        (rf"^lam({NUM})_ep(\d+)$",                 lambda m: ["--lambda-forget", m[0], "--epochs", m[1]]),
    "rmu":             (rf"^a({NUM})_c({NUM})_ep(\d+)$",          lambda m: ["--alpha", m[0], "--c", m[1], "--epochs", m[2]]),
    "feature_unlearn": (rf"^alpha({NUM})_ep(\d+)(_n1)?(?:_ms({NUM}))?(_pt1)?(_te1)?(?:_ea({NUM}))?$",
                        lambda m: ["--alpha", m[0], "--epochs", m[1]]
                        + (["--recon-norm", "l1"] if m[2] else [])
                        + (["--ms-ssim", m[3]] if m[3] else [])
                        + (["--project-to", "threshold"] if m[4] else [])
                        + (["--train-encoder"] if m[5] else [])
                        + (["--elbo-anchor", m[6]] if m[6] else [])),
    "fisher":          (rf"^sig({NUM})_ep1$",                     lambda m: ["--noise-scale", m[0]]),
    "leace":           (rf"^leace_ep1(?:_a({NUM}))?$",            lambda m: (["--erase-strength", m[0]] if m[0] else [])),
    "inlp":            (rf"^k(\d+)_ep1$",                         lambda m: ["--n-iter", m[0]]),
}

# Pilot: spans weak -> strong -> overshoot for all 7 methods; cheap (1-epoch) methods first.
PILOT = {
    "dsprites": [
        ("fisher", "sig0.004_ep1"), ("fisher", "sig0.012_ep1"),
        ("leace", "leace_ep1_a0.01"), ("leace", "leace_ep1_a0.2"), ("leace", "leace_ep1_a0.5"), ("leace", "leace_ep1"),
        ("inlp", "k1_ep1"), ("inlp", "k2_ep1"), ("inlp", "k4_ep1"),
        ("rmu", "a3.0_c1.0_ep2"), ("rmu", "a10.0_c1.0_ep2"), ("rmu", "a115.0_c1.0_ep4"), ("rmu", "a50.0_c1.0_ep5"),
        ("rmu", "a2000.0_c5.0_ep5"),
        ("neg_grad", "lam0.0001_ep10"), ("neg_grad", "lam0.008_ep10"), ("neg_grad", "lam0.1_ep10"),
        ("feature_unlearn", "alpha0.5_ep10"), ("feature_unlearn", "alpha10.0_ep20"),
        ("finetune", "lam0.0_ep30"),
        ("rmu", "a125.0_c1.25_ep40"),
    ],
}


# Per-method quotas for the stratified subset (~20 points). Within a method, points are spread
# evenly (extremes included) over the FORGET-SET LOSS, i.e. unlearning intensity -- deliberately
# not over either criterion (attack / U-LiRA) so the selection cannot bias their comparison.
QUOTA = {"neg_grad": 3, "rmu": 5, "leace": 3, "inlp": 2, "fisher": 2, "feature_unlearn": 3, "finetune": 1}


def stratified(rows, quota=QUOTA):
    import numpy as np
    pts = []
    for method, k in quota.items():
        cand = [r for r in rows if r["method"] == method and cli_for(method, r["run"]) is not None]
        cand.sort(key=lambda r: r["forget"])
        if not cand:
            continue
        idx = sorted(set(np.linspace(0, len(cand) - 1, min(k, len(cand))).round().astype(int)))
        pts += [(method, cand[i]["run"]) for i in idx]
    return pts


def cli_for(method: str, run: str):
    pat, fn = PARSERS[method]
    m = re.match(pat, run)
    return None if m is None else fn(m.groups())


def expected_dir(dataset: str, method: str, run: str, tag: str) -> Path:
    return ROOT / f"results/{dataset}/{dataset}_{method}/{run}_bl{tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(EPOCH))
    ap.add_argument("--tags", nargs="+", default=["ts1", "ts2", "ts3"])
    ap.add_argument("--points", choices=["pilot", "stratified", "all"], default="pilot")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    ds = a.dataset

    if a.points == "pilot":
        pts = PILOT[ds]
    else:
        rows = json.loads((ROOT / f"results/{ds}/dense_recovery.json").read_text())
        pts = (stratified(rows) if a.points == "stratified"
               else [(r["method"], r["run"]) for r in rows if r["method"] in PARSERS])
    todo, skipped = [], []
    for method, run in pts:
        flags = cli_for(method, run)
        if flags is None:
            skipped.append((method, run)); continue
        todo.append((method, run, flags))
    print(f"{ds}: {len(todo)} operating points x {len(a.tags)} shadows = {len(todo)*len(a.tags)} runs"
          f"   (unparseable, skipped: {skipped})", flush=True)

    ok = bad = done = 0
    t_start = time.time()
    for method, run, flags in todo:
        for tag in a.tags:
            bl = ROOT / f"results/{ds}/{ds}_baseline_{tag}/{EPOCH[ds]}"
            out = expected_dir(ds, method, run, tag)
            if (out / "final.pt").exists():
                done += 1; continue
            free_gb = shutil.disk_usage(ROOT).free / 1e9
            if free_gb < MIN_FREE_GB:        # a full disk silently truncates torch.save (seen once)
                print(f'!! STOP: only {free_gb:.1f} GB free (< {MIN_FREE_GB}); free space and re-run '
                      f'(resumable)', flush=True)
                sys.exit(3)
            if not bl.exists():
                print(f"!! missing replicate baseline {bl}", flush=True); bad += 1; continue
            cmd = [sys.executable, "-u", str(ROOT / "experiments/unlearn/run.py"),
                   "--dataset", ds, "--method", method, *flags, "--baseline-ckpt", str(bl),
                   "--save-every", "0"]   # shadows are only ever read via final.pt; per-epoch ckpts
                                          # filled the disk (Errno 28) during the first extension
            print(f">>> [{time.time()-t_start:6.0f}s] {method} {run} {tag}", flush=True)
            if a.dry:
                print("    ", " ".join(cmd)); continue
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc == 0 and (out / "final.pt").exists():
                ok += 1
            else:
                bad += 1
                print(f"!! FAILED rc={rc} or name mismatch (expected {out})", flush=True)
    print(f"DONE ok={ok} already={done} failed={bad}", flush=True)


if __name__ == "__main__":
    main()
