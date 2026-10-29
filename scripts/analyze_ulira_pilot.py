"""
scripts/analyze_ulira_pilot.py
==============================
Compare U-LiRA calibrations on the SAME checkpoints (ledger D4/E29/E30):

  v3      stored `ulira_leak`: target vs UNTOUCHED baselines (IN) / oracles (OUT), raw negELBO,
          biased pooled sigma, mean posterior.            [what we had]
  v3log   same, log-scaled, unbiased sigma.               [cheap fix: no shadow unlearning]
  v4raw   Hayes-faithful IN (same method+config on baseline_ts1..3 + the target), raw scale.
  v4      Hayes-faithful IN, log scale, LOO attack advantage.     [chosen design]

Needs the shadow runs from scripts/run_ulira_shadows.py. Writes
results/<ds>/ulira_v4_pilot.json and does NOT touch dense_recovery.json.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import run_ulira as R                                       # shadow_dirs, scores_over, forget_loader
from src.evaluation import ulira as U
from src.evaluation.membership import per_sample_negative_elbo
from notebooks.nb_utils import load_vae_checkpoint

TAGS = ("ts1", "ts2", "ts3")


def ck(ds, method, run, tag=None):
    d = ROOT / f"results/{ds}/{ds}_{method}/{run}{('_bl' + tag) if tag else ''}"
    return d / "final.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dsprites")
    a = ap.parse_args()
    ds = a.dataset
    rows = json.loads((ROOT / f"results/{ds}/dense_recovery.json").read_text())
    by = {(r["method"], r["run"]): r for r in rows}
    loader = R.forget_loader(ds)
    IN0, OUT = R.shadow_dirs(ds)
    base_raw = R.scores_over(IN0, loader)       # untouched baselines (n_in, N)
    out_raw = R.scores_over(OUT, loader)        # oracles (n_out, N)
    N = base_raw.shape[1]
    print(f"{ds}: forget n={N}  untouched IN={len(IN0)}  OUT={len(OUT)}", flush=True)

    # positive control: "no unlearning" == untouched baselines vs oracles
    ctl = U.score_point(base_raw, out_raw, log=True)
    print(f"positive control (no unlearning): adv={ctl['adv']:.3f} auc={ctl['auc']:.4f} "
          f"gap={ctl['gap_sigma_in']:.1f} sigma_in", flush=True)

    # operating points that have all shadows
    pts = []
    for (method, run), r in by.items():
        if method in ("baseline", "oracle"):
            continue
        sh = [ck(ds, method, run, t) for t in TAGS]
        if ck(ds, method, run).exists() and all(p.exists() for p in sh):
            pts.append((method, run))
    pts.sort()
    print(f"{len(pts)} operating points have the target + {len(TAGS)} shadows", flush=True)

    out = []
    for method, run in pts:
        r = by[(method, run)]
        mats = []
        for p in [ck(ds, method, run)] + [ck(ds, method, run, t) for t in TAGS]:
            m, _ = load_vae_checkpoint(p, R.DEVICE)
            mats.append(per_sample_negative_elbo(m, loader, R.DEVICE))
        in_unl = np.vstack(mats)                     # row 0 = the target itself
        tgt = in_unl[0]
        v3_re = U.target_posterior(tgt, base_raw, out_raw, unbiased=False)
        v3log = U.target_posterior(U.log_scale(tgt), U.log_scale(base_raw), U.log_scale(out_raw), unbiased=True)
        v4 = U.score_point(in_unl, out_raw, log=True)
        v4raw = U.score_point(in_unl, out_raw, log=False)
        rec = dict(method=method, run=run, attack=r["attack_at_r"], tofu_ks=r["tofu_ks"], val=r["val"],
                   v3_stored=r["ulira_leak"], v3_recomputed=v3_re, v3log=v3log,
                   v4_adv=v4["adv"], v4_post_in=v4["post_in"], v4_post_out=v4["post_out"], v4_auc=v4["auc"],
                   v4_sat=v4["sat_frac"], v4_sigma_in=v4["sigma_in"], v4_sigma_out=v4["sigma_out"],
                   v4_gap_sigma_in=v4["gap_sigma_in"],
                   v4raw_adv=v4raw["adv"], v4raw_post_in=v4raw["post_in"], v4raw_sat=v4raw["sat_frac"],
                   v4raw_sigma_in=v4raw["sigma_in"], mean_negelbo=float(tgt.mean()),
                   shadow_mean_negelbo=[float(x.mean()) for x in in_unl[1:]])
        out.append(rec)
        print(f"  {method:<16}{run:<22} v3={rec['v3_stored']:.4f} v3log={v3log:.4f} | "
              f"v4adv={v4['adv']:.3f} post_in={v4['post_in']:.3f} sat={v4['sat_frac']:.2f} "
              f"sig_in={v4['sigma_in']:.3f} | raw adv={v4raw['adv']:.3f}", flush=True)

    (ROOT / f"results/{ds}/ulira_v4_pilot.json").write_text(json.dumps(
        {"positive_control": ctl, "points": out}, indent=2))
    if len(out) < 4:
        print("too few points for the summary"); return

    g = lambda k: np.array([o[k] for o in out], dtype=float)
    print("\n================ SUMMARY (n=%d points) ================" % len(out))
    print(f"harness check  max|v3_recomputed - v3_stored| = {np.abs(g('v3_recomputed') - g('v3_stored')).max():.2e}")
    for name, key, thr in [("v3 stored", "v3_stored", 0.01), ("v3log (cheap)", "v3log", 0.01),
                           ("v4raw post_in", "v4raw_post_in", 0.01), ("v4 post_in", "v4_post_in", 0.01)]:
        v = g(key)
        print(f"{name:<16} points <{thr}: {int((v < thr).sum()):>2}/{len(v)}   "
              f">0.99: {int((v > 0.99).sum()):>2}   mid(0.05-0.95): {int(((v > .05) & (v < .95)).sum()):>2}   "
              f"distinct(2dp): {len(set(np.round(v, 2)))}")
    print(f"mean fraction of examples saturated (|Lambda|>20):  v4raw={g('v4raw_sat').mean():.2f}   v4(log)={g('v4_sat').mean():.2f}")
    print(f"pooled sigma_in:  v3 untouched-baselines=%.3f   v4raw unlearned-IN median=%.3f   v4 log median=%.3f"
          % (R.gaussian_params(base_raw)[1], np.median(g('v4raw_sigma_in')), np.median(g('v4_sigma_in'))))
    for key, nm in [("attack", "active attack@r"), ("tofu_ks", "TOFU KS")]:
        y = g(key)
        s = "  ".join(f"{lab}={spearmanr(g(k), y).statistic:+.2f}" for lab, k in
                      [("v3", "v3_stored"), ("v3log", "v3log"), ("v4raw_adv", "v4raw_adv"), ("v4_adv", "v4_adv")])
        print(f"Spearman vs {nm:<16} {s}")
    print("\nper-method v4 advantage (min/median/max):")
    for m in sorted(set(o["method"] for o in out)):
        v = np.array([o["v4_adv"] for o in out if o["method"] == m])
        print(f"  {m:<16} n={len(v):<2} {v.min():.3f} / {np.median(v):.3f} / {v.max():.3f}")


if __name__ == "__main__":
    main()
