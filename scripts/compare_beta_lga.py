"""
scripts/compare_beta_lga.py
===========================
The beta=1 vs beta=4 verdict: does beta=4's stronger latent organisation make LGA
(and its components) predict adversarial recoverability, where beta=1 did not?

For each variant's dense_recovery.json it reports, for the proposed metric LGA and
its components (CKA-to-oracle, Mahalanobis oracle-gap) plus the trivial no-oracle
baseline dir_norm:
  * marginal Spearman(metric, attack@r) with a 95% bootstrap CI, over the method
    operating points (refs dropped);
  * the within-method partial (method-mean-removed ranks) -- the confound-controlled
    number that decided the beta=1 finding.
A metric that "works" shows a large, CI-excludes-0 correlation that SURVIVES the
within-method control. Run: python scripts/compare_beta_lga.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga

CRIT = "attack_at_r"
METRICS = [("lga", "LGA (proposed)"), ("cka_oracle", "  CKA-to-oracle"),
           ("oracle_gap", "  Maha oracle-gap"), ("dir_norm", "dir_norm (no oracle)")]


def load(variant):
    rows = json.loads((ROOT / f"results/{variant}/dense_recovery.json").read_text())
    inject_lga(rows)
    return [r for r in rows if r["method"] not in ("baseline", "oracle")
            and r.get("lga") is not None and r.get(CRIT) is not None]


def boot_spear(x, y, n=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            continue
        s.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = (np.percentile(s, [2.5, 97.5]) if s else (float("nan"), float("nan")))
    return rho, lo, hi


def within_method_partial(rows, key, min_range=0.0):
    rx, ry, per = [], [], []
    meth = {}
    for r in rows:
        meth.setdefault(r["method"], []).append(r)
    for m, rs in meth.items():
        vals = [r[key] for r in rs]
        if len(set(vals)) < 2 or (max(vals) - min(vals)) < min_range:
            continue
        if len({r[CRIT] for r in rs}) < 2:
            continue
        x = rankdata(vals); y = rankdata([r[CRIT] for r in rs])
        rx += list(x - x.mean()); ry += list(y - y.mean())
        if len(rs) >= 4:
            per.append(spearmanr(vals, [r[CRIT] for r in rs]).statistic)
    pr = spearmanr(rx, ry).statistic if len(rx) >= 4 else float("nan")
    return pr, per


def between_method(rows, key):
    """Method-mean(key) vs method-mean(attack) -- the confound axis. If this is
    strong while the within-method partial is weak, the marginal is method-identity."""
    import numpy as np
    meth = {}
    for r in rows:
        meth.setdefault(r["method"], []).append(r)
    mx = [np.mean([r[key] for r in rs]) for rs in meth.values() if len(rs) >= 2]
    my = [np.mean([r[CRIT] for r in rs]) for rs in meth.values() if len(rs) >= 2]
    return spearmanr(mx, my).statistic if len(mx) >= 3 else float("nan")


def report(variant, label):
    rows = load(variant)
    print(f"\n{'='*90}\n{label}   (n={len(rows)} method operating points)\n{'='*90}")
    print(f"{'metric':<22}{'marginal [95% CI]':<26}{'within-method':<15}{'within(movers)':<15}{'between-method':<14}")
    print("-" * 90)
    for key, name in METRICS:
        rho, lo, hi = boot_spear([r[key] for r in rows], [r[CRIT] for r in rows])
        pr, per = within_method_partial(rows, key)
        pr_mov, _ = within_method_partial(rows, key, min_range=0.15)  # methods that really move it
        btw = between_method(rows, key)
        sig = "*" if not (lo <= 0 <= hi) else " "
        signs = f"({sum(c<0 for c in per)}-,{sum(c>0 for c in per)}+)" if per else ""
        print(f"{name:<22}{f'{rho:+.2f}[{lo:+.2f},{hi:+.2f}]{sig}':<26}"
              f"{f'{pr:+.2f}{signs}':<15}{f'{pr_mov:+.2f}':<15}{f'{btw:+.2f}':<14}")
    return rows


def main():
    # Every (beta=1 base, beta=4 variant) pair whose dense_recovery.json exists.
    pairs = [("dsprites", "dsprites_b4", "dSprites"),
             ("mpi3d", "mpi3d_b4", "MPI3D"),
             ("celeba", "celeba_b4", "CelebA")]
    for base, variant, nice in pairs:
        if not (ROOT / f"results/{variant}/dense_recovery.json").exists():
            continue
        report(base, f"beta=1 {nice} (original)")
        report(variant, f"beta=4 {nice} (more latent organisation)")
    print(f"\n{'='*74}\nVERDICT\n{'='*74}")
    print("If beta=4's LGA marginal AND within-method partial are strongly negative and")
    print("CI-excludes-0 where beta=1's were ~0/ns, then more latent organisation DID make")
    print("geometry predict recoverability. If both stay ~0/ns, the negative result")
    print("generalises to beta=4 too (geometry-alignment is not a valid proxy regardless).")


if __name__ == "__main__":
    main()
