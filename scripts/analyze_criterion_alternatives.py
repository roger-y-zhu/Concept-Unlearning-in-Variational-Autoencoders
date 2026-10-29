"""
scripts/analyze_criterion_alternatives.py
=========================================
Is attack@r the right recoverability criterion?  Compare it with radius-free summaries of the same
attack and with a WEIGHT-SPACE relearning attack (scripts/run_relearn_attack.py), on the same
operating points, and ask whether the project's negative finding (geometry / TOFU / U-LiRA do not
predict recoverability) is specific to attack@r.   (supervisor action item, Oct 2026; no GPU)

Candidate criteria (higher = more recoverable, all computed per model):
  attack@r        decoder-side input attack at the report radius (current gold standard)
  attack_auc      mean attack success over the whole radius grid (radius-free)
  attack_-r50     minus the (interpolated) radius at which success reaches 0.5 (radius-free)
  relearnA        mean over attacker budgets s of  A(s) = (E_oracle'(s) - E_model'(s)) / (E_oracle(0) - E_baseline(0))
                  ['forget' arm: fine-tune on k forget examples; E' = held-out forget negELBO after s steps]
  relearnR        the same LEVEL score for the RETAIN arm: R(s) = (E_o^ret(s) - E_m^ret(s)) / (E_o(0) - E_b(0)).
                  Benign relearning (Hu et al. 2024; Deeb & Roger 2024): the attacker never sees the concept.
  relearnX        EXCESS concept-specific relearning: X(s) = [(E_m^ret - E_m^for) - (E_o^ret - E_o^for)] / gap, defined ONLY
                  for models that actually forgot (forget_depth >= 0.25). Unrestricted, X fails the reference check (a model
                  that still knows the concept has nothing to regain, so baseline scores BELOW oracle), hence the restriction.
  relearnF        HEADROOM-normalised excess: F(s) = (E_m^ret - E_m^for)/(E_m(0) - E_b(0)) - (E_o^ret - E_o^for)/gap.
                  The fraction of the model's OWN lost gap that seeing the concept (beyond generic repair) wins back,
                  minus the same fraction for the oracle (learning the concept from scratch). Oracle-like model => 0.
                  Added 2026-10-16 AFTER seeing that A, R and X are confounded with forget depth (rho -0.96/-0.98/+0.73):
                  the criterion family was refined post hoc, and every variant is reported.
  forget_depth    (E_m(0) - E_baseline(0)) / gap : 0 = still behaves like the baseline on held-out forget images, 1 = like the oracle.
                  LEVEL scores (A, R) are confounded with this: a model that never forgot stays near baseline after any attack.
  linear/mlp AUROC  passive decodability (for reference)

Questions: (V1) do the criteria sit right on the references (baseline knows / oracle does not)?
(V2) do they agree with each other (pooled and method-controlled)?  (V3) are they saturated?
(V4) does "metric X does not predict recoverability" survive the change of criterion?
Statistics follow scripts/export_poster_data.py (ranks within (dataset, beta) cell / n, pooled Pearson;
bootstrap resampling rows within cells, seed 0).  Writes results/analysis_oct2026/criterion_alternatives.{json,md,png}.
"""
from __future__ import annotations
import json, math, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(os.environ.get("CU_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga  # noqa: E402

DATASETS = ["dsprites", "celeba", "mpi3d"]
FAMS = [("b1", ""), ("b4", "_b4")]
B = 1000
OUT = ROOT / "results" / "analysis_oct2026"
CRIT = ["attack_r", "attack_auc", "attack_r50", "relearnA", "relearnR", "relearnX", "relearnF", "forget_depth", "mlp_auroc", "linear_auroc"]
CRIT_LABEL = dict(attack_r="attack@r", attack_auc="attack AUC(radii)", attack_r50="attack -r50", relearnA="relearn A",
                  relearnR="relearn R (retain-only, benign)", relearnX="relearn X (excess, forgetting models)", relearnF="relearn F (headroom-normalised)", forget_depth="forget depth E0", mlp_auroc="MLP AUROC (passive)", linear_auroc="linear AUROC (passive)")
METRICS = [("lga", "LGA"), ("tofu_ks", "TOFU KS"), ("ulira_leak", "U-LiRA v3 leak"), ("ulira_adv", "U-LiRA v4 adv"),
           ("dir_norm", "dir_norm"), ("knn_cohesion", "kNN cohesion")]


def canon(m):
    return "neg_grad" if m == "finetune" else m


def r50(att):
    xs = sorted((float(k), v) for k, v in att.items())
    prev = None
    for r, v in xs:
        if v >= 0.5:
            if prev is None:
                return r
            (r0, v0) = prev
            return r0 + (0.5 - v0) * (r - r0) / (v - v0) if v > v0 else r
        prev = (r, v)
    return xs[-1][0] * 1.25                     # never reaches 0.5 on the grid: cap just beyond the largest radius


def relearn_scores(rec, refs, steps, min_depth=0.25):
    """rec/refs: {'forget': {step: E}, 'retain': {step: E}}; refs = (baseline, oracle) rows.
    -> (A, R, X, forget_depth, F); X and F are None unless the model forgot at least min_depth of the baseline->oracle gap."""
    b, o = refs
    gap = o["forget"]["0"] - b["forget"]["0"]
    if not gap > 1e-6:
        return None, None, None, None, None
    depth = (rec["forget"]["0"] - b["forget"]["0"]) / gap
    A, Rr, X, Fh = [], [], [], []
    for s in steps:
        s = str(s)
        A.append((o["forget"][s] - rec["forget"][s]) / gap)
        Rr.append((o["retain"][s] - rec["retain"][s]) / gap)
        X.append(((rec["retain"][s] - rec["forget"][s]) - (o["retain"][s] - o["forget"][s])) / gap)
        hr = rec["forget"]["0"] - b["forget"]["0"]
        Fh.append((rec["retain"][s] - rec["forget"][s]) / hr - (o["retain"][s] - o["forget"][s]) / gap if hr > 1e-6 else float("nan"))
    ok_d = depth >= min_depth
    return (float(np.mean(A)), float(np.mean(Rr)), (float(np.mean(X)) if ok_d else None), float(depth),
            (float(np.mean(Fh)) if ok_d else None))


def load():
    cells, refs = {}, {}
    for ds in DATASETS:
        for fam, suf in FAMS:
            pj = ROOT / "results" / f"{ds}{suf}" / "dense_recovery.json"
            pr = ROOT / "results" / f"{ds}{suf}" / "relearn_attack.json"
            if not (pj.exists() and pr.exists()):
                continue
            rows = json.loads(pj.read_text()); inject_lga(rows)
            rl = json.loads(pr.read_text()); steps = rl["meta"]["eval_at"]
            rmap = {(r["method"], r["run"]): r for r in rl["rows"]}
            rb, ro = rmap.get(("baseline", "baseline")), rmap.get(("oracle", "oracle"))
            if rb is None or ro is None:
                continue
            drow = {r["method"]: r for r in rows if r["method"] in ("baseline", "oracle")}
            ref = dict(rb=rb, ro=ro, steps=steps)

            def crit(r, rrow):
                a = r.get("attack") or {}
                A, Rl, Xs, depth, Fs = relearn_scores(rrow, (rb, ro), steps) if rrow else (None, None, None, None, None)
                return dict(attack_r=r.get("attack_at_r"), attack_auc=float(np.mean(list(a.values()))) if a else None,
                            attack_r50=-r50(a) if a else None, relearnA=A, relearnR=Rl, relearnX=Xs, relearnF=Fs, forget_depth=depth,
                            mlp_auroc=r.get("mlp_auroc"), linear_auroc=r.get("linear_auroc"))
            refs[(ds, fam)] = {m: crit(drow[m], rmap.get((m, m))) for m in ("baseline", "oracle") if m in drow}
            keep = []
            for r in rows:
                if r["method"] in ("baseline", "oracle"):
                    continue
                rrow = rmap.get((r["method"], r["run"]))
                if rrow is None:
                    continue
                d = dict(_ds=ds, _fam=fam, _m=canon(r["method"]), run=r["run"], in_budget=r.get("in_budget"), val=r.get("val"))
                d.update(crit(r, rrow))
                for k, _ in METRICS:
                    d[k] = r.get(k)
                keep.append(d)
            cells[(ds, fam)] = keep
    return cells, refs


# ------------------------------------------------------------------ stats
def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def gb(rows, keyf):
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    return g


def ok(r, *ks):
    return all(r.get(k) is not None and isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ks)


def pooled(rows, x, y):
    xs, ys = [], []
    for rr in gb(rows, lambda r: (r["_ds"], r["_fam"])).values():
        rr = [r for r in rr if ok(r, x, y)]
        if len(rr) < 3:
            continue
        n = len(rr)
        xs += list(rankdata([r[x] for r in rr]) / n); ys += list(rankdata([r[y] for r in rr]) / n)
    return pearson(xs, ys)


def within(rows, x, y, min_n=5):
    xs, ys = [], []
    for rr in gb(rows, lambda r: (r["_ds"], r["_fam"], r["_m"])).values():
        rr = [r for r in rr if ok(r, x, y)]
        if len(rr) < min_n:
            continue
        a, b = rankdata([r[x] for r in rr]), rankdata([r[y] for r in rr])
        xs += list(a - a.mean()); ys += list(b - b.mean())
    return pearson(xs, ys)


def boot(rows, fn, B=B, seed=0):
    rng = np.random.default_rng(seed)
    cells = list(gb(rows, lambda r: (r["_ds"], r["_fam"], r["_m"])).values())
    est = fn(rows); v = []
    for _ in range(B):
        rs = [rr[i] for rr in cells for i in rng.integers(0, len(rr), len(rr))]
        t = fn(rs)
        if math.isfinite(t):
            v.append(t)
    lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 50 else (float("nan"),) * 2
    n = sum(1 for _ in rows)
    return dict(rho=round(est, 3) if math.isfinite(est) else None, lo=round(float(lo), 3), hi=round(float(hi), 3), n=n)


def fm(e):
    return "n/a" if not e or e["rho"] is None else f"{e['rho']:+.2f} [{e['lo']:+.2f},{e['hi']:+.2f}]"


def main():
    cells, refs = load()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for rr in cells.values() for r in rr]
    print("cells:", {f"{d}_{f}": len(v) for (d, f), v in cells.items()})
    R = dict(n={f"{d}_{f}": len(v) for (d, f), v in cells.items()})
    L = ["# Alternatives to attack@r (generated by scripts/analyze_criterion_alternatives.py)\n",
         f"Rows with relearning data: {len(rows)} ({', '.join(f'{d}_{f}: {len(v)}' for (d, f), v in cells.items())}).\n"]

    # V1 references
    L += ["## V1 - reference models through each criterion (baseline should be HIGH, oracle LOW)\n",
          "| cell | criterion | baseline | oracle | baseline > oracle? |", "|---|---|---|---|---|"]
    R["V1"] = {}
    for (d, f), rf in refs.items():
        for c in CRIT:
            b, o = rf.get("baseline", {}).get(c), rf.get("oracle", {}).get(c)
            if b is None or o is None:
                continue
            R["V1"][f"{d}_{f}|{c}"] = dict(baseline=b, oracle=o)
            L.append(f"| {d} {f} | {CRIT_LABEL[c]} | {b:.3f} | {o:.3f} | {'yes' if b > o else '**NO**'} |")

    # V2 convergent matrix
    L += ["\n## V2 - agreement between criteria: pooled rho  (method-controlled rho in brackets)\n",
          "| | " + " | ".join(CRIT_LABEL[c] for c in CRIT) + " |", "|---|" + "---|" * len(CRIT)]
    R["V2"] = {}
    for a in CRIT:
        cellsv = []
        for b in CRIT:
            if a == b:
                cellsv.append("1"); continue
            p, w = pooled(rows, a, b), within(rows, a, b)
            R["V2"][f"{a}|{b}"] = dict(pooled=p, within=w)
            cellsv.append(f"{p:+.2f} ({w:+.2f})")
        L.append(f"| {CRIT_LABEL[a]} | " + " | ".join(cellsv) + " |")

    # V3 saturation
    L += ["\n## V3 - saturation / usable range (share of unlearned models within 10% of the baseline-oracle span of either reference)\n",
          "| cell | criterion | at oracle floor | at baseline ceiling | middle (informative) |", "|---|---|---|---|---|"]
    R["V3"] = {}
    for (d, f), rr in cells.items():
        for c in CRIT:
            b, o = refs[(d, f)].get("baseline", {}).get(c), refs[(d, f)].get("oracle", {}).get(c)
            vals = [r[c] for r in rr if ok(r, c)]
            if b is None or o is None or not vals or abs(b - o) < 1e-9:
                continue
            span = abs(b - o)
            lo_end, hi_end = (o, b) if b > o else (b, o)
            fl = np.mean([abs(v - o) <= 0.1 * span or (v - o) * np.sign(b - o) < 0 for v in vals])
            ce = np.mean([abs(v - b) <= 0.1 * span or (v - b) * np.sign(b - o) > 0 for v in vals])
            R["V3"][f"{d}_{f}|{c}"] = dict(floor=float(fl), ceiling=float(ce))
            L.append(f"| {d} {f} | {CRIT_LABEL[c]} | {fl:.0%} | {ce:.0%} | {max(0, 1 - fl - ce):.0%} |")

    # V4: does the negative finding survive the criterion change?
    L += ["\n## V4 - rho(metric, criterion): does 'metric X does not predict recoverability' depend on the criterion?\n",
          "Pooled rho [95% CI] / method-controlled rho [95% CI].  n = rows with both values.\n",
          "| metric | criterion | n | pooled | within-method |", "|---|---|---|---|---|"]
    R["V4"] = {}
    for mk, ml in METRICS:
        for c in ["attack_r", "attack_auc", "relearnA", "relearnR", "relearnX", "relearnF", "forget_depth"]:
            rr = [r for r in rows if ok(r, mk, c)]
            if len(rr) < 20:
                continue
            e1, e2 = boot(rr, lambda z: pooled(z, mk, c)), boot(rr, lambda z: within(z, mk, c))
            R["V4"][f"{mk}|{c}"] = dict(pooled=e1, within=e2)
            L.append(f"| {ml} | {CRIT_LABEL[c]} | {len(rr)} | {fm(e1)} | {fm(e2)} |")

    # per dataset for LGA vs relearn
    L += ["\n## V4b - LGA vs criteria, per dataset x beta (pooled within the cell)\n",
          "| cell | n | vs attack@r | vs attack AUC | vs relearn A | vs relearn F |", "|---|---|---|---|---|---|"]
    R["V4b"] = {}
    for (d, f), rr in cells.items():
        row = []
        for c in ["attack_r", "attack_auc", "relearnA", "relearnF"]:
            z = [r for r in rr if ok(r, "lga", c)]
            v = pearson(rankdata([r["lga"] for r in z]), rankdata([r[c] for r in z])) if len(z) > 4 else float("nan")
            row.append(v); R["V4b"][f"{d}_{f}|{c}"] = v
        L.append(f"| {d} {f} | {len(rr)} | " + " | ".join("n/a" if not math.isfinite(v) else f"{v:+.2f}" for v in row) + " |")

    L += ["\n## V6 - dependence on forgetting depth (how much each criterion merely re-measures 'did the model forget?')\n",
          "| criterion | n | pooled rho with forget depth | method-controlled |", "|---|---|---|---|"]
    R["V6"] = {}
    for c in CRIT:
        if c == "forget_depth":
            continue
        rr = [r for r in rows if ok(r, c, "forget_depth")]
        if len(rr) < 20:
            continue
        e1, e2 = boot(rr, lambda z: pooled(z, c, "forget_depth"), B=500), boot(rr, lambda z: within(z, c, "forget_depth"), B=500)
        R["V6"][c] = dict(pooled=e1, within=e2)
        L.append(f"| {CRIT_LABEL[c]} | {len(rr)} | {fm(e1)} | {fm(e2)} |")
    (OUT / "criterion_alternatives.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "criterion_alternatives.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # figure: attack@r vs relearn X ranks per dataset
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = dict(neg_grad="#4c72b0", feature_unlearn="#55a868", rmu="#c44e52", fisher="#8172b2", leace="#937860", inlp="#da8bc3")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
        for a, ds in zip(ax, DATASETS):
            for fam, mk in (("b1", "o"), ("b4", "^")):
                rr = [r for r in cells.get((ds, fam), []) if ok(r, "attack_r", "relearnF")]
                for m in cols:
                    z = [r for r in rr if r["_m"] == m]
                    if z:
                        a.scatter([r["attack_r"] for r in z], [r["relearnF"] for r in z], s=26, marker=mk, c=cols[m],
                                  alpha=.8, label=m if (fam == "b1" and ds == "dsprites") else None)
            a.set_title(ds); a.set_xlabel("attack@r"); a.set_ylabel("relearn F (headroom-normalised excess)")
            a.axhline(0, c="#aaa", lw=.8)
        fig.legend(loc="lower center", ncol=6, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "criterion_scatter.png", dpi=150)
    except Exception as e:
        print("figure skipped:", e)
    print("\n".join(L[:60]))
    print(f"-> {OUT/'criterion_alternatives.md'}")


if __name__ == "__main__":
    main()
