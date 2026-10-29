"""
scripts/analyze_lga_sign.py
===========================
What does the SIGN of LGA's correlation with attack@r mean?  (supervisor action item, Oct 2026)

LGA = 0.5*(CKA_oracle - CKA_base)/(1 - CKA_base) + 0.5*(1 - gap/gap_base): higher = latent geometry
closer to the retrained oracle.  A useful proxy would give rho(LGA, attack@r) < 0 ("more oracle-like =>
less recoverable").  The project found small, sign-unstable correlations.  This script tests WHY, with
six pre-specified decompositions on results/{ds}{,_b4,_btc}/dense_recovery.json (no GPU):

  D1 between- vs within-method   pooled rho vs method-demeaned rho vs correlation of method means
  D2 dose control                partial Spearman(LGA, attack | unlearning strength) inside method cells;
                                 strength = retain-val-loss rank (the same axis the dense study stratifies on)
  D3 non-monotone dose-response  within method: rho(dose, LGA), rho(dose, attack), and the gain from a quadratic
                                 attack~dose fit (an interior peak/trough => sign depends on operating region)
  D4 decoder damage              rho restricted to in-budget models (val within the retain budget)
  D5 model family                the same estimates for beta=1 / beta=4 / beta-TCVAE (3 datasets each)
  D6 components                  CKA-to-oracle and (-oracle_gap) separately

Conventions identical to scripts/export_poster_data.py: baseline/oracle rows excluded; ranks within each
(dataset, family) cell divided by n; pooled Pearson on ranks (= Spearman); percentile bootstrap (B=2000,
seed 0) resampling rows WITHIN their (dataset, family, method) cell.
Writes results/analysis_oct2026/lga_sign.{json,md}.   Root: env CU_ROOT.
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
FAMS = [("b1", ""), ("b4", "_b4"), ("btc", "_btc")]
Y = "attack_at_r"
B = 2000
OUT = ROOT / "results" / "analysis_oct2026"


def canon(m):                       # finetune is the lambda=0 NegGrad point (ledger E22): one family
    return "neg_grad" if m == "finetune" else m


def load():
    cells = {}                      # (ds, fam) -> list of row dicts
    for ds in DATASETS:
        for fam, suf in FAMS:
            p = ROOT / "results" / f"{ds}{suf}" / "dense_recovery.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())
            try:
                inject_lga(rows)
            except Exception as e:                              # no baseline row etc.
                print(f"skip {ds}{suf}: {e}"); continue
            keep = []
            for r in rows:
                if r["method"] in ("baseline", "oracle"):
                    continue
                if r.get("lga") is None or r.get(Y) is None or not math.isfinite(r[Y]) or r.get("val") is None:
                    continue
                r = dict(r); r["_ds"], r["_fam"], r["_m"] = ds, fam, canon(r["method"])
                r["_ncka"] = r.get("cka_oracle")
                r["_ngap"] = -r["oracle_gap"] if r.get("oracle_gap") is not None else None
                keep.append(r)
            if len(keep) >= 5:
                cells[(ds, fam)] = keep
    return cells


# --------------------------------------------------------------------------------- primitives
def rk(v):
    return rankdata(v) / len(v)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def group_by(rows, keyf):
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    return g


def flat(cells, fams=None):
    return [r for (ds, f), rr in cells.items() if fams is None or f in fams for r in rr]


def _vec(rows, k):
    return np.array([r[k] for r in rows], float)


def pooled(rows, xk):
    """rank within (ds,fam) cell /n, pooled Pearson  (= the poster convention)."""
    xs, ys = [], []
    for rr in group_by(rows, lambda r: (r["_ds"], r["_fam"])).values():
        rr = [r for r in rr if r.get(xk) is not None]
        if len(rr) < 3:
            continue
        xs += list(rk(_vec(rr, xk))); ys += list(rk(_vec(rr, Y)))
    return pearson(xs, ys)


def within(rows, xk, min_n=5):
    """method-demeaned ranks inside each (ds,fam,method) cell, pooled."""
    xs, ys = [], []
    for rr in group_by(rows, lambda r: (r["_ds"], r["_fam"], r["_m"])).values():
        rr = [r for r in rr if r.get(xk) is not None]
        if len(rr) < min_n:
            continue
        x, y = rankdata(_vec(rr, xk)), rankdata(_vec(rr, Y))
        xs += list(x - x.mean()); ys += list(y - y.mean())
    return pearson(xs, ys)


def partial(rows, xk, dose="val", min_n=6):
    """within-cell partial Spearman(x, y | dose): residualise rank(x), rank(y) on rank(dose)."""
    xs, ys = [], []
    for rr in group_by(rows, lambda r: (r["_ds"], r["_fam"], r["_m"])).values():
        rr = [r for r in rr if r.get(xk) is not None and r.get(dose) is not None]
        if len(rr) < min_n:
            continue
        d = rankdata(_vec(rr, dose)); d = d - d.mean()
        res = []
        for v in (rankdata(_vec(rr, xk)), rankdata(_vec(rr, Y))):
            v = v - v.mean()
            beta = (d @ v) / (d @ d) if d @ d > 0 else 0.0
            res.append(v - beta * d)
        xs += list(res[0]); ys += list(res[1])
    return pearson(xs, ys)


def between(rows, xk):
    """per (ds,fam): correlation across METHOD MEANS of the within-cell ranks; pooled over cells (n=#methods)."""
    xs, ys = [], []
    for cell in group_by(rows, lambda r: (r["_ds"], r["_fam"])).values():
        cell = [r for r in cell if r.get(xk) is not None]
        if len(cell) < 5:
            continue
        rx, ry = rk(_vec(cell, xk)), rk(_vec(cell, Y))
        by = defaultdict(list)
        for r, a, b in zip(cell, rx, ry):
            by[r["_m"]].append((a, b))
        if len(by) < 3:
            continue
        mx = np.array([np.mean([p[0] for p in v]) for v in by.values()])
        my = np.array([np.mean([p[1] for p in v]) for v in by.values()])
        xs += list(mx - mx.mean()); ys += list(my - my.mean())
    return pearson(xs, ys)


def boot(rows, fn, B=B, seed=0):
    """percentile CI, resampling rows within (ds,fam,method) cells."""
    rng = np.random.default_rng(seed)
    cells = list(group_by(rows, lambda r: (r["_ds"], r["_fam"], r["_m"])).values())
    est = fn(rows)
    vals = []
    for _ in range(B):
        rs = [rr[i] for rr in cells for i in rng.integers(0, len(rr), len(rr))]
        v = fn(rs)
        if math.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 50 else (float("nan"),) * 2
    return dict(rho=round(est, 3), lo=round(float(lo), 3), hi=round(float(hi), 3), n=len(rows))


def f(e):
    return "n/a" if e is None or not math.isfinite(e["rho"]) else f"{e['rho']:+.2f} [{e['lo']:+.2f},{e['hi']:+.2f}] n={e['n']}"


# --------------------------------------------------------------------------------- analyses
def sign_table(cells, fams, xk="lga"):
    rows = flat(cells, fams)
    return dict(pooled=boot(rows, lambda r: pooled(r, xk)),
                within=boot(rows, lambda r: within(r, xk)),
                between=boot(rows, lambda r: between(r, xk)),
                partial_dose=boot(rows, lambda r: partial(r, xk)))


def per_method(cells, fams):
    out = {}
    rows = flat(cells, fams)
    for m in sorted({r["_m"] for r in rows}):
        rr = [r for r in rows if r["_m"] == m]
        out[m] = dict(pooled=boot(rr, lambda r: pooled(r, "lga"), B=500),
                      partial_dose=boot(rr, lambda r: partial(r, "lga"), B=500))
    return out


def dose_response(cells, fams, min_n=6):
    """per (ds,fam,method) cell: rho(dose,LGA), rho(dose,attack), quadratic gain, location of attack extremum."""
    res = []
    for (ds, fam), rr in cells.items():
        if fam not in fams:
            continue
        for m, cc in group_by(rr, lambda r: r["_m"]).items():
            if len(cc) < min_n:
                continue
            d = rk(_vec(cc, "val")); a = _vec(cc, Y); l = _vec(cc, "lga")
            X1 = np.c_[np.ones_like(d), d]; X2 = np.c_[X1, d ** 2]
            ra = rankdata(a)

            def r2(X, y):
                c, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ c
                return 1 - (e @ e) / (((y - y.mean()) ** 2).sum() or 1.0)
            r1, r2_ = r2(X1, ra), r2(X2, ra)
            c2, *_ = np.linalg.lstsq(X2, ra, rcond=None)
            ext = -c2[1] / (2 * c2[2]) if abs(c2[2]) > 1e-9 else float("nan")
            res.append(dict(ds=ds, fam=fam, method=m, n=len(cc), rho_dose_lga=round(spearman(d, l), 2),
                            rho_dose_attack=round(spearman(d, a), 2), r2_lin=round(r1, 2),
                            r2_quad=round(r2_, 2), quad_gain=round(r2_ - r1, 2),
                            extremum_dose=round(float(ext), 2) if np.isfinite(ext) else None,
                            curvature=("hump" if c2[2] < 0 else "trough")))
    return res


def mediation(cells, fams, min_n=6):
    """D7: if unlearning DOSE mediates, rho(LGA,attack) in a cell ~ rho(LGA,dose)*rho(dose,attack).
    Cells with a constant LGA (decoder-only feature_unlearn: encoder frozen => LGA == 0 by construction) are skipped."""
    out = []
    for (ds, fam), rr in cells.items():
        if fam not in fams:
            continue
        for m, cc in group_by(rr, lambda r: r["_m"]).items():
            if len(cc) < min_n:
                continue
            l, a, d = _vec(cc, "lga"), _vec(cc, Y), _vec(cc, "val")
            if np.ptp(l) < 1e-9 or np.ptp(a) < 1e-9:
                continue
            obs, ld, da = spearman(l, a), spearman(l, d), spearman(d, a)
            out.append(dict(ds=ds, fam=fam, method=m, n=len(cc), obs=round(obs, 2), rho_lga_dose=round(ld, 2),
                            rho_dose_attack=round(da, 2), predicted=round(ld * da, 2)))
    obs = np.array([o["obs"] for o in out]); pred = np.array([o["predicted"] for o in out])
    ok = np.isfinite(obs) & np.isfinite(pred)
    strong = ok & (np.abs(pred) > 0.25) & (np.abs(obs) > 0.25)
    rng = np.random.default_rng(0)
    bs = []
    idx = np.where(ok)[0]
    for _ in range(B):
        j = rng.choice(idx, len(idx)); v = spearman(obs[j], pred[j])
        if math.isfinite(v):
            bs.append(v)
    summ = dict(n_cells=int(ok.sum()), spearman_obs_vs_pred=round(spearman(obs[ok], pred[ok]), 2),
                ci=[round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])],
                sign_agree_strong=f"{int((np.sign(obs[strong]) == np.sign(pred[strong])).sum())}/{int(strong.sum())}",
                sign_agree_all=f"{int((np.sign(obs[ok]) == np.sign(pred[ok])).sum())}/{int(ok.sum())}")
    return summ, out


def main():
    cells = load()
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(cells={f"{d}_{fm}": len(v) for (d, fm), v in cells.items()})
    for tag, fams in (("b1", ["b1"]), ("b4", ["b4"]), ("btc", ["btc"]), ("b1+b4 (poster)", ["b1", "b4"]),
                      ("all three", ["b1", "b4", "btc"])):
        R[f"D1_{tag}"] = sign_table(cells, fams)
    # D4: in-budget only
    inb = {k: [r for r in v if r.get("in_budget")] for k, v in cells.items()}
    inb = {k: v for k, v in inb.items() if len(v) >= 5}
    for tag, fams in (("b1+b4 (poster)", ["b1", "b4"]), ("btc", ["btc"])):
        R[f"D4_inbudget_{tag}"] = sign_table(inb, fams)
        R[f"D4_inbudget_{tag}"]["n_removed"] = sum(len(cells[k]) - len(inb.get(k, [])) for k in cells if k[1] in fams)
    # D6: components
    for xk, nm in (("_ncka", "CKA-to-oracle"), ("_ngap", "-oracle_gap (Mahalanobis)")):
        rows = [r for r in flat(cells, ["b1", "b4"]) if r.get(xk) is not None]
        R[f"D6_{nm}"] = dict(pooled=boot(rows, lambda r: pooled(r, xk), B=1000),
                             within=boot(rows, lambda r: within(r, xk), B=1000))
    nofu = {k: [r for r in v if r["_m"] != "feature_unlearn"] for k, v in cells.items()}
    nofu = {k: v for k, v in nofu.items() if len(v) >= 5}
    for tag, fams in (("b1+b4 (poster) excl. feature_unlearn", ["b1", "b4"]), ("all three excl. feature_unlearn", ["b1", "b4", "btc"])):
        R[f"D1_{tag}"] = sign_table(nofu, fams)
    for tag, fams in (("b1+b4", ["b1", "b4"]), ("btc", ["btc"]), ("all three", ["b1", "b4", "btc"])):
        summ, cellsout = mediation(cells, fams)
        R[f"D7_{tag}"] = dict(summary=summ, cells=cellsout)
    R["D2_per_method_b1+b4"] = per_method(cells, ["b1", "b4"])
    R["D2_per_method_btc"] = per_method(cells, ["btc"])
    R["D3_dose_response"] = dose_response(cells, ["b1", "b4", "btc"])
    # D5: per-dataset x family pooled/partial (sign table)
    d5 = {}
    for (ds, fm), rr in cells.items():
        d5[f"{ds}|{fm}"] = dict(n=len(rr), pooled=boot(rr, lambda r: pooled(r, "lga"), B=500),
                                within=boot(rr, lambda r: within(r, "lga"), B=500),
                                partial=boot(rr, lambda r: partial(r, "lga"), B=500))
    R["D5_cells"] = d5
    (OUT / "lga_sign.json").write_text(json.dumps(R, indent=1))
    try:                                   # figure: the dose-mediation prediction (D7), one point per method cell
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mc = {"neg_grad": "#4c72b0", "rmu": "#c44e52", "fisher": "#8172b2", "leace": "#937860", "inlp": "#da8bc3"}
        mk = {"b1": "o", "b4": "^", "btc": "s"}
        sm = R["D7_all three"]["summary"]
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        for c in R["D7_all three"]["cells"]:
            ax.scatter(c["predicted"], c["obs"], s=46, c=mc.get(c["method"], "#888"), marker=mk[c["fam"]], alpha=.85,
                       edgecolor="k", linewidth=.4)
        ax.plot([-1, 1], [-1, 1], c="#999", lw=.8, ls="--"); ax.axhline(0, c="#ddd", lw=.8); ax.axvline(0, c="#ddd", lw=.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xlabel(r"predicted:  $\rho$(LGA, dose) $\times$ $\rho$(dose, attack)")
        ax.set_ylabel(r"observed:  $\rho$(LGA, attack@r)")
        ax.set_title(f"LGA's sign follows dose (Spearman {sm['spearman_obs_vs_pred']:+.2f}, {sm['n_cells']} method cells)", fontsize=10)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", c=v, label=k) for k, v in mc.items()] + \
            [Line2D([], [], marker=m, ls="", c="#555", label=k) for k, m in mk.items()]
        ax.legend(handles=h, fontsize=7, frameon=False, ncol=2, loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / "lga_sign_mediation.png", dpi=150)
    except Exception as e:
        print("figure skipped:", e)

    L = ["# LGA sign analysis (generated by scripts/analyze_lga_sign.py)\n",
         "Spearman-type rho(LGA, attack@r); negative = the hoped-for direction. 95% bootstrap CI.\n",
         "## D1/D2/D5 - decomposition by family\n",
         "| family | n | pooled | within-method | between-method | partial (dose) |", "|---|---|---|---|---|---|"]
    for k, v in R.items():
        if k.startswith("D1_"):
            L.append(f"| {k[3:]} | {v['pooled']['n']} | {f(v['pooled'])} | {f(v['within'])} | {f(v['between'])} | {f(v['partial_dose'])} |")
    L += ["\n## D7 - does dose mediate?  observed rho(LGA,attack) vs rho(LGA,dose)*rho(dose,attack), per method cell\n",
          "| set | cells | Spearman(obs, predicted) [95% CI over cells] | sign agreement (both abs>.25) | sign agreement (all) |", "|---|---|---|---|---|"]
    for k, v in R.items():
        if k.startswith("D7_"):
            sm = v["summary"]
            L.append(f"| {k[3:]} | {sm['n_cells']} | {sm['spearman_obs_vs_pred']:+.2f} [{sm['ci'][0]:+.2f},{sm['ci'][1]:+.2f}] | {sm['sign_agree_strong']} | {sm['sign_agree_all']} |")
    L += ["\n## D4 - in-budget models only\n", "| family | removed | pooled | within | partial |", "|---|---|---|---|---|"]
    for k, v in R.items():
        if k.startswith("D4_"):
            L.append(f"| {k[12:]} | {v['n_removed']} | {f(v['pooled'])} | {f(v['within'])} | {f(v['partial_dose'])} |")
    L += ["\n## D6 - components (b1+b4)\n", "| component | pooled | within |", "|---|---|---|"]
    for k, v in R.items():
        if k.startswith("D6_"):
            L.append(f"| {k[3:]} | {f(v['pooled'])} | {f(v['within'])} |")
    for fam, key in (("b1+b4", "D2_per_method_b1+b4"), ("btc", "D2_per_method_btc")):
        L += [f"\n## D2 - per method ({fam}): pooled vs partial(dose)\n", "| method | pooled | partial |", "|---|---|---|"]
        for m, v in R[key].items():
            L.append(f"| {m} | {f(v['pooled'])} | {f(v['partial_dose'])} |")
    L += ["\n## D3 - dose-response per cell (dose = retain-val-loss rank)\n",
          "| dataset | fam | method | n | rho(dose,LGA) | rho(dose,attack) | R2 lin | R2 quad | gain | extremum@dose | shape |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R["D3_dose_response"]:
        L.append(f"| {r['ds']} | {r['fam']} | {r['method']} | {r['n']} | {r['rho_dose_lga']:+.2f} | {r['rho_dose_attack']:+.2f} | "
                 f"{r['r2_lin']:.2f} | {r['r2_quad']:.2f} | {r['quad_gain']:+.2f} | {r['extremum_dose']} | {r['curvature']} |")
    L += ["\n## D5 - per dataset x family\n", "| cell | n | pooled | within | partial |", "|---|---|---|---|---|"]
    for k, v in R["D5_cells"].items():
        L.append(f"| {k} | {v['n']} | {f(v['pooled'])} | {f(v['within'])} | {f(v['partial'])} |")
    (OUT / "lga_sign.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))
    print(f"-> {OUT/'lga_sign.md'}")


if __name__ == "__main__":
    main()
