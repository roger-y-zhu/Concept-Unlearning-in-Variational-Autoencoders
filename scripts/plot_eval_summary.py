"""
scripts/plot_eval_summary.py
============================
Summary bar plot. Left = naive question (linear probe, passive decodability),
right = active question (latent attack, generative reachability). For each of
the three eval measures (Geometry / TOFU / U-LiRA), one bar = the pooled
Spearman rho over ALL operating points (every dataset x every unlearning
method) between that measure and the question's success.

Produces TWO versions:
  eval_summary_no_error.png       -- bars only
  eval_summary_condition_std.png  -- +/- 1 std of the per-(dataset x method)
                                     correlations, i.e. how much the
                                     relationship varies across conditions.
                                     (Deliberately NOT bootstrap std, which is
                                     far smaller and understates real spread.)

Run:  python scripts/plot_eval_summary.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures" / "synthesis"
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga

DATASETS = ["dsprites", "celeba", "mpi3d"]
MEASURES = ["lga", "tofu_ks", "ulira_leak"]
MEASURE_LABELS = ["LGA\n(mine)", "TOFU", "U-LiRA"]
QUESTIONS = [  # (json target, panel title, colour)
    ("linear_auroc", "Linear probe: passive decodability",     "steelblue"),
    ("attack_at_r",  "Active attack: generative reachability", "indianred"),
]


def add_geo_suite(data):
    """progress-to-oracle score = the proposed geometry SUITE as one number:
    within each dataset, rank-normalise CKA-to-oracle and the Mahalanobis
    oracle-gap to [0,1], orient both so higher = more oracle-like, average.
    Injected as row['geo_suite']."""
    for ds in DATASETS:
        inject_lga(data[ds])   # injects row['lga']: baseline->oracle progress (ledger D2)


def spearman_rows(rows, target, metric):
    """Plain Spearman over one flat set of rows (one condition)."""
    rows = [r for r in rows if r.get(metric) is not None and r.get(target) is not None]
    n = len(rows)
    if n < 3:
        return np.nan
    x = np.array([r[metric] for r in rows]); y = np.array([r[target] for r in rows])
    xr, yr = rankdata(x), rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return pearsonr(xr, yr)[0]


def pooled_spearman(data, target, metric):
    """Bar height: rank within each dataset, then pool across datasets."""
    xr, yr = [], []
    for ds in DATASETS:
        rows = [r for r in data[ds]
                if r.get(metric) is not None and r.get(target) is not None]
        if len(rows) < 3:
            continue
        x = np.array([r[metric] for r in rows]); y = np.array([r[target] for r in rows])
        xr += list(rankdata(x)); yr += list(rankdata(y))
    if len(xr) < 3 or np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return pearsonr(xr, yr)[0]


def condition_std(data, target, metric):
    """+/- error: std of the correlation across every (dataset x method) cell
    that has enough points -- captures method-to-method and dataset-to-dataset
    variance together."""
    corrs = []
    for ds in DATASETS:
        for m in sorted({r.get("method") for r in data[ds]}):
            c = spearman_rows([r for r in data[ds] if r.get("method") == m],
                              target, metric)
            if not np.isnan(c):
                corrs.append(c)
    return float(np.std(corrs)) if len(corrs) >= 2 else 0.0


def plot(data, n_total, with_error):
    x = np.arange(len(MEASURES))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (target, title, colour) in zip(axes, QUESTIONS):
        pts = [pooled_spearman(data, target, m) for m in MEASURES]
        err = [condition_std(data, target, m) for m in MEASURES] if with_error else None
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.85", lw=0.8)
        ax.bar(x, pts, 0.6, yerr=err, capsize=4, color=colour)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(MEASURE_LABELS)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("predictive power\n(pooled Spearman ρ with the attack's success)")
    note = ("\n±1 std across dataset × unlearning-method conditions"
            if with_error else "")
    fig.suptitle("How well do the 3 eval measures predict the naïve vs the active question?\n"
                 f"pooled over all datasets × unlearning methods (n={n_total})" + note,
                 fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / ("eval_summary_condition_std.png" if with_error
                     else "eval_summary_no_error.png")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
    print("saved", out)


def main():
    data = {ds: json.load(open(ROOT / "results" / ds / "dense_recovery.json"))
            for ds in DATASETS}
    add_geo_suite(data)
    n_total = sum(len(v) for v in data.values())
    plot(data, n_total, with_error=False)
    plot(data, n_total, with_error=True)


if __name__ == "__main__":
    main()
