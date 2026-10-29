"""
scripts/plot_passive_vs_active.py
=================================
Two-panel bar chart: each of three unlearning-evaluation measures
(my oracle-geometry suite, TOFU, U-LiRA) correlated with

    left  -- the linear probe        (passive decodability)
    right -- the active latent attack (generative reachability)

sliced by dataset (coloured bars) and aggregated over all 7 unlearning
methods. Correlation = pooled Spearman rho; error bars = +/- 1 bootstrap std.

Reads results/<dataset>/dense_recovery.json (produced by
scripts/run_dense_recovery.py). Writes the PNG into results/figures/synthesis/.

Run:  python scripts/plot_passive_vs_active.py
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

# --- knobs you'll likely want to tweak -------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "synthesis" / "passive_vs_active_by_dataset.png"
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga

DATASETS = ["dsprites", "celeba", "mpi3d"]
MEASURES = ["lga", "tofu_ks", "ulira_leak"]        # the json fields
MEASURE_LABELS = ["LGA\n(mine)", "TOFU", "U-LiRA"]
GROUP_SPACING = 1.6           # >1 widens the gap between measure groups
N_BOOTSTRAP = 800
SEED = 0

# (json field for the target, y-axis label, panel title)
PANELS = [
    ("linear_auroc",
     "Spearman rho with naive probe\n(linear-probe AUROC)",
     "Linear probe: passive decodability"),
    ("attack_at_r",
     "Spearman rho with active attack",
     "Active attack: generative reachability"),
]
SUPTITLE = ("three unlearning-eval measures' correlation with naive probe "
            "and active membership attack")
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)


def add_geo_suite(data):
    """Geometry SUITE (ledger D2) as one score: within each dataset, rank-normalise
    CKA-to-oracle and the Mahalanobis oracle-gap to [0,1], orient both so higher =
    more oracle-like (CKA as-is, gap inverted), average. Injected as row['geo_suite']."""
    for ds in DATASETS:
        inject_lga(data[ds])   # injects row['lga']: baseline->oracle progress (ledger D2)


def spearman(rows, target, metric, resample=False):
    """Spearman rho between one measure and one target over a set of rows."""
    rows = [r for r in rows if r.get(metric) is not None and r.get(target) is not None]
    n = len(rows)
    if n < 3:
        return np.nan
    x = np.array([r[metric] for r in rows])
    y = np.array([r[target] for r in rows])
    idx = rng.integers(0, n, n) if resample else np.arange(n)
    xr, yr = rankdata(x[idx]), rankdata(y[idx])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return pearsonr(xr, yr)[0]


def estimate(rows, target, metric):
    """Point estimate = Spearman over all rows in this dataset (pooled across
    unlearning methods). Error bar = std of the per-method correlations, i.e.
    how much the correlation varies from one unlearning method to another.
    Methods with too few points to correlate are skipped."""
    p = spearman(rows, target, metric)
    if np.isnan(p):
        return np.nan, 0.0
    per_method = []
    for m in sorted({r.get("method") for r in rows}):
        c = spearman([r for r in rows if r.get("method") == m], target, metric)
        if not np.isnan(c):
            per_method.append(c)
    sd = float(np.std(per_method)) if len(per_method) >= 2 else 0.0
    return p, sd


def main():
    data = {ds: json.load(open(ROOT / "results" / ds / "dense_recovery.json"))
            for ds in DATASETS}
    add_geo_suite(data)
    x = np.arange(len(MEASURES)) * GROUP_SPACING
    width = 0.8 / len(DATASETS)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (target, ylab, title) in zip(axes, PANELS):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.85", lw=0.8)
        for ci, ds in enumerate(DATASETS):
            pts = [estimate(data[ds], target, m)[0] for m in MEASURES]
            off = (ci - (len(DATASETS) - 1) / 2) * width
            ax.bar(x + off, [0 if np.isnan(p) else p for p in pts], width, label=ds)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(MEASURE_LABELS)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8, title="dataset")
    fig.suptitle(SUPTITLE)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120)
    print("saved", OUT)


if __name__ == "__main__":
    main()
