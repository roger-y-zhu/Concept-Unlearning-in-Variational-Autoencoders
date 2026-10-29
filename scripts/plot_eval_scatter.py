"""
scripts/plot_eval_scatter.py
============================
Scatter version of the eval summary. 2 x 3 grid:

    columns -- the 3 eval measures (Geometry / TOFU / U-LiRA)
    rows    -- top: naive question (linear probe), bottom: active attack

Each point is one operating point (every dataset x every unlearning method).
x = the eval measure's score, y = the question's success value. Each panel is
annotated with the pooled Spearman rho (rank within dataset, then pooled) +/- 1
bootstrap std -- the same numbers as the bar summary.

Run:  python scripts/plot_eval_scatter.py
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
OUT = ROOT / "results" / "figures" / "synthesis" / "eval_scatter.png"
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga

DATASETS = ["dsprites", "celeba", "mpi3d"]
MEASURES = ["lga", "tofu_ks", "ulira_leak"]
MEASURE_TITLES = ["LGA (mine)", "TOFU", "U-LiRA"]
MEASURE_XLABELS = ["LGA (baseline->oracle progress)", "TOFU KS-to-oracle", "U-LiRA leak"]
ROWS = [  # (json target, y-label)
    ("linear_auroc", "naïve success\n(linear-probe AUROC)"),
    ("attack_at_r",  "active success\n(attack @ r)"),
]
DATASET_COLOURS = {"dsprites": "tab:blue", "celeba": "tab:orange", "mpi3d": "tab:green"}
N_BOOTSTRAP = 800
SEED = 0
rng = np.random.default_rng(SEED)


def add_geo_suite(data):
    """progress-to-oracle score = the proposed geometry SUITE as one number:
    within each dataset, rank-normalise CKA-to-oracle and the Mahalanobis
    oracle-gap to [0,1], orient both so higher = more oracle-like (CKA as-is,
    gap inverted), and average. Injected as row['geo_suite']."""
    for ds in DATASETS:
        inject_lga(data[ds])   # injects row['lga']: baseline->oracle progress (ledger D2)


def pooled_spearman(data, target, metric, resample=False):
    xr, yr = [], []
    for ds in DATASETS:
        rows = [r for r in data[ds]
                if r.get(metric) is not None and r.get(target) is not None]
        n = len(rows)
        if n < 3:
            continue
        x = np.array([r[metric] for r in rows]); y = np.array([r[target] for r in rows])
        idx = rng.integers(0, n, n) if resample else np.arange(n)
        xr += list(rankdata(x[idx])); yr += list(rankdata(y[idx]))
    if len(xr) < 3 or np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return pearsonr(xr, yr)[0]


def rho_pm(data, target, metric, B=N_BOOTSTRAP):
    p = pooled_spearman(data, target, metric)
    boot = [b for b in (pooled_spearman(data, target, metric, True) for _ in range(B))
            if not np.isnan(b)]
    return p, float(np.std(boot))


def main():
    data = {ds: json.load(open(ROOT / "results" / ds / "dense_recovery.json"))
            for ds in DATASETS}
    add_geo_suite(data)
    allrows = [r for ds in DATASETS for r in data[ds]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex="col")
    for ri, (target, ylab) in enumerate(ROWS):
        for ci, metric in enumerate(MEASURES):
            ax = axes[ri, ci]
            for ds in DATASETS:
                rows = [r for r in data[ds]
                        if r.get(metric) is not None and r.get(target) is not None]
                x = np.array([r[metric] for r in rows])
                y = np.array([r[target] for r in rows])
                if metric == "lga":
                    x = rankdata(x) / len(x)   # unbounded/per-dataset-scaled -> rank for display
                ax.scatter(x, y, s=18, alpha=0.55, color=DATASET_COLOURS[ds],
                           edgecolor="none", label=ds)
            # annotate pooled rho
            rho, sd = rho_pm(data, target, metric)
            ax.text(0.04, 0.94, f"ρ = {rho:+.2f} ± {sd:.2f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, color="0.9", lw=0.7)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(MEASURE_TITLES[ci], fontsize=11)
            if ri == 1:
                ax.set_xlabel(f"{MEASURE_XLABELS[ci]} score")
            if ci == 0:
                ax.set_ylabel(ylab)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:len(DATASETS)], labels[:len(DATASETS)], title="dataset",
               loc="lower center", ncol=len(DATASETS), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Eval measure score vs question success — every operating point "
                 f"(all datasets × unlearning methods, n={len(allrows)})", fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120)
    print("saved", OUT)


if __name__ == "__main__":
    main()
