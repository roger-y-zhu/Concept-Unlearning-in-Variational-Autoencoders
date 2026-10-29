"""
scripts/plot_method_vs_reference_plain.py
==========================================
Plain matplotlib dot plots: for each dataset, one panel per evaluation metric,
showing where every method's champion run landed relative to the untouched
baseline and the retrain-from-scratch oracle. No custom styling beyond what
makes the reference points and error bars readable.

    python scripts/plot_method_vs_reference_plain.py

Reads results/master_table.json directly -- no intermediate cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("linear_auroc", "Linear probe AUROC"),
    ("mlp_auroc", "MLP probe AUROC"),
    ("knn_cohesion", "kNN cohesion"),
    ("dir_norm", "Direction norm"),
    ("adv_at_r", "Attack success @ r"),
]
METHOD_ORDER = ["neg_grad", "finetune", "fisher", "rmu", "leace", "inlp", "feature_unlearn"]
LABELS = {
    "neg_grad": "NegGrad+", "finetune": "Finetune", "fisher": "Fisher scrubbing",
    "rmu": "RMU", "leace": "LEACE", "inlp": "INLP", "feature_unlearn": "Feature Unlearning",
}

ROW_H = 0.42          # inches per method row -- enough that adjacent dots never touch
HEADER_H = 0.55        # inches reserved for each panel's title + top margin


def plot_dataset(table: dict, ds: str) -> Path:
    rows_by_method = {r["method"]: r for r in table["rows"]}
    base_agg = rows_by_method["baseline"]["agg"]
    orac_agg = rows_by_method["oracle"]["agg"]

    n_rows = len(METHOD_ORDER)
    panel_h = HEADER_H + ROW_H * n_rows
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(9.5, panel_h * len(METRICS)),
        gridspec_kw={"hspace": 0.75, "left": 0.16, "right": 0.97, "top": 0.965, "bottom": 0.02},
    )
    fig.suptitle(f"{ds} -- method vs. baseline vs. oracle  (attack radius r={table['adv_r']})",
                 fontsize=13, y=0.995)

    ys = list(range(n_rows))
    for ax, (key, label) in zip(axes, METRICS):
        b = base_agg.get(key, [None])[0]
        o = orac_agg.get(key, [None])[0]
        ax.axvline(b, color="black", lw=1.5, label="baseline")
        ax.axvline(o, color="gray", lw=1.5, ls="--", label="oracle")

        labels = []
        for y, m in zip(ys, METHOD_ORDER):
            labels.append(LABELS[m])
            row = rows_by_method.get(m)
            agg = row["agg"] if row else {}
            if not agg or key not in agg:
                ax.text((b + o) / 2, y, "no in-budget champion", va="center", ha="center",
                         fontsize=8, color="gray", style="italic")
                continue
            v, sd, n = agg[key]
            ax.errorbar(v, y, xerr=(sd or 0), fmt="o", color="C0", ecolor="C0",
                         capsize=3, ms=6, zorder=5)

        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_title(label, fontsize=10.5, loc="left")
        ax.grid(axis="x", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    out = OUT / f"{ds}_method_vs_reference.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    table = json.loads((ROOT / "results/master_table.json").read_text())
    # Plot whichever datasets the table actually contains, in a stable order, so
    # a newly-added dataset (mpi3d) appears without editing this list.
    order = ["dsprites", "celeba", "mpi3d"]
    for ds in [d for d in order if d in table] + [d for d in table if d not in order]:
        out = plot_dataset(table[ds], ds)
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
