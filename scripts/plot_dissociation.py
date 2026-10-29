"""
scripts/plot_dissociation.py
=============================
The one figure that is the thesis: latent metric on x, attack success on y, one
point per method, both datasets.

WHY THIS FIGURE
---------------
The project's central claim (RESEARCH_LEDGER E4, E13) is that latent-geometry
metrics do not predict resistance to an active attack -- and that at the
extremes the two are anti-correlated. That claim currently lives in a nine-row
table per dataset, where a reader has to hold two columns in their head and
notice that they run in opposite directions.

Plotting them against each other makes it a shape rather than an inference. It
also makes the two anchor cases visible at a glance:

  * `feature_unlearn` sits at the baseline's exact x (its encoder is frozen by
    construction, so its latent metrics are bit-identical) but far below it in
    y. A vertical drop with zero horizontal movement is the cleanest possible
    statement of "these axes are independent".
  * `leace` / `inlp` sit at the far LEFT (best latent metrics in the study --
    LEACE's linear AUROC is below chance) and near the TOP (most attackable).

Error bars are drawn on both axes where replicates exist (n=5 training seeds
per champion; n=4 for the post-hoc erasure methods after refitting them on the
queue28 baseline replicates). Points without replicates are drawn hollow and
labelled, rather than being silently indistinguishable from replicated ones.

Spearman rho is reported per dataset over the METHODS only -- baseline and
oracle are reference points, not methods, and including them would let the
reference dominate a 7-point correlation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from scipy.stats import spearmanr         # noqa: E402

OUT = ROOT / "results/figures/dissociation.png"

#: x-axis choices. linear_auroc is the headline latent metric; the others are
#: available for the appendix version of this figure.
X_METRIC = "linear_auroc"
X_LABEL = "Linear probe AUROC on the latent  (lower = concept less decodable)"

REFERENCES = {"baseline", "oracle"}

STYLE = {
    "baseline":        dict(color="#333333", marker="s", label="baseline"),
    "oracle":          dict(color="#7a7a7a", marker="D", label="oracle (retrain-free)"),
    "feature_unlearn": dict(color="#1b7837", marker="o", label="feature_unlearn"),
    "neg_grad":        dict(color="#4393c3", marker="o", label="neg_grad"),
    "finetune":        dict(color="#92c5de", marker="o", label="finetune"),
    "fisher":          dict(color="#f4a582", marker="o", label="fisher"),
    "rmu":             dict(color="#d6604d", marker="o", label="rmu"),
    "leace":           dict(color="#b2182b", marker="^", label="leace"),
    "inlp":            dict(color="#67001f", marker="v", label="inlp"),
}


def load_tables() -> dict:
    p = ROOT / "results/master_table.json"
    if not p.exists():
        raise SystemExit("results/master_table.json missing -- run build_master_table.py first")
    return json.loads(p.read_text())


def points(table: dict) -> list:
    """[(method, x, xsd, xn, y, ysd, yn)] for rows that have both axes."""
    out = []
    for row in table["rows"]:
        agg = row.get("agg", {})
        if X_METRIC not in agg or "adv_at_r" not in agg:
            continue
        x, xsd, xn = agg[X_METRIC]
        y, ysd, yn = agg["adv_at_r"]
        if x is None or y is None:
            continue
        out.append((row["method"], x, xsd, xn, y, ysd, yn))
    return out


def main() -> None:
    tables = load_tables()
    datasets = [d for d in ("dsprites", "celeba") if d in tables]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2 * len(datasets), 6.0),
                             squeeze=False)

    for ax, ds in zip(axes[0], datasets):
        table = tables[ds]
        pts = points(table)
        r = table.get("adv_r")

        for method, x, xsd, xn, y, ysd, yn in pts:
            st = STYLE.get(method, dict(color="#888888", marker="o", label=method))
            replicated = (yn or 1) > 1
            ax.errorbar(
                x, y,
                xerr=(xsd if (xn or 1) > 1 else None),
                yerr=(ysd if replicated else None),
                color=st["color"], marker=st["marker"],
                markersize=11 if method not in REFERENCES else 12,
                markerfacecolor=st["color"] if replicated else "none",
                markeredgewidth=1.8, capsize=3, elinewidth=1.4,
                linestyle="none", zorder=3,
            )
            ax.annotate(
                st["label"], (x, y), textcoords="offset points",
                xytext=(9, 5), fontsize=8.5, color=st["color"],
            )

        # correlation over METHODS only -- references are not methods
        m = [(x, y) for meth, x, _, _, y, _, _ in pts if meth not in REFERENCES]
        if len(m) >= 3:
            rho, p = spearmanr([a for a, _ in m], [b for _, b in m])
            sub = f"Spearman rho = {rho:+.3f}  (p = {p:.3g}, n = {len(m)} methods)"
        else:
            sub = "too few methods for a correlation"

        # The anchor case, drawn explicitly because it is the cleanest logical
        # statement in the project: feature_unlearn's encoder is FROZEN, so its
        # latent metrics are bit-identical to the baseline's -- same x, to full
        # float precision -- while its attack resistance is far better. A purely
        # vertical segment is "these two axes are independent" with no statistics
        # required, and it is the one comparison in the figure that cannot be
        # explained away as a confound.
        bx = next((x for meth, x, _, _, _, _, _ in pts if meth == "baseline"), None)
        by = next((y for meth, _, _, _, y, _, _ in pts if meth == "baseline"), None)
        fx = next((x for meth, x, _, _, _, _, _ in pts if meth == "feature_unlearn"), None)
        fy = next((y for meth, _, _, _, y, _, _ in pts if meth == "feature_unlearn"), None)
        # Tolerance 1e-6, not exact equality. On dSprites the two x-values are
        # bit-identical (delta exactly 0.0). On CelebA they differ by 7.7e-08 --
        # eight significant figures agree, but not the last bits, because the
        # two encodings are accumulated over different batch orders and float
        # addition is not associative. The encoder weights ARE identical (frozen
        # by construction); the residual is summation order, not a real
        # difference. Requiring bit-equality here silently dropped the CelebA
        # arrow, which is the panel where the effect is largest.
        if None not in (bx, by, fx, fy) and abs(bx - fx) < 1e-6:
            ax.annotate(
                "", xy=(fx, fy), xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color="#1b7837", lw=2.0,
                                shrinkA=9, shrinkB=9, alpha=0.85),
                zorder=2,
            )
            # Caption pinned to AXES coordinates, not to the arrow's midpoint.
            # Anchoring it to the data put it straight through the neg_grad
            # marker on CelebA, and every data-relative position that clears one
            # panel collides in the other -- the two panels have different
            # spreads. A fixed corner is always legible and needs no per-panel
            # tuning.
            ax.text(
                0.03, 0.52,
                f"feature_unlearn vs baseline:\n"
                f"identical latent metrics (frozen encoder)\n"
                f"{by / fy:.1f}x more attack-resistant",
                transform=ax.transAxes, fontsize=8.5, color="#1b7837",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1b7837", alpha=0.9),
                zorder=4,
            )

        base = next((y for meth, _, _, _, y, _, _ in pts if meth == "baseline"), None)
        if base is not None:
            ax.axhline(base, color="#333333", lw=0.8, ls=":", zorder=1)
            ax.annotate("baseline attack level", (ax.get_xlim()[0], base),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color="#333333")

        ax.set_title(f"{ds}   (attack radius r = {r})\n{sub}", fontsize=11)
        ax.set_xlabel(X_LABEL, fontsize=9.5)
        ax.set_ylabel(f"consensus attack success @ r={r}   (lower = more resistant)",
                      fontsize=9.5)
        ax.grid(alpha=0.25, zorder=0)

    fig.suptitle(
        "Latent decodability does not predict attack resistance\n"
        "filled = replicated (error bars are 1 sd across seeds); hollow = single run",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"-> {OUT.relative_to(ROOT)}")

    for ds in datasets:
        pts = points(tables[ds])
        m = [(meth, x, y) for meth, x, _, _, y, _, _ in pts if meth not in REFERENCES]
        if len(m) >= 3:
            rho, p = spearmanr([x for _, x, _ in m], [y for _, _, y in m])
            print(f"{ds:<10} n={len(m)} methods   Spearman rho={rho:+.3f}  p={p:.3g}")


if __name__ == "__main__":
    main()
