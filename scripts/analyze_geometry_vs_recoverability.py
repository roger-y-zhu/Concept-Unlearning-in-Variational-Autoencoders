"""Cross-dataset synthesis: does latent-geometry change predict attack recoverability?

The thesis' central claim, quantified. For each dataset we have, per champion, four
latent-geometry metrics (linear/mlp AUROC, kNN cohesion, dir_norm) and the
ball-constrained attack success at that dataset's max-separation radius (lower =
more resistant = less recoverable). If geometry predicted recoverability, a latent
metric would correlate with attack success with a consistent sign across datasets.

Outputs a per-dataset and pooled Spearman table + a scatter figure
(results/figures/synthesis/geometry_vs_recoverability.png). Reads results/master_table.json
(all three datasets), so it reflects the error-bar-validated champions.
"""
from __future__ import annotations
import json
from pathlib import Path
from scipy.stats import spearmanr, rankdata
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results/master_table.json").read_text())
DATASETS = ["dsprites", "celeba", "mpi3d"]
METRICS = [("linear_auroc", "Linear AUROC"), ("mlp_auroc", "MLP AUROC"),
           ("knn_cohesion", "kNN cohesion"), ("dir_norm", "dir_norm")]
COLORS = {"dsprites": "#2196F3", "celeba": "#F44336", "mpi3d": "#4CAF50"}


def val(row, k):
    a = row["agg"].get(k)
    return a[0] if a else None


# Collect per-dataset method-level points (exclude baseline/oracle references).
rows = {}
for ds in DATASETS:
    tab = d[ds]
    pts = []
    for r in tab["rows"]:
        if r["method"] in ("baseline", "oracle"):
            continue
        at = val(r, "adv_at_r")
        if at is None:
            continue
        rec = {"method": r["method"], "attack": at}
        for mk, _ in METRICS:
            rec[mk] = val(r, mk)
        pts.append(rec)
    rows[ds] = pts
    print(f"{ds}: {len(pts)} champions (adv_r={tab.get('adv_r')})")

print("\n=== Per-dataset Spearman(metric, attack) — sign should be inconsistent if geometry doesn't predict recoverability ===")
print(f"{'metric':<16}" + "".join(f"{ds:>14}" for ds in DATASETS) + f"{'pooled(rank)':>16}")
pooled_axis = {}
for mk, name in METRICS:
    line = f"{name:<16}"
    pooled_x, pooled_y = [], []
    for ds in DATASETS:
        xs = [p[mk] for p in rows[ds] if p[mk] is not None]
        ys = [p["attack"] for p in rows[ds] if p[mk] is not None]
        if len(set(xs)) < 2 or len(xs) < 3:
            line += f"{'n/a':>14}"          # e.g. AUROC saturated on mpi3d
        else:
            rho = spearmanr(xs, ys).statistic
            line += f"{rho:>+14.2f}"
        # pool by within-dataset rank (removes cross-dataset scale/radius differences)
        if len(set(xs)) >= 2 and len(xs) >= 3:
            pooled_x += list(rankdata(xs) / len(xs))
            pooled_y += list(rankdata(ys) / len(ys))
    if len(pooled_x) >= 4:
        prho = spearmanr(pooled_x, pooled_y).statistic
        line += f"{prho:>+16.2f}"
        pooled_axis[mk] = (pooled_x, pooled_y, prho)
    else:
        line += f"{'n/a':>16}"
    print(line)

# ── Scatter figure: attack vs each latent metric, per dataset ────────────────
fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4))
for ax, (mk, name) in zip(axes, METRICS):
    for ds in DATASETS:
        xs = [p[mk] for p in rows[ds] if p[mk] is not None]
        ys = [p["attack"] for p in rows[ds] if p[mk] is not None]
        ax.scatter(xs, ys, color=COLORS[ds], label=ds, s=45, alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.set_xlabel(name); ax.set_ylabel("attack success (lower = more resistant)")
    ax.set_title(name); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, title="dataset")
fig.suptitle("Geometry vs recoverability across datasets — no consistent relationship\n"
             "(each point = one method's champion; y = attack success at the dataset's max-separation radius)",
             fontsize=11)
fig.tight_layout()
out = ROOT / "results/figures/synthesis/geometry_vs_recoverability.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n-> {out.relative_to(ROOT)}")
(ROOT / "results/synthesis_geometry_vs_recoverability.json").write_text(
    json.dumps({ds: rows[ds] for ds in DATASETS}, indent=2))
