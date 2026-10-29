"""
scripts/plot_rundown_figures.py
================================
Figures for docs/THESIS_RUNDOWN.md.

These are decision-support figures, not thesis figures. The rundown argues three
things and each gets one plot:

  fig 1  the provable dissociation (E1/E7) -- two axes, one dataset, one sweep
  fig 2  the headline comparison with model-level error bars (E13)
  fig 3  effort already spent vs. the weight the rundown recommends giving it

Fig 3 in particular is the one that makes the cut argument concrete rather than
assertive: RMU accounts for 64% of every training run in the project and its
headline finding was retracted (R1). That is hard to argue with once it is
drawn, and hard to see in prose.

Everything reads from files already on disk -- no recomputation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402

FIG = ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#222222"
ACCENT = "#b2182b"
COOL = "#2166ac"
GREEN = "#1b7837"


# ---------------------------------------------------------------------------
# fig 1 -- the provable dissociation
# ---------------------------------------------------------------------------

def dose_metrics() -> list:
    """[(a, linear_auroc, dir_norm)] from the dSprites LEACE dose sweep."""
    d = ROOT / "results/dsprites/dsprites_leace"
    rows = []
    for r in sorted(d.iterdir()):
        if not r.is_dir():
            continue
        m = re.match(r"leace_ep1_a([\d.]+)$", r.name)
        a = float(m.group(1)) if m else (1.0 if r.name == "leace_ep1" else None)
        if a is None:
            continue
        cands = list((r / "cache").glob("bespoke_metrics_*.json"))
        cands.append(r / "cache" / "bespoke_metrics.json")
        for f in cands:
            if f.exists():
                j = json.loads(f.read_text())
                rows.append((a, j["linear_auroc"], j["dir_norm"]))
                break
    return sorted(rows)


def fig_dissociation() -> None:
    rows = dose_metrics()
    a = np.array([r[0] for r in rows])
    auroc = np.array([r[1] for r in rows])
    dnorm = np.array([r[2] for r in rows])

    ds_att = json.loads((ROOT / "results/dsprites/attack_robustness_dose.json").read_text())
    ce_att = json.loads((ROOT / "results/celeba/attack_robustness_dose.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- panel A: what the latent metrics say
    ax = axes[0]
    ax.plot(a, auroc, "o-", color=COOL, lw=2, ms=7, label="linear probe AUROC")
    ax.axhline(0.5, color="#999999", ls=":", lw=1)
    ax.annotate("chance", (0.02, 0.5), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color="#777777")
    ax.set_ylim(0.45, 0.70)
    ax.set_xlabel("erasure dose  $a$")
    ax.set_ylabel("linear probe AUROC", color=COOL)
    ax.tick_params(axis="y", labelcolor=COOL)

    ax2 = ax.twinx()
    ax2.plot(a, dnorm, "s--", color=ACCENT, lw=2, ms=6, label="direction norm")
    ax2.set_ylabel("concept direction norm", color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)

    ax.set_title("A. dSprites: the two latent metrics disagree with each other\n"
                 "AUROC flat at 0.6307 for every $a<1$; direction norm falls 21x",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.annotate("$\\det(P_a)=(1-a)$: invertible for every $a<1$,\n"
                "so linear separability CANNOT change",
                xy=(0.5, 0.6307), xytext=(0.14, 0.545), fontsize=8.5, color=COOL,
                arrowprops=dict(arrowstyle="->", color=COOL, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COOL, alpha=0.9))

    # --- panel B: what an active attack says
    ax = axes[1]
    for name, att, colour, marker in (("dSprites (r=2)", ds_att, COOL, "o"),
                                      ("CelebA (r=4)", ce_att, ACCENT, "s")):
        xs = np.array(sorted(float(k) for k in att))
        ys = np.array([att[f"{x:g}"] if f"{x:g}" in att else att[str(x)] for x in xs])
        ax.plot(xs, ys, marker=marker, ls="-", color=colour, lw=2, ms=6, label=name)
    ax.set_xlabel("erasure dose  $a$")
    ax.set_ylabel("adversarial attack success  (higher = easier to attack)")
    ax.set_title("B. The same sweep, judged by an active attack\n"
                 "rises monotonically as the latent metric stays flat",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    fig.suptitle("The dissociation is provable, not statistical  "
                 "(RESEARCH_LEDGER E1 / E7)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "rundown_fig1_dissociation.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"-> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# fig 2 -- headline comparison, with error bars
# ---------------------------------------------------------------------------

def fig_attack_ordering() -> None:
    tables = json.loads((ROOT / "results/master_table.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for ax, ds in zip(axes, ("dsprites", "celeba")):
        t = tables[ds]
        pts = []
        for row in t["rows"]:
            agg = row.get("agg", {})
            if "adv_at_r" not in agg:
                continue
            mean, sd, n = agg["adv_at_r"]
            if mean is None:
                continue
            pts.append((row["method"], mean, sd, n))
        pts.sort(key=lambda p: p[1])

        names = [p[0] for p in pts]
        means = np.array([p[1] for p in pts])
        sds = np.array([p[2] or 0.0 for p in pts])
        ns = [p[3] for p in pts]
        y = np.arange(len(pts))

        colours = []
        for nm in names:
            if nm == "baseline":
                colours.append(INK)
            elif nm == "oracle":
                colours.append("#7a7a7a")
            elif nm in ("leace", "inlp"):
                colours.append(ACCENT)
            elif nm == "feature_unlearn":
                colours.append(GREEN)
            else:
                colours.append(COOL)

        ax.barh(y, means, xerr=sds, color=colours, alpha=0.85,
                error_kw=dict(ecolor="#333333", lw=1.3, capsize=3))
        base = dict(zip(names, means)).get("baseline")
        if base is not None:
            ax.axvline(base, color=INK, ls=":", lw=1.4)
            ax.annotate("baseline", (base, len(pts) - 0.4), rotation=90,
                        fontsize=8, color=INK, ha="right", va="top")

        ax.set_yticks(y)
        ax.set_yticklabels([f"{nm}  (n={n})" for nm, n in zip(names, ns)], fontsize=9)
        ax.set_xlabel(f"consensus attack success @ r={t['adv_r']}   "
                      f"(lower = more resistant)")
        ax.set_title(f"{ds}", fontsize=11)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Attack resistance at matched retain budget, with model-level error bars\n"
                 "red = closed-form erasure (best latent metrics, worst resistance)  ·  "
                 "green = frozen encoder", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG / "rundown_fig2_attack_ordering.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"-> {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# fig 3 -- effort spent vs. recommended thesis weight
# ---------------------------------------------------------------------------

#: The rundown's recommendation (section 3/4). A judgement, presented as one.
WEIGHT = {
    "rmu":             ("compress to a section", ACCENT),
    "feature_unlearn": ("Tier 1 — central", GREEN),
    "fisher":          ("compress to a section", ACCENT),
    "leace":           ("Tier 1 — central", GREEN),
    "neg_grad":        ("Tier 3 — keep, brief", COOL),
    "inlp":            ("Tier 1 — central", GREEN),
    "finetune":        ("Tier 3 — keep, brief", COOL),
}

def fig_effort() -> None:
    counts = {}
    for ds in ("dsprites", "celeba"):
        root = ROOT / "results" / ds
        if not root.is_dir():
            continue
        for sweep in sorted(root.iterdir()):
            if not sweep.is_dir() or "PREBNFIX" in sweep.name:
                continue
            m = sweep.name.replace(f"{ds}_", "")
            if m not in WEIGHT:
                continue
            n = sum(1 for r in sweep.iterdir()
                    if r.is_dir() and (r / "history.json").exists())
            counts[m] = counts.get(m, 0) + n

    items = sorted(counts.items(), key=lambda t: t[1])
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items])
    total = int(vals.sum())
    colours = [WEIGHT[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colours, alpha=0.88)

    # xlim from the DATA, with headroom for the labels. An earlier version set
    # it to 62% of the TOTAL, which was less than rmu's own 64% share -- so the
    # largest bar was clipped at the axis edge and its annotation landed off
    # canvas entirely. Scale to the largest bar, never to the total.
    ax.set_xlim(0, vals.max() * 1.28)
    for i, (n, v) in enumerate(items):
        ax.text(v + vals.max() * 0.015, i, f"{v}  ({v / total:.0%})",
                va="center", fontsize=9.5, color="#333333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([n + "\n" + WEIGHT[n][0] for n in names], fontsize=9)
    ax.set_xlabel(f"training runs on disk  (total {total} across both datasets)")
    ax.set_title(
        "Effort already spent vs. the weight the rundown recommends\n"
        "green = Tier 1, central   ·   blue = keep but brief   ·   red = compress",
        fontsize=11.5)
    ax.grid(axis="x", alpha=0.25)

    ax.annotate(
        "RMU is 64% of every training run in the project\n"
        "and its headline finding was RETRACTED (R1).\n"
        "The single biggest available trim.",
        xy=(vals[-1] * 0.70, len(names) - 1),
        xytext=(vals.max() * 0.34, len(names) - 3.6),
        fontsize=9.5, color=ACCENT,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6),
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=ACCENT, alpha=0.97),
    )

    fig.tight_layout()
    out = FIG / "rundown_fig3_effort_vs_weight.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    fig_dissociation()
    fig_attack_ordering()
    fig_effort()
