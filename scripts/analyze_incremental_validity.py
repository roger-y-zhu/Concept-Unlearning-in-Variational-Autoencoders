"""
scripts/analyze_incremental_validity.py
=======================================
The evaluation-strengthening study's analysis (ledger D1). Answers the
supervisor's three asks from the dense operating-point data
(results/<ds>/dense_recovery.json, produced by run_dense_recovery.py):

  1. COMPARE the proposed metric vs existing metrics  -> correlation matrix.
  2. (SOTA folded in separately: DCI, TOFU forget-quality, U-LiRA.)
  3. Does the PROPOSED metric add information beyond STANDARD metrics?
     Two-sided incremental validity:
       (a) redundancy  -- can the proposed oracle-geometry metric be predicted
           from the standard probe/geometry battery? (low R^2 = a distinct axis)
       (b) validity    -- does it predict the CRITERION (adversarial recovery)
           better than / beyond the standard metrics? (the decisive test)

ROLES (D1): proposed = oracle-referenced geometry (CKA-to-oracle, Mahalanobis
oracle-gap); standard = linear/MLP probe AUROC, kNN cohesion, dir_norm;
criterion = consensus adversarial attack success at each dataset's report radius.

Cross-dataset pooling rank-normalises every column WITHIN each dataset first
(as E17 does), so different latent scales / attack radii cannot drive the pooled
relationship. Writes a report + figures to results/figures/synthesis/.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # small-fold R^2 warnings on partial data; n/a at full n

import numpy as np
from scipy.stats import spearmanr, rankdata, kendalltau
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.lga import inject_lga
DATASETS = ["dsprites", "celeba", "mpi3d"]
COLORS = {"dsprites": "#2196F3", "celeba": "#F44336", "mpi3d": "#4CAF50"}

CRITERION = ("attack_at_r", "Attack success")
PROPOSED = [("cka_oracle", "CKA-to-oracle"), ("oracle_gap", "Maha oracle-gap")]
COMBINED = [("lga", "LGA (proposed)")]      # the single combined score = f(PROPOSED)
SOTA = [("tofu_ks", "TOFU KS-to-oracle")]            # modern oracle-distance criterion
STANDARD = [("linear_auroc", "Linear AUROC"), ("mlp_auroc", "MLP AUROC"),
            ("knn_cohesion", "kNN cohesion"), ("dir_norm", "dir_norm")]
ORACLE_DISTANCE = PROPOSED + SOTA                    # everything of the "match the oracle" family
ALL_METRICS = COMBINED + PROPOSED + SOTA + STANDARD
OUT = ROOT / "results/figures/synthesis"


def load_points(include_refs=False) -> dict:
    """Per-dataset list of operating-point rows (method points; refs optional)."""
    out = {}
    for ds in DATASETS:
        p = ROOT / f"results/{ds}/dense_recovery.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        inject_lga(rows)   # anchored on baseline/oracle rows -> must precede ref drop
        if not include_refs:
            rows = [r for r in rows if r["method"] not in ("baseline", "oracle")]
        # keep rows with a finite criterion and all metrics present
        keep = []
        for r in rows:
            vals = [r.get(k) for k, _ in ALL_METRICS] + [r.get(CRITERION[0])]
            if all(v is not None and np.isfinite(v) for v in vals):
                keep.append(r)
        out[ds] = keep
    return out


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def rank_norm(x):
    return rankdata(x) / len(x)


def boot_spearman(x, y, n_boot=2000, seed=0):
    """Spearman rho with a 95% bootstrap CI (resample operating points with
    replacement). Returns (rho, lo, hi)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    if len(x) < 4:
        return rho, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        stats.append(spearmanr(x[idx], y[idx]).statistic)
    if not stats:
        return rho, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return rho, lo, hi


def pooled_matrix(data):
    """Stack rank-normalised (within dataset) columns for every metric + criterion."""
    keys = [k for k, _ in ALL_METRICS] + [CRITERION[0]]
    cols = {k: [] for k in keys}
    for ds, rows in data.items():
        if len(rows) < 4:
            continue
        for k in keys:
            cols[k] += list(rank_norm(col(rows, k)))
    return {k: np.array(v) for k, v in cols.items()}, keys


# ── 1. Correlation of every metric with the CRITERION, per dataset + pooled ───
def criterion_correlations(data, pooled):
    print("\n" + "=" * 78)
    print("1. Spearman(metric, attack criterion) — PER DATASET (95% bootstrap CI), then pooled")
    print("   (lead per-dataset: pooling sign-inconsistent relationships can manufacture a")
    print("    trend; the pooled column is secondary and uses within-dataset rank-normalisation)")
    print("=" * 78)
    result = {}
    for k, name in ALL_METRICS:
        print(f"\n  {name}")
        for ds in DATASETS:
            rows = data.get(ds, [])
            if len(rows) >= 4 and len(set(col(rows, k))) > 1:
                rho, lo, hi = boot_spearman(col(rows, k), col(rows, CRITERION[0]))
                flag = "" if (lo <= 0 <= hi) else "  *sig (CI excludes 0)*"
                print(f"    {ds:<9} rho = {rho:>+.2f}  [95% CI {lo:>+.2f}, {hi:>+.2f}]  n={len(rows)}{flag}")
            else:
                print(f"    {ds:<9} n/a (constant/too few)")
        prho, plo, phi = boot_spearman(pooled[k], pooled[CRITERION[0]])
        result[k] = prho
        print(f"    {'POOLED':<9} rho = {prho:>+.2f}  [95% CI {plo:>+.2f}, {phi:>+.2f}]  n={len(pooled[k])}")
    print("\nReading: a metric that PREDICTS recoverability would show a large, "
          "consistent-sign\ncorrelation ACROSS datasets with CIs excluding 0. Sign flips "
          "across datasets\n(e.g. oracle-gap) mean the pooled number is not a real effect. The "
          "proposed\noracle-geometry metrics are listed first; the standard probes follow.")
    return result


# ── 2. Full metric×metric correlation heatmap (pooled) ───────────────────────
def correlation_heatmap(pooled, keys):
    labels = [dict(ALL_METRICS + [CRITERION])[k] for k in keys]
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            M[i, j] = spearmanr(pooled[a], pooled[b]).statistic
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.55 else "black", fontsize=7)
    ax.set_title("Pooled Spearman correlation — metrics + attack criterion\n"
                 "(proposed = CKA-to-oracle / oracle-gap; criterion = attack)", fontsize=10)
    # mark the criterion row/col
    ax.add_patch(plt.Rectangle((n - 1 - 0.5, -0.5), 1, n, fill=False, edgecolor="k", lw=2))
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "metric_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\n-> {OUT / 'metric_correlation_heatmap.png'}")


# ── 3. Incremental validity: predict the CRITERION from metric sets ──────────
def incremental_validity(pooled):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold

    y = pooled[CRITERION[0]]
    std_keys = [k for k, _ in STANDARD]
    prop_keys = [k for k, _ in PROPOSED]
    od_keys = [k for k, _ in ORACLE_DISTANCE if k in pooled]   # proposed + TOFU (if present)

    def X(keys):
        return np.column_stack([pooled[k] for k in keys])

    def cv_r2(Xm, model, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        return cross_val_score(model, Xm, y, cv=kf, scoring="r2").mean()

    print("\n" + "=" * 78)
    print("3. Incremental validity — cross-validated R^2 predicting the ATTACK criterion")
    print("=" * 78)
    for label, mk in [("Ridge (linear)", Ridge(alpha=1.0)),
                      ("RandomForest ", RandomForestRegressor(n_estimators=300, random_state=0))]:
        r_std = cv_r2(X(std_keys), mk)
        r_prop = cv_r2(X(prop_keys), mk)
        r_lga = cv_r2(X(["lga"]), mk)
        r_std_lga = cv_r2(X(std_keys + ["lga"]), mk)
        r_od = cv_r2(X(od_keys), mk)
        r_both = cv_r2(X(std_keys + prop_keys), mk)
        r_all = cv_r2(X(std_keys + od_keys), mk)
        print(f"\n  [{label}]  (n={len(y)} pooled operating points)")
        print(f"    standard proxies only                 R^2 = {r_std:+.3f}")
        print(f"    proposed (our oracle-geometry)         R^2 = {r_prop:+.3f}")
        print(f"    LGA alone (single combined score)      R^2 = {r_lga:+.3f}")
        print(f"    oracle-distance family (ours + TOFU)   R^2 = {r_od:+.3f}")
        print(f"    standard + proposed                    R^2 = {r_both:+.3f}")
        print(f"    standard + oracle-distance family      R^2 = {r_all:+.3f}")
        print(f"    -> proposed adds beyond std:  dR^2 = {r_both - r_std:+.3f}")
        print(f"    -> LGA alone adds beyond std: dR^2 = {r_std_lga - r_std:+.3f}")
        print(f"    -> full oracle-distance adds: dR^2 = {r_all - r_std:+.3f}")
        if label.startswith("RandomForest"):
            print(f"    NOTE: RF R^2 on n={len(y)} with 6 features is overfitting-prone even under CV;")
            print("          the Ridge dR^2 (below) is the more honest 'does it add info' number.")

    # Bootstrap CI on the HONEST number: ridge dR^2 (proposed adds beyond standard).
    print("\n  Bootstrap 95% CI on the Ridge incremental dR^2 (proposed beyond standard):")
    rng = np.random.default_rng(0)
    N = len(y)
    def ridge_dr2(idx):
        yy = y[idx]
        def r2(keys):
            Xm = np.column_stack([pooled[k][idx] for k in keys])
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            return cross_val_score(Ridge(alpha=1.0), Xm, yy, cv=kf, scoring="r2").mean()
        return r2(std_keys + prop_keys) - r2(std_keys)
    base = ridge_dr2(np.arange(N))
    boot = [ridge_dr2(rng.integers(0, N, N)) for _ in range(300)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"    dR^2 (proposed) = {base:+.3f}   [95% CI {lo:+.3f}, {hi:+.3f}]  "
          f"{'(CI includes 0 -> not distinguishable from no added info)' if lo <= 0 <= hi else ''}")
    print("\nReading: if EVERY R^2 is near zero, no metric set predicts the attack — the "
          "proposed\nmetric is not a valid proxy for recoverability, and neither are the "
          "standard ones\n(H2, quantified). A large standard-only R^2 with small dR^2 would "
          "instead mean the\nproposed metric is redundant. The dR^2 is the direct answer to "
          "'does it add info?'.")

    # Redundancy: is each oracle-distance metric (ours + SOTA) predictable from standard?
    print("\n  Redundancy — CV R^2 predicting each oracle-distance metric from the standard battery:")
    for k, name in ORACLE_DISTANCE + COMBINED:
        if k not in pooled:
            continue
        yk = pooled[k]
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        r = cross_val_score(Ridge(alpha=1.0), X(std_keys), yk, cv=kf, scoring="r2").mean()
        print(f"    {name:<18} R^2 = {r:+.3f}   ({'REDUNDANT' if r > 0.6 else 'distinct axis'})")


# ── 4. Rank disagreement + concrete false-privacy cases ──────────────────────
def disagreement(data):
    print("\n" + "=" * 78)
    print("4. Rank disagreement with the criterion (Kendall tau, per dataset)")
    print("=" * 78)
    for ds in DATASETS:
        rows = data.get(ds, [])
        if len(rows) < 4:
            continue
        y = col(rows, CRITERION[0])
        print(f"\n  {ds} (n={len(rows)}):")
        for k, name in ALL_METRICS:
            if len(set(col(rows, k))) < 2:
                continue
            tau = kendalltau(col(rows, k), y).statistic
            print(f"    {name:<18} tau={tau:+.2f}")

    # False sense of privacy: proxy says "removed / oracle-like" yet attack is high.
    print("\n  'False sense of privacy' cases — looks unlearned by a proxy, still recoverable:")
    print("  (proxy in its 'most-unlearned' quartile, yet attack in the most-recoverable "
          "quartile)")
    for ds in DATASETS:
        rows = data.get(ds, [])
        if len(rows) < 6:
            continue
        atk = col(rows, CRITERION[0])
        atk_hi = np.quantile(atk, 0.75)
        # proxies oriented so "more unlearned" = low linear_auroc / high cka_oracle
        checks = [("linear_auroc", "low", "Linear AUROC~chance"),
                  ("cka_oracle", "high", "oracle-like geometry")]
        for key, direction, desc in checks:
            v = col(rows, key)
            thr = np.quantile(v, 0.25 if direction == "low" else 0.75)
            for i, r in enumerate(rows):
                looks = (v[i] <= thr) if direction == "low" else (v[i] >= thr)
                if looks and atk[i] >= atk_hi:
                    print(f"    [{ds}] {r['method']}/{r['run'][:22]}: {desc} "
                          f"({key}={v[i]:.3f}) yet attack@{r['adv_r']}={atk[i]:.3f}")


# ── 5. Scatter: attack vs proposed and vs standard ───────────────────────────
def scatter_figure(data):
    panels = [("cka_oracle", "CKA-to-oracle (proposed)"),
              ("tofu_ks", "TOFU KS-to-oracle (SOTA)"),
              ("linear_auroc", "Linear AUROC (standard)"),
              ("dir_norm", "dir_norm (standard)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (k, name) in zip(axes, panels):
        for ds in DATASETS:
            rows = data.get(ds, [])
            if not rows:
                continue
            ax.scatter(col(rows, k), col(rows, CRITERION[0]), s=28, alpha=0.75,
                       color=COLORS[ds], label=ds, edgecolor="k", linewidth=0.3)
        ax.set_xlabel(name); ax.set_ylabel("attack success (lower = more resistant)")
        ax.set_title(name, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="dataset")
    fig.suptitle("Does any metric predict adversarial recoverability?  "
                 "(each point = one operating point)", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "metric_vs_attack_scatter.png", dpi=150, bbox_inches="tight")
    print(f"\n-> {OUT / 'metric_vs_attack_scatter.png'}")


def criterion_agreement(data):
    """Do the two independent strong CRITERIA — the active adversarial attack and the
    per-example U-LiRA membership leakage — agree with each other, while both defy the
    geometry metrics? Agreement between two very different attacks strengthens the case
    that they measure genuine recoverability, not one attack's idiosyncrasy."""
    if not any("ulira_leak" in r and r["ulira_leak"] is not None
               for rows in data.values() for r in rows):
        return
    print("\n" + "=" * 78)
    print("5. Criterion cross-check — active attack vs per-example U-LiRA leakage (SOTA)")
    print("=" * 78)
    print(f"{'':<18}" + "".join(f"{ds:>12}" for ds in DATASETS) + f"{'pooled':>10}")
    # agreement of the two criteria
    px, py = [], []
    line = f"{'attack ~ U-LiRA':<18}"
    for ds in DATASETS:
        rows = [r for r in data.get(ds, []) if r.get("ulira_leak") is not None]
        if len(rows) >= 4:
            a = col(rows, CRITERION[0]); u = col(rows, "ulira_leak")
            line += f"{spearmanr(a, u).statistic:>+12.2f}"
            px += list(rank_norm(a)); py += list(rank_norm(u))
        else:
            line += f"{'n/a':>12}"
    line += f"{spearmanr(px, py).statistic:>+10.2f}" if len(px) >= 4 else f"{'n/a':>10}"
    print(line)
    print("  (higher = the two criteria rank operating points the same way)\n")
    # does any metric predict the U-LiRA criterion?
    print("  Spearman(metric, U-LiRA leakage) — pooled, alongside metric~attack from §1:")
    pooled_u = {}
    for ds in DATASETS:
        rows = [r for r in data.get(ds, []) if r.get("ulira_leak") is not None]
        if len(rows) < 4:
            continue
        for k, _ in ALL_METRICS:
            pooled_u.setdefault(k, {"m": [], "u": []})
            pooled_u[k]["m"] += list(rank_norm(col(rows, k)))
            pooled_u[k]["u"] += list(rank_norm(col(rows, "ulira_leak")))
    for k, name in ALL_METRICS:
        d = pooled_u.get(k)
        if d and len(d["u"]) >= 4:
            print(f"    {name:<18} rho(metric, U-LiRA) = {spearmanr(d['m'], d['u']).statistic:>+.2f}")


def radius_sensitivity(data):
    """Is the metric<->attack decoupling specific to the reported radius, or robust
    across the whole ball-radius curve? For each radius r, pooled Spearman(metric,
    attack@r). Flat, near-zero lines => the E18 finding is not a cherry-picked-radius
    artefact; the reported adv_r is just one point on a stable curve."""
    ex = next((r for rows in data.values() for r in rows if r.get("attack")), None)
    if not ex:
        return
    radii = sorted(float(k) for k in ex["attack"])
    metrics = [("cka_oracle", "CKA-to-oracle"), ("tofu_ks", "TOFU"),
               ("linear_auroc", "Linear AUROC"), ("dir_norm", "dir_norm")]
    print("\n" + "=" * 78)
    print("2b. Radius sensitivity — pooled Spearman(metric, attack@r) across the attack ball")
    print("=" * 78)
    hdr = f"{'metric':<16}" + "".join(f"{'r='+str(r):>9}" for r in radii)
    print(hdr); print("-" * len(hdr))
    curves = {}
    for k, name in metrics:
        ys, line = [], f"{name:<16}"
        for r in radii:
            mx, ax_ = [], []
            for ds, rows in data.items():
                if len(rows) < 4:
                    continue
                mv = col(rows, k)
                av = np.array([rr["attack"][str(r)] for rr in rows], dtype=float)
                mx += list(rank_norm(mv)); ax_ += list(rank_norm(av))
            rho = spearmanr(mx, ax_).statistic
            ys.append(rho); line += f"{rho:>+9.2f}"
        curves[name] = ys; print(line)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, ys in curves.items():
        ax.plot(radii, ys, marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("attack ball radius r"); ax.set_ylabel("pooled Spearman(metric, attack@r)")
    ax.set_ylim(-1, 1)
    ax.set_title("Metric↔attack correlation vs attack radius\n"
                 "(flat & near-zero ⇒ the decoupling is not radius-specific)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "radius_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"\n-> {OUT / 'radius_sensitivity.png'}")
    from src.unlearning.spec import REPORT_RADIUS as _RR
    print(f"Reading: the reported adv_r ({_RR} per dataset) is one point on")
    print("these curves; if they stay weak/flat, no radius rescues the geometry metrics.")


def main():
    data = load_points()
    counts = {ds: len(rows) for ds, rows in data.items()}
    print(f"Operating points loaded: {counts}  (total {sum(counts.values())})")
    if sum(counts.values()) < 8:
        print("Not enough data yet — run scripts/run_dense_recovery.py first.")
        return
    pooled, keys = pooled_matrix(data)
    criterion_correlations(data, pooled)
    radius_sensitivity(data)
    correlation_heatmap(pooled, keys)
    incremental_validity(pooled)
    disagreement(data)
    criterion_agreement(data)
    scatter_figure(data)
    # persist the pooled table for the writeup
    (ROOT / "results/incremental_validity.json").write_text(json.dumps(
        {"counts": counts,
         "criterion_spearman_pooled": {k: float(spearmanr(pooled[k], pooled[CRITERION[0]]).statistic)
                                       for k, _ in ALL_METRICS}}, indent=2))
    print(f"\n-> results/incremental_validity.json")


if __name__ == "__main__":
    main()
