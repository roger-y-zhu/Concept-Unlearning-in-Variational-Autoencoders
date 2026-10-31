"""
scripts/run_ulira.py
====================
A per-example U-LiRA-style membership criterion (Hayes et al. 2024, arXiv:2403.01218,
papers/summaries/hayes2024_ulira.md; built on Carlini et al. 2022 LiRA,
papers/summaries/carlini2022_lira.md), adapted to VAE concept unlearning and added
to every dense operating point for the incremental-validity study (ledger D1/E18).

Hayes' point: a PER-EXAMPLE attacker (a calibrated test per forget example) is far
stronger than a POPULATION attacker (one threshold for all), and is the criterion
that avoids a "false sense of privacy". We instantiate it with the per-example
negative-ELBO as the membership signal (members = trained-on = lower loss), calibrated
against shadow models:

  IN  shadows = the baseline replicates  (trained WITH the concept)      -> 4 models
  OUT shadows = the oracle replicates    (retrained WITHOUT the concept) -> 4 models

For each forget (concept-positive) example x_i we fit a Gaussian of its negELBO under
the IN shadows and under the OUT shadows (Carlini's global-variance variant: per-example
means, a single pooled variance per side, robust with few shadows). For a target model
the per-example log-likelihood-ratio Lambda_i = logN(s_i|IN) - logN(s_i|OUT) gives the
membership posterior p_i = sigmoid(Lambda_i); the model's **U-LiRA leakage** is
mean_i p_i -- how concept-retaining it looks, per-example-calibrated. Reference rows
(baseline/oracle) are scored leave-one-out (their own shadow excluded).

Writes `ulira_leak` (mean membership posterior; high = concept retained = leaky) and
`ulira_auc` (per-example AUROC separating IN-shadow from OUT-shadow scores on this
forget set -- the attack's own strength) into results/<ds>/dense_recovery.json.
Also compares per-example vs population calibration (Hayes' central claim).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.membership import per_sample_negative_elbo  # noqa: E402
from src.evaluation import ulira as U4  # noqa: E402  (v4 scorer, ledger D4)
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402
from run_tofu_forget_quality import forget_loader, row_ckpt  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = {"dsprites": "epoch_0150.pt", "celeba": "epoch_0250.pt", "mpi3d": "epoch_0120.pt"}
EPOCH.update({f"{d}_b4": e for d, e in list(EPOCH.items())})   # beta=4 variants: same epoch counts


def shadow_dirs(dataset: str):
    b = [f"results/{dataset}/{dataset}_baseline"] + \
        [f"results/{dataset}/{dataset}_baseline_ts{i}" for i in (1, 2, 3)]
    o = [f"results/{dataset}/{dataset}_oracle"] + \
        [f"results/{dataset}/{dataset}_oracle_ts{i}" for i in (1, 2, 3)]
    ep = EPOCH[dataset]
    IN = [ROOT / d / ep for d in b if (ROOT / d / ep).exists()]
    OUT = [ROOT / d / ep for d in o if (ROOT / d / ep).exists()]
    return IN, OUT


def scores_over(ckpts, loader):
    """(n_models, n_forget) matrix of per-example negative ELBO."""
    rows = []
    for ck in ckpts:
        m, _ = load_vae_checkpoint(ck, DEVICE)
        rows.append(per_sample_negative_elbo(m, loader, DEVICE))
    return np.vstack(rows)


def gaussian_params(mat, exclude_path=None, paths=None):
    """Per-example means + a single pooled std (global-variance LiRA). Optionally
    leave one shadow out (for scoring a reference model that is itself a shadow)."""
    if exclude_path is not None and paths is not None:
        keep = [i for i, p in enumerate(paths) if p != exclude_path]
        mat = mat[keep]
    mu = mat.mean(axis=0)
    pooled_std = float(np.sqrt(((mat - mu) ** 2).mean())) or 1.0
    return mu, max(pooled_std, 1e-6)


# Bump when the U-LiRA computation changes. Version-aware skip so a code change
# forces recompute instead of being silently skipped as "already present".
#   v2 — introduced the version-aware skip (replaced the "already present" check)
#   v3 — attack-strength AUROC now leave-one-out (was self-inclusion-biased; R1 audit)
ULIRA_VERSION = 3        # v3 = leave-one-out attack-strength AUROC (R1 audit fix)


def run_dataset(dataset: str) -> None:
    agg = ROOT / f"results/{dataset}/dense_recovery.json"
    if not agg.exists():
        print(f"  {dataset}: no dense_recovery.json"); return
    rows = json.loads(agg.read_text())
    if rows and all(r.get("_ulira_v") == ULIRA_VERSION for r in rows):
        print(f"  {dataset}: ulira up to date (v{ULIRA_VERSION})"); return

    IN, OUT = shadow_dirs(dataset)
    print(f"\n{dataset}: {len(IN)} IN (baseline) + {len(OUT)} OUT (oracle) shadows")
    if len(IN) < 2 or len(OUT) < 2:
        print("  too few shadows, skipping"); return
    loader = forget_loader(dataset)
    in_mat = scores_over(IN, loader)          # (n_in, n_forget)
    out_mat = scores_over(OUT, loader)        # (n_out, n_forget)
    n_forget = in_mat.shape[1]
    print(f"  forget set n={n_forget}")

    # Attack strength (per-example AUROC): can the calibrated per-example signal
    # separate an IN-shadow score from an OUT-shadow score on this forget set?
    # Compare to a POPULATION baseline (single global threshold) -- Hayes' claim.
    # Attack-strength AUROC (IN- vs OUT-shadow separability) MUST be leave-one-out:
    # scoring a shadow against a Gaussian fit that INCLUDES itself inflates the
    # separation (self-inclusion), badly with only 4 shadows -- it made the reported
    # per-example AUROC optimistic. Each shadow is scored against its OWN side fit on
    # the OTHER shadows; the opposite side (which never contains it) uses the full
    # fit. (The per-operating-point leak below is already LOO for reference rows.)
    mu_in_all, s_in_all = gaussian_params(in_mat)
    mu_out_all, s_out_all = gaussian_params(out_mat)
    in_paths, out_paths = [str(p) for p in IN], [str(p) for p in OUT]

    def _loo_lam(mat, paths, side_in):
        rows_lam = []
        for i, p in enumerate(paths):
            mu_s, s_s = gaussian_params(mat, p, paths)   # this side, fit without shadow i
            if side_in:
                rows_lam.append(norm.logpdf(mat[i], mu_s, s_s)
                                - norm.logpdf(mat[i], mu_out_all, s_out_all))
            else:
                rows_lam.append(norm.logpdf(mat[i], mu_in_all, s_in_all)
                                - norm.logpdf(mat[i], mu_s, s_s))
        return np.concatenate(rows_lam)

    in_lam = _loo_lam(in_mat, in_paths, True)
    out_lam = _loo_lam(out_mat, out_paths, False)
    y = np.r_[np.ones_like(in_lam), np.zeros_like(out_lam)]
    orient = lambda a: max(a, 1 - a)   # a membership signal is direction-free; report its strength
    perex_auc = orient(roc_auc_score(y, np.r_[in_lam, out_lam]))
    pop_auc = orient(roc_auc_score(y, np.r_[in_mat.ravel(), out_mat.ravel()]))  # raw negELBO, no per-example calib
    print(f"  attack AUROC (IN vs OUT shadow scores):  per-example={perex_auc:.3f}  "
          f"population={pop_auc:.3f}  (Hayes: per-example should be >= population)")

    for r in rows:
        ck = row_ckpt(dataset, r)
        if not ck.exists():
            r["ulira_leak"], r["ulira_auc"], r["_ulira_v"] = None, None, ULIRA_VERSION
            continue
        excl = str(ck) if r["method"] in ("baseline", "oracle") else None
        mu_i, s_i = gaussian_params(in_mat, excl, [str(p) for p in IN])
        mu_o, s_o = gaussian_params(out_mat, excl, [str(p) for p in OUT])
        m, _ = load_vae_checkpoint(ck, DEVICE)
        s = per_sample_negative_elbo(m, loader, DEVICE)
        lam_i = norm.logpdf(s, mu_i, s_i) - norm.logpdf(s, mu_o, s_o)
        leak = float(np.mean(1.0 / (1.0 + np.exp(-lam_i))))   # mean per-example membership posterior
        r["ulira_leak"] = leak                     # high = concept retained = leaky
        r["ulira_auc"] = float(perex_auc)          # dataset-level attack strength (constant per ds)
        r["_ulira_v"] = ULIRA_VERSION
        print(f"  [{r['method']:<14} {r['run'][:24]:<24}] ulira_leak={leak:.3f} "
              f"attack@{r['adv_r']}={r['attack_at_r']:.3f}")

    agg.write_text(json.dumps(rows, indent=2))
    print(f"  -> updated {agg.relative_to(ROOT)}")


# ---------------------------------------------------------------------------------------------
# v4 (ledger D4/E30): Hayes-faithful. IN = the SAME method+config applied to the replicate
# baselines (baseline_ts1..3, produced by scripts/run_ulira_shadows.py) plus the target itself;
# OUT = oracle replicates; log-scaled negELBO; leave-one-out balanced attack ADVANTAGE.
# Adds fields only -- the legacy v3 `ulira_leak` is left untouched (kept as the
# "baseline-likeness" score; mirrored to `ulira_leak_vs_baseline` for clarity).
#   ulira_adv      Hayes balanced advantage, 0 = indistinguishable from retrain, 1 = fully distinguishable
#   ulira_adv_raw  same on the raw (non-log) scale -- a sensitivity check, not the headline
#   ulira_post_in  mean P(unlearned | unlearned model);  ulira_auc4 / ulira_sat: attack AUROC / saturated fraction
# Rows without all shadows get None (never a fabricated value).
ULIRA4_VERSION = 1
SHADOW_TAGS = ("ts1", "ts2", "ts3")


def shadow_ckpts(dataset: str, row: dict):
    m, run = row["method"], row["run"]
    base = ROOT / f"results/{dataset}/{dataset}_{m}"
    return [base / f"{run}_bl{t}" / "final.pt" for t in SHADOW_TAGS]


def run_dataset_v4(dataset: str) -> None:
    agg = ROOT / f"results/{dataset}/dense_recovery.json"
    rows = json.loads(agg.read_text())
    _, OUT = shadow_dirs(dataset)
    loader = forget_loader(dataset)
    out_raw = scores_over(OUT, loader)
    n_done = n_skip = 0
    for r in rows:
        r["ulira_leak_vs_baseline"] = r.get("ulira_leak")
        r.update(ulira_adv=None, ulira_adv_raw=None, ulira_post_in=None, ulira_auc4=None,
                 ulira_sat=None, ulira_n_shadows=0, _ulira4_v=ULIRA4_VERSION)
        if r["method"] in ("baseline", "oracle"):
            n_skip += 1; continue
        tgt, sh = row_ckpt(dataset, r), shadow_ckpts(dataset, r)
        if not tgt.exists() or not all(p.exists() for p in sh):
            n_skip += 1; continue
        mats = []
        for ck in [tgt] + sh:
            m, _ = load_vae_checkpoint(ck, DEVICE)
            mats.append(per_sample_negative_elbo(m, loader, DEVICE))
        in_raw = np.vstack(mats)
        d, dr = U4.score_point(in_raw, out_raw, log=True), U4.score_point(in_raw, out_raw, log=False)
        r.update(ulira_adv=d["adv"], ulira_adv_raw=dr["adv"], ulira_post_in=d["post_in"],
                 ulira_auc4=d["auc"], ulira_sat=d["sat_frac"], ulira_n_shadows=len(sh))
        n_done += 1
        print(f"  [{r['method']:<14} {r['run'][:24]:<24}] adv={d['adv']:.3f} raw={dr['adv']:.3f} "
              f"post_in={d['post_in']:.3f} (v3 {r.get('ulira_leak')})", flush=True)
    agg.write_text(json.dumps(rows, indent=2))
    print(f"{dataset}: v4 scored {n_done} points, {n_skip} without shadows -> {agg.relative_to(ROOT)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(EPOCH) + ["all"], default="all")   # "all" = the beta=1 three
    ap.add_argument("--v4", action="store_true",
                    help="Hayes-faithful scoring from per-config shadow unlearnings (ledger D4); "
                         "adds ulira_adv etc., leaves the legacy ulira_leak untouched")
    args = ap.parse_args()
    datasets = ["dsprites", "celeba", "mpi3d"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset_v4(ds) if args.v4 else run_dataset(ds)


if __name__ == "__main__":
    main()
