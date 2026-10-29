"""
src/evaluation/separation.py
=============================
How far apart are the concept-positive and concept-negative latents, measured
in a way that survives the model rescaling its own latent space.

Why this module exists
----------------------
The comparison notebooks report

    direction_norm = || mean(mu | forget) - mean(mu | retain) ||

captioned "lower = concept more erased from latent space". That caption has
one confirmed problem and one claim that turned out, on closer checking, to
be a checkpoint bug rather than a real finding -- both worth recording so the
mistake doesn't get remade.

1. CONFIRMED: it is not scale-invariant. Any method that contracts the latent
   space uniformly shrinks it without removing anything decodable, and any
   method that inflates mu norms (NegGrad+ ascends the forget ELBO, whose KL
   term is maximised by pushing ||mu_f|| up) inflates it without adding
   anything. cohens_d and mahalanobis below divide the gap by the spread, so
   a global rescale cancels. This is the reason to prefer them over raw
   direction_norm regardless of anything else in this docstring.

2. RETRACTED: an earlier version of this docstring reported that the dSprites
   oracle shows HIGHER direction_norm/cohens_d/mahalanobis than the baseline
   ("hearts are OOD for the oracle, so it separates them more"). That was
   measured against results/dsprites/dsprites_oracle/unlearned.pt -- a
   leftover artefact from a deleted training path (src/unlearning/oracle.py,
   recoverable only from git's index) that reinitialised weights with
   PyTorch's default reset_parameters() instead of this project's Kaiming
   init, and trained with val_loader=None -- no validation tracking at all.
   The CORRECT oracle checkpoint is epoch_0150.pt, produced by the same
   config-driven experiments/train.py pipeline as the baseline. Against it,
   on 8192 dSprites test images:

       model      direction_norm  cohens_d  mahalanobis
       baseline            0.365     0.133        0.459
       oracle              0.198     0.083        0.240

   The oracle separates hearts LESS than the baseline, as naively expected.
   direction_norm/cohens_d/mahalanobis do NOT invert here. The inversion is
   real but lives elsewhere: mlp_auroc, knn_cohesion, and lift_norm all still
   score the (correct) oracle higher than baseline -- see the caveat in
   nb_utils.compute_progress's docstring, which used the correct checkpoint
   throughout and did not need correcting.

What to report
---------------
Even without the inversion motivating it, oracle_gap() is still the more
defensible framing: it makes explicit that the TARGET is wherever the oracle
sits, not zero, which protects against exactly the failure mode above (a
method that undershoots past the oracle would look "better than gold standard"
under "lower is better" -- nothing in this project has undershot that far
yet, but the framing shouldn't depend on that staying true). Use it alongside
cohens_d/mahalanobis rather than raw direction_norm.
"""

from __future__ import annotations

import numpy as np


def concept_separation(mu, labels, std=None, *, forget_val=1, ridge=1e-6) -> dict:
    """Separation between concept-positive and concept-negative latents.

    Parameters
    ----------
    mu     : (N, D) posterior means.
    labels : (N,) concept labels.  Compared against ``forget_val``; pass an
             already-binarised array and leave forget_val=1, or pass raw
             dSprites shape classes (0/1/2) with forget_val=2.
    std    : (N, D) posterior stds, optional.  Only needed for sigma_ratio.
    ridge  : added to the pooled covariance diagonal before inversion, as a
             fraction of its mean diagonal.  Guards a collapsed latent
             dimension -- which is exactly what an aggressive unlearning run
             produces, so this is a real case, not a formality.

    Returns
    -------
    dict with:
        direction_norm  || mean_f - mean_r ||          raw, NOT scale-invariant
        cohens_d        direction_norm / pooled_std    invariant to isotropic scaling
        mahalanobis     sqrt(d' S^-1 d), S pooled      invariant to any invertible
                                                       linear map of the latent space
        sigma_ratio     mean(std | forget) / mean(std | retain), or None
        n_forget, n_retain
    """
    mu = np.asarray(mu, dtype=np.float64)
    y = np.asarray(labels).reshape(-1) == forget_val
    n_f, n_r = int(y.sum()), int((~y).sum())
    if n_f == 0 or n_r == 0:
        raise ValueError(f"need both classes present, got {n_f} forget / {n_r} retain")

    f, r = mu[y], mu[~y]
    d = f.mean(0) - r.mean(0)
    direction_norm = float(np.linalg.norm(d))

    # Pooled within-class scatter.  Summing per-dimension variances (rather
    # than using the trace of the covariance separately per class) keeps this
    # the exact multivariate analogue of Cohen's d.
    pooled_var = 0.5 * (f.var(0).sum() + r.var(0).sum())
    pooled_std = float(np.sqrt(pooled_var))
    cohens_d = float(direction_norm / pooled_std) if pooled_std > 0 else 0.0

    # Pooled covariance, degrees-of-freedom weighted as in LDA.
    cov = ((n_f - 1) * np.cov(f, rowvar=False) +
           (n_r - 1) * np.cov(r, rowvar=False)) / max(n_f + n_r - 2, 1)
    cov = np.atleast_2d(cov)
    cov = cov + ridge * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
    m2 = float(d @ np.linalg.solve(cov, d))
    mahalanobis = float(np.sqrt(max(m2, 0.0)))

    sigma_ratio = None
    if std is not None:
        std = np.asarray(std, dtype=np.float64)
        denom = float(std[~y].mean())
        sigma_ratio = float(std[y].mean() / denom) if denom > 0 else None

    return {
        "direction_norm": direction_norm,
        "cohens_d":       cohens_d,
        "mahalanobis":    mahalanobis,
        "sigma_ratio":    sigma_ratio,
        "n_forget":       n_f,
        "n_retain":       n_r,
    }


def oracle_gap(model_metrics: dict, oracle_metrics: dict, keys=None) -> dict:
    """Absolute distance from the oracle, per metric -- lower is better.

    The oracle is the definition of successful unlearning, so it is the target
    rather than an extreme.  On metrics where the oracle does not sit at an
    extreme (see this module's docstring: it separates the concept MORE than
    the baseline), "minimise the raw metric" and "match the oracle" point in
    opposite directions, and only the second one is defensible.

    Keys whose value is None in either dict are skipped rather than treated as
    zero, so an absent sigma_ratio does not silently score as a perfect match.
    """
    keys = keys or [k for k in model_metrics
                    if isinstance(model_metrics.get(k), (int, float))
                    and not isinstance(model_metrics.get(k), bool)]
    out = {}
    for k in keys:
        a, b = model_metrics.get(k), oracle_metrics.get(k)
        if a is None or b is None:
            continue
        out[k] = float(abs(float(a) - float(b)))
    return out
