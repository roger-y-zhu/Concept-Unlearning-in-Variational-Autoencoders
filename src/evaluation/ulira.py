"""U-LiRA scoring (Hayes et al. 2024, arXiv:2403.01218) -- pure numpy, no model code.

Specification (ledger D4; see E29 for why the previous calibration saturated)
---------------------------------------------------------------------------
Q1 ("IN")  = models trained WITH the concept and then UNLEARNED by the algorithm under test
             (same method, same hyper-parameters, replicate baselines).
Q0 ("OUT") = retrained models (oracle replicates).
Signal     = per-example log negative-ELBO (log-scaling is the analogue of Hayes' rescaled
             logit: it makes the per-example score approximately Gaussian and homoscedastic;
             the raw OUT spread is heavy-tailed, per-example std 37 median vs 7720 max).
Attack     = per-example Gaussian likelihood ratio with one POOLED (global) variance per side
             (Carlini et al.'s few-shadow variant); Lambda = logN(x|IN) - logN(x|OUT).
Reported   = Hayes' balanced attack ADVANTAGE: mean over examples of (TPR - FPR) at Lambda>0,
             where TPR is over IN models and FPR over OUT models, every model scored
             LEAVE-ONE-OUT (a shadow is never scored against a fit that contains it).
             0 = unlearning is indistinguishable from retraining, 1 = fully distinguishable.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

LOG_EPS = 1e-3


def log_scale(s, eps: float = LOG_EPS):
    return np.log(np.maximum(np.asarray(s, dtype=np.float64), eps))


def fit(mat: np.ndarray, unbiased: bool = True):
    """Per-example mean + ONE pooled std over all examples. mat: (n_models, n_examples).
    `unbiased` divides by n-1 (with only 3-4 shadows the biased estimator is ~13-25% small)."""
    n = mat.shape[0]
    if n < 2:
        raise ValueError("need >=2 shadow models to estimate a variance")
    mu = mat.mean(axis=0)
    denom = (n - 1) if unbiased else n
    var = float(((mat - mu) ** 2).sum() / (denom * mat.shape[1]))
    return mu, max(float(np.sqrt(var)), 1e-6)


def llr(x, in_fit, out_fit):
    return norm.logpdf(x, *in_fit) - norm.logpdf(x, *out_fit)


def loo_llrs(in_mat: np.ndarray, out_mat: np.ndarray, unbiased: bool = True):
    """Leave-one-out Lambda for every IN and every OUT model. Returns (n_in,N), (n_out,N)."""
    out_all, in_all = fit(out_mat, unbiased), fit(in_mat, unbiased)
    in_l = np.stack([llr(in_mat[i], fit(np.delete(in_mat, i, 0), unbiased), out_all)
                     for i in range(in_mat.shape[0])])
    out_l = np.stack([llr(out_mat[j], in_all, fit(np.delete(out_mat, j, 0), unbiased))
                      for j in range(out_mat.shape[0])])
    return in_l, out_l


def summarize(in_l: np.ndarray, out_l: np.ndarray, sat: float = 20.0) -> dict:
    """Attack advantage + diagnostics from the LOO Lambda matrices."""
    tpr = (in_l > 0).mean(axis=0)
    fpr = (out_l > 0).mean(axis=0)
    y = np.r_[np.ones(in_l.size), np.zeros(out_l.size)]
    auc = float(roc_auc_score(y, np.r_[in_l.ravel(), out_l.ravel()]))
    return {
        "adv": float((tpr - fpr).mean()),
        "post_in": float(expit(in_l).mean()),     # P(unlearned | an unlearned model)
        "post_out": float(expit(out_l).mean()),   # P(unlearned | a retrained model)
        "auc": auc,
        "sat_frac": float((np.abs(np.r_[in_l.ravel(), out_l.ravel()]) > sat).mean()),
    }


def score_point(in_raw: np.ndarray, out_raw: np.ndarray, *, log: bool = True,
                unbiased: bool = True) -> dict:
    """Full U-LiRA score for one operating point from raw per-example negELBO matrices."""
    f = log_scale if log else (lambda a: np.asarray(a, dtype=np.float64))
    in_m, out_m = f(in_raw), f(out_raw)
    in_l, out_l = loo_llrs(in_m, out_m, unbiased)
    d = summarize(in_l, out_l)
    d["sigma_in"] = fit(in_m, unbiased)[1]
    d["sigma_out"] = fit(out_m, unbiased)[1]
    d["gap_sigma_in"] = float(np.abs(out_m.mean(0) - in_m.mean(0)).mean() / d["sigma_in"])
    return d


def target_posterior(x_target, in_mat, out_mat, *, unbiased: bool = False) -> float:
    """Mean per-example membership posterior of ONE target model against fixed IN/OUT fits.
    With unbiased=False and raw scores this is exactly the legacy v3 `ulira_leak`."""
    lam = llr(x_target, fit(in_mat, unbiased), fit(out_mat, unbiased))
    return float(expit(lam).mean())
