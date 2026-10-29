"""
src/evaluation/disentanglement.py
===================================
MIG and DCI -- the standard disentanglement metrics from the VAE literature
(Chen et al. 2018; Eastwood & Williams 2018), applied here to a question that
literature never asks: does UNLEARNING damage disentanglement of the factors
that were never targeted?

Why this module exists
-----------------------
docs/EVALUATION_LANDSCAPE.md's gap analysis: the disentanglement literature
already discovered that its OWN metrics disagree about which model is best
(Locatello et al. 2019, "A Sober Look..." -- DCI and FactorVAE disagree, while
beta-VAE tops both MIG and SAP). That is this project's central finding, in a
neighbouring field, about a neighbouring latent-space property. Computing both
here gives the same argument a second, independent measurement axis, on the
non-concept factors this project has never separately scored.

dSprites-ONLY: MIG and DCI both need ground-truth generative factors, which
dSprites has (scale, orientation, x, y -- shape is the forget concept and
color is constant, so both are excluded) and CelebA does not.

MIG (Chen et al. 2018)
-----------------------
For each factor v_k, take the two latent dims with highest mutual information
with v_k. The gap between them, normalised by H(v_k), measures whether a
SINGLE dimension captures the factor (high MIG) or the factor is smeared
across several (low MIG).

DCI-Disentanglement (Eastwood & Williams 2018)
------------------------------------------------
Fit a regressor predicting each factor from ALL latent dims; take feature
importances as a dim x factor relevance matrix R. A dimension that is
important for exactly one factor scores high (low entropy of its importance
row); a dimension smeared across many factors scores low. The overall score
weights each dimension's entropy by how much it matters overall, so an unused
dimension cannot inflate the score just by being spuriously "pure".

MIG and DCI measure related but NOT identical things (this is itself a
documented disagreement in the source literature -- see module docstring
above) -- MIG only ever credits the TOP-2 dimensions per factor and ignores
the rest of the importance distribution entirely, while DCI's entropy term
integrates over all D dimensions. Report both; do not average them.
"""

from __future__ import annotations

import numpy as np


def _discretise(x: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Equal-width binning for mutual-information estimation.

    MI estimators need discrete inputs on both sides. Latents are continuous;
    equal-width (not equal-frequency) binning is used because it is the
    standard choice in the disentanglement-metrics literature and keeps bin
    membership comparable across runs with different latent scales -- an
    equal-frequency binning would silently normalise away exactly the kind of
    scale change this project's other metrics (separation.py) had to guard
    against.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros(len(x), dtype=int)
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[-1] += 1e-9  # include the max value in the last bin
    return np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)


def _mutual_info_discrete(a: np.ndarray, b: np.ndarray) -> float:
    """MI in nats between two discrete arrays, via the joint histogram.

    Not sklearn.metrics.mutual_info_score to avoid a version-dependent
    dependency for something this small; verified against it in tests.
    """
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ca = {v: i for i, v in enumerate(np.unique(a))}
    cb = {v: i for i, v in enumerate(np.unique(b))}
    joint = np.zeros((len(ca), len(cb)))
    for x, y in zip(a, b):
        joint[ca[x], cb[y]] += 1
    joint /= n
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (pa * pb)
        terms = joint * np.log(ratio)
    return float(np.nansum(terms[joint > 0]))


def _entropy_discrete(a: np.ndarray) -> float:
    a = np.asarray(a)
    _, counts = np.unique(a, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def mutual_info_gap(mu: np.ndarray, factors: dict[str, np.ndarray],
                    *, n_bins: int = 20) -> dict:
    """MIG per factor and the overall mean.

    Parameters
    ----------
    mu      : (N, D) latent means.
    factors : {name: (N,) ground-truth factor values}, continuous or discrete
              (both are re-binned internally so mixed types are fine).

    Returns
    -------
    dict with:
        per_factor : {name: {mig, top_dim, top2_dim, I_top, I_top2, H_factor}}
        mig        : mean MIG across factors
    """
    mu = np.asarray(mu, dtype=np.float64)
    n, d = mu.shape
    z_bins = [_discretise(mu[:, j], n_bins) for j in range(d)]

    per_factor = {}
    for name, v in factors.items():
        v_bins = _discretise(np.asarray(v, dtype=np.float64), n_bins) \
            if np.issubdtype(np.asarray(v).dtype, np.floating) else np.asarray(v)
        h_v = _entropy_discrete(v_bins)
        mis = np.array([_mutual_info_discrete(z_bins[j], v_bins) for j in range(d)])
        order = np.argsort(mis)[::-1]
        top, top2 = order[0], order[1] if d > 1 else order[0]
        i_top, i_top2 = mis[top], mis[top2]
        mig = float((i_top - i_top2) / h_v) if h_v > 0 else float("nan")
        per_factor[name] = {
            "mig": mig, "top_dim": int(top), "top2_dim": int(top2),
            "I_top": float(i_top), "I_top2": float(i_top2), "H_factor": float(h_v),
        }
    valid = [v["mig"] for v in per_factor.values() if v["mig"] == v["mig"]]
    return {"per_factor": per_factor,
            "mig": float(np.mean(valid)) if valid else float("nan")}


def dci_disentanglement(mu: np.ndarray, factors: dict[str, np.ndarray],
                        *, n_estimators: int = 50, seed: int = 0) -> dict:
    """DCI-Disentanglement score and the per-dimension importance matrix.

    Uses a small RandomForestRegressor per factor (fast, no scaling needed,
    handles the nonlinear factor/latent relationships a VAE can produce).

    Returns
    -------
    dict with:
        importance : (D, K) matrix, R[j,k] = importance of dim j for factor k
        rho        : (D,) relative importance of each dimension overall
        per_dim_disentanglement : (D,) 1 - normalised entropy of dim j's row
        disentanglement : rho-weighted mean of per_dim_disentanglement
    """
    from sklearn.ensemble import RandomForestRegressor

    mu = np.asarray(mu, dtype=np.float64)
    n, d = mu.shape
    names = list(factors.keys())
    k = len(names)
    R = np.zeros((d, k))
    for ki, name in enumerate(names):
        v = np.asarray(factors[name], dtype=np.float64)
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                    n_jobs=-1)
        reg.fit(mu, v)
        R[:, ki] = reg.feature_importances_

    row_sums = R.sum(1, keepdims=True)
    # A dimension the forest never used (all-zero row) has an undefined
    # importance DISTRIBUTION, not a "perfectly disentangled" one -- score it
    # 0, and give it zero weight via rho so it cannot inflate the mean either.
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(row_sums > 0, R / row_sums, 0.0)
    log_k = np.log(k) if k > 1 else 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.array([_entropy_discrete_from_probs(p[j]) if row_sums[j] > 0 else np.nan
                      for j in range(d)])
    per_dim = np.where(np.isnan(h), 0.0, 1.0 - h / log_k)

    total = R.sum()
    rho = (R.sum(1) / total) if total > 0 else np.zeros(d)

    return {
        "importance": R.tolist(),
        "rho": rho.tolist(),
        "per_dim_disentanglement": per_dim.tolist(),
        "disentanglement": float(np.sum(rho * per_dim)),
        "factor_names": names,
    }


def _entropy_discrete_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if len(p) else 0.0
