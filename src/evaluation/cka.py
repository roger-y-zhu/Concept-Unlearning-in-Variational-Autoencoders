"""
src/evaluation/cka.py
Centered Kernel Alignment (CKA) for comparing latent representations.

Reference
---------
Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
https://arxiv.org/abs/1905.00414

Two estimators are provided:

  linear_cka  — linear kernel, operates in feature space → O(n·p·q) time and
                O(p·q) memory.  Preferred for large n (e.g. 70 k dSprites samples).

  kernel_cka  — RBF kernel, operates in sample space → O(n²) time and memory.
                Use sample_n to subsample when n is large.

Both return a scalar in [0, 1].  CKA = 1 means the two representations are
identical up to orthogonal transformation and scaling; CKA = 0 means they share
no linear structure.

Public API
----------
    linear_cka(X, Y)                   → float
    kernel_cka(X, Y, sigma, sample_n)  → float
    cka_matrix(reps_dict, method)       → {"matrix": ndarray, "labels": list[str]}
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _debiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (Song et al. 2012 / Kornblith et al. 2019 eq. 2).

    Both K and L must be (n, n) kernel matrices with zeros on the diagonal
    (set diagonal to 0 before calling).

    Returns a scalar (can be negative for the unbiased estimator; CKA clips
    the ratio to [0, 1] afterwards).
    """
    n = K.shape[0]
    assert K.shape == L.shape == (n, n), "K and L must be square matrices of the same size"

    KL   = K @ L
    tr   = np.trace(KL)
    s_K  = K.sum()
    s_L  = L.sum()
    s_KL = KL.sum()

    return (
        tr
        + s_K * s_L / ((n - 1) * (n - 2))
        - 2 * s_KL / (n - 2)
    ) / (n * (n - 3))


def _gram_rbf(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix K_ij = exp(-||x_i - x_j||² / (2σ²)).

    If sigma is None, use the median heuristic: σ = median(pairwise distances).
    """
    # Squared pairwise distances via the identity ||a-b||² = ||a||²+||b||²-2a·b
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    D2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)  # numerical safety

    if sigma is None:
        # median heuristic on the upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(len(X), k=1)
        median_d = np.median(np.sqrt(D2[triu_idx]))
        sigma = median_d if median_d > 0 else 1.0

    K = np.exp(-D2 / (2.0 * sigma ** 2))
    return K


# ──────────────────────────────────────────────────────────────────────────────
# Public estimators
# ──────────────────────────────────────────────────────────────────────────────

def linear_cka(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    standardise: bool = False,
) -> float:
    """
    Linear-kernel CKA between two representation matrices.

    Operates entirely in feature space — no n×n matrix is formed — so it
    scales to large datasets.

    Parameters
    ----------
    X : (n, p) array — representations from model / layer A.
    Y : (n, q) array — representations from model / layer B.
        Must have the same number of rows (same samples in the same order).
    standardise : bool
        If True, standardise each feature to unit variance before computing.
        Default False (mean-centring is always applied internally).

    Returns
    -------
    float in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows"

    if standardise:
        std_X = X.std(axis=0, keepdims=True)
        std_Y = Y.std(axis=0, keepdims=True)
        X = X / np.where(std_X > 0, std_X, 1.0)
        Y = Y / np.where(std_Y > 0, std_Y, 1.0)

    # Mean-centre (equivalent to applying the centering matrix H)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # ||Y^T X||_F² / (||X^T X||_F · ||Y^T Y||_F)
    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    num   = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denom == 0.0:
        return 0.0
    return float(np.clip(num / denom, 0.0, 1.0))


def kernel_cka(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sigma: float | None = None,
    standardise: bool = False,
    sample_n: int | None = 5_000,
    seed: int = 0,
) -> float:
    """
    RBF-kernel CKA between two representation matrices (unbiased estimator).

    Captures nonlinear similarity that linear CKA can miss.  Requires forming
    n×n kernel matrices, so subsample for large datasets.

    Parameters
    ----------
    X, Y      : (n, p) and (n, q) arrays — same samples, same order.
    sigma     : RBF bandwidth; None → median heuristic applied independently
                to X and Y (standard practice).
    standardise : bool — standardise features before computing kernels.
    sample_n  : subsample to this many points before computing (None = use all).
    seed      : RNG seed for subsampling.

    Returns
    -------
    float in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of rows"

    if standardise:
        std_X = X.std(axis=0, keepdims=True)
        std_Y = Y.std(axis=0, keepdims=True)
        X = X / np.where(std_X > 0, std_X, 1.0)
        Y = Y / np.where(std_Y > 0, std_Y, 1.0)

    # Optional subsampling
    n = X.shape[0]
    if sample_n is not None and n > sample_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, sample_n, replace=False)
        X, Y = X[idx], Y[idx]

    # RBF gram matrices
    K = _gram_rbf(X, sigma)
    L = _gram_rbf(Y, sigma)

    # Zero the diagonals (required by the unbiased HSIC estimator)
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    hsic_kl = _debiased_hsic(K, L)
    hsic_kk = _debiased_hsic(K, K)
    hsic_ll = _debiased_hsic(L, L)

    denom = np.sqrt(max(hsic_kk, 0.0) * max(hsic_ll, 0.0))
    if denom == 0.0:
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise CKA matrix
# ──────────────────────────────────────────────────────────────────────────────

def cka_matrix(
    representations: dict[str, np.ndarray],
    *,
    method: str = "linear",
    **kwargs,
) -> dict:
    """
    Compute the pairwise CKA matrix for a collection of representations.

    Parameters
    ----------
    representations : dict mapping model name → (n, d) mu array.
        All arrays must contain the **same samples in the same order** so that
        row i in every array corresponds to the same input image.
    method : "linear" | "kernel"
        Which CKA estimator to use.
    **kwargs : forwarded to the chosen estimator (e.g. sigma, sample_n, seed).

    Returns
    -------
    dict with:
        "matrix" : (k, k) float64 ndarray, symmetric, diagonal = 1.
        "labels" : list[str] — model names in insertion order.
    """
    if method == "linear":
        fn = linear_cka
    elif method == "kernel":
        fn = kernel_cka
    else:
        raise ValueError(f"method must be 'linear' or 'kernel', got {method!r}")

    labels = list(representations.keys())
    k = len(labels)
    matrix = np.eye(k, dtype=np.float64)

    for i in range(k):
        for j in range(i + 1, k):
            val = fn(representations[labels[i]], representations[labels[j]], **kwargs)
            matrix[i, j] = val
            matrix[j, i] = val

    return {"matrix": matrix, "labels": labels}
