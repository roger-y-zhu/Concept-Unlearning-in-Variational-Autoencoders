"""
src/unlearning/erasure.py
=========================
Linear concept erasure: the closed-form math for LEACE and INLP, kept separate
from the UnlearnMethod plumbing in methods.py so the fitting functions can be
tested in isolation against the papers' stated guarantees.

Both operate on the latent representation (the encoder's posterior mean mu),
not on weights.  Because each eraser is an affine map r(mu) = P mu + b and the
encoder's final mean head is a biased Linear layer (fc_mu), the eraser can be
FOLDED EXACTLY into the checkpoint:

    fc_mu:  mu = A h + c        =>      A <- P A,   c <- P c + b

after which the modified model *is* the erased model -- every downstream
consumer (encode_dataset_cached, bespoke metrics, CKA, the val loop) sees the
erasure with no special-casing.  See fold_affine_into_fc_mu.

LEACE  (Belrose et al. 2023, arXiv:2306.03819)
----------------------------------------------
Theorem 4.3 gives the eraser in closed form:

    r(x) = P* x + b*,
    P*   = I - W^+ . P_{W Sigma_XZ} . W,
    b*   = E[X] - P* E[X]

where W = (Sigma_XX^{1/2})^+ is the whitening map (so W^+ = Sigma_XX^{1/2}),
and P_{W Sigma_XZ} is the ORTHOGONAL projector onto colsp(W Sigma_XZ).

Guarantee (Thm 3.4 + 4.1): r(X) linearly guards Z -- no linear classifier
with a convex loss beats the trivial constant predictor -- if and only if
Cov(r(X), Z) = 0, which P* achieves exactly (P* Sigma_XZ = 0; verified
symbolically in this module's tests, not just numerically).

Optimality (Thm 4.2): among ALL affine maps achieving that guarantee, P*
minimises E||r(x) - x||^2_M simultaneously for every p.s.d. quadratic norm M
-- the solution is norm-independent.  The practical reading: LEACE is the
gentlest possible linear scrub, so any damage it does to retain quality is a
LOWER BOUND on the collateral cost of linear guardedness.

INLP  (Ravfogel et al. 2020, arXiv:2004.07667)
----------------------------------------------
Algorithm 1: repeatedly train a linear concept classifier w_i on the current
(projected) representation, then project onto the intersection of the
nullspaces of all classifiers so far; iterate.  Composition here follows the
authors' reference implementation rather than naive projector multiplication:
the product of individual nullspace projectors is not itself a projector, so
P_i is recomputed each round as I - U U^T with U an orthonormal basis of
span{w_1..w_i} (projection onto the intersection of nullspaces).

Contrasts with LEACE worth keeping straight when reporting:
  * INLP removes one rank per iteration (binary concept) and only guards
    against the classifiers it has already seen; LEACE removes exactly
    rank(Sigma_XZ) (= 1 for a binary concept) and guards against ALL linear
    classifiers in one shot.
  * INLP is a pure linear projection (no bias), so it zeroes classifier
    DIRECTIONS but does not equalise class means exactly; LEACE is affine and
    equalises the class-conditional means exactly (Thm 3.4).
  * Neither says anything about nonlinear probes -- that is precisely what
    src/evaluation/recovery.py's probe-recovery curves measure afterwards.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["fit_leace", "fit_inlp", "erase", "fold_affine_into_fc_mu"]


# ---------------------------------------------------------------------------
# Shared linear algebra
# ---------------------------------------------------------------------------

def _psd_sqrt_and_pinv_sqrt(sigma: torch.Tensor, *, rtol: float = 1e-10
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """(Sigma^{1/2}, (Sigma^{1/2})^+) for symmetric PSD Sigma, via eigh.

    Eigenvalues below rtol * max are treated as zero (pseudo-inverse), which
    is the rank-deficient handling the paper prescribes -- a collapsed latent
    dimension must not blow up the whitening map.
    """
    evals, U = torch.linalg.eigh(sigma)
    evals = evals.clamp_min(0.0)
    top = float(evals.max())
    keep = evals > top * rtol if top > 0 else torch.zeros_like(evals, dtype=torch.bool)
    Us, s = U[:, keep], evals[keep]
    sqrt = (Us * s.sqrt()) @ Us.T           # Sigma^{1/2}
    pinv_sqrt = (Us / s.sqrt()) @ Us.T      # (Sigma^{1/2})^+
    return sqrt, pinv_sqrt


def _as_x_z(x, z) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardise inputs: x -> (N, d) float64 tensor, z -> (N, k) float64.

    A 1-D z is treated as a single binary/real concept column.  One-hot input
    is accepted as-is; centering inside the covariance handles E[Z] so no
    separate label normalisation is required (paper section 4.1).
    """
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).double()
    z = torch.as_tensor(np.asarray(z) if not torch.is_tensor(z) else z).double()
    if z.ndim == 1:
        z = z.unsqueeze(1)
    if x.ndim != 2 or z.shape[0] != x.shape[0]:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)}, z {tuple(z.shape)}")
    return x, z


# ---------------------------------------------------------------------------
# LEACE
# ---------------------------------------------------------------------------

def fit_leace(x, z, *, rtol: float = 1e-10, svd_tol: float = 0.01
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the LEACE eraser r(v) = P v + b  (Belrose et al. 2023, Thm 4.3).

    Parameters
    ----------
    x : (N, d) representations (posterior means here).
    z : (N,) binary concept labels in {0, 1}, or (N, k) one-hot / real.
    svd_tol : absolute floor on the singular values of the WHITENED
        cross-covariance W Sigma_XZ that count as real concept directions.
        Matches the reference implementation's default (0.01): whitened X has
        unit scale and a binary z has sd <= 0.5, so genuinely correlated
        directions sit orders of magnitude above this, while directions born
        of float roundoff (~1e-17) sit orders below -- erasing those would
        scrub a random direction for no informational gain.  svd_tol=0
        recovers the strict paper formula (erase colsp(Sigma_XZ) whatever its
        magnitude).

    Returns
    -------
    (P, b) as float64 tensors, P (d, d), b (d,).

    Exactness notes, matching the paper:
      * P Sigma_XZ = 0 exactly (up to float64 roundoff): colsp(Sigma_XZ) lies
        inside range(Sigma_XX) -- both are spans of the centered data -- so
        the W^+ P W sandwich removes it entirely.  This is the necessary and
        sufficient condition of Thm 3.4.
      * b = mean - P mean makes the map affine around the feature mean, which
        is what equalises the class-conditional means rather than merely
        zeroing a direction.
      * P is idempotent (an oblique projection): P^2 = P.
    """
    x, z = _as_x_z(x, z)
    n, d = x.shape

    mean_x = x.mean(0)
    xc = x - mean_x
    zc = z - z.mean(0)

    sigma_xx = (xc.T @ xc) / max(n - 1, 1)
    sigma_xz = (xc.T @ zc) / max(n - 1, 1)                       # (d, k)

    sqrt, w = _psd_sqrt_and_pinv_sqrt(sigma_xx, rtol=rtol)       # W^+ = sqrt, W = pinv sqrt

    m = w @ sigma_xz                                             # whitened cross-cov (d, k)
    # Orthonormal basis Q of colsp(M): P_{W Sigma_XZ} = Q Q^T.  Rank cutoff is
    # the max of the relative pinv cutoff and the absolute svd_tol floor (see
    # the parameter docstring for why an absolute floor is needed at all).
    q, sv, _ = torch.linalg.svd(m, full_matrices=False)
    if sv.numel() == 0 or float(sv.max()) == 0.0:
        rank = 0
    else:
        rel = float(sv.max()) * max(m.shape) * torch.finfo(torch.float64).eps
        rank = int((sv > max(rel, float(svd_tol))).sum())
    q = q[:, :rank]

    eye = torch.eye(d, dtype=torch.float64)
    p = eye - sqrt @ q @ (q.T @ w)                               # I - W^+ QQ^T W
    b = mean_x - p @ mean_x
    return p, b


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------

def fit_inlp(x, z, *, n_iter: int = 8, seed: int = 0, max_iter: int = 1000
             ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Fit the INLP projection (Ravfogel et al. 2020, Algorithm 1).

    Each round trains a logistic-regression concept classifier on the
    currently-projected representation, then re-forms P as the projection
    onto the intersection of all classifier nullspaces so far
    (I - U U^T over an orthonormal basis of span{w_1..w_i} -- the authors'
    own composition, since multiplying individual projectors is not itself
    a projection).

    Returns
    -------
    (P, b, aurocs):
        P        (d, d) float64 projection (symmetric, idempotent),
        b        zeros(d) -- INLP is linear, not affine; kept for interface
                 parity with fit_leace,
        aurocs   per-iteration training AUROC of the classifier *before* its
                 direction was removed.  A healthy run decays toward 0.5;
                 the paper stops when the classifier reaches chance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    x, z = _as_x_z(x, z)
    if z.shape[1] != 1:
        raise ValueError("this INLP port is binary-concept only; pass z as (N,)")
    X = x.cpu().numpy()
    y = z.squeeze(1).cpu().numpy().astype(int)
    d = X.shape[1]

    ws: list[np.ndarray] = []
    P = np.eye(d)
    aurocs: list[float] = []

    for _ in range(int(n_iter)):
        Xp = X @ P.T
        clf = LogisticRegression(max_iter=max_iter, random_state=seed,
                                 class_weight="balanced")
        clf.fit(Xp, y)
        proba = clf.predict_proba(Xp)[:, 1]
        aurocs.append(float(roc_auc_score(y, proba)))

        w = clf.coef_[0].astype(np.float64)
        norm = float(np.linalg.norm(w))
        if norm < 1e-12:
            break                                   # nothing left to remove
        ws.append(w / norm)

        wmat = np.stack(ws)                          # (i, d) removed directions
        u, sv, _ = np.linalg.svd(wmat.T, full_matrices=False)
        cutoff = sv.max() * max(wmat.shape) * np.finfo(np.float64).eps
        u = u[:, : int((sv > cutoff).sum())]
        P = np.eye(d) - u @ u.T

    return (torch.from_numpy(P),
            torch.zeros(d, dtype=torch.float64),
            aurocs)


# ---------------------------------------------------------------------------
# Applying an eraser
# ---------------------------------------------------------------------------

def erase(x, p: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply r(x) = P x + b row-wise.  Accepts (N, d) or (d,)."""
    x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x).double()
    single = x.ndim == 1
    if single:
        x = x.unsqueeze(0)
    out = x @ p.T + b
    return out.squeeze(0) if single else out


@torch.no_grad()
def fold_affine_into_fc_mu(model, p: torch.Tensor, b: torch.Tensor) -> None:
    """Compose r(mu) = P mu + b into the encoder's mean head, in place.

    fc_mu computes mu = A h + c, so the composed head is

        P (A h + c) + b = (P A) h + (P c + b)

    -- an EXACT reparameterisation, not an approximation: the folded model's
    encode() equals post-hoc erasure of the original encode() to float32
    roundoff (tested).  fc_std is untouched: these erasers are defined on the
    representation the metrics read (mu), and sigma keeps its own path.  If
    concept information hides in sigma, probing mu cannot see it either way
    -- see probing.sample_z's measured null on dSprites.
    """
    lin = model.fc_mu
    w, c = lin.weight, lin.bias
    p_ = p.to(dtype=w.dtype, device=w.device)
    b_ = b.to(dtype=w.dtype, device=w.device)
    lin.weight.copy_(p_ @ w)
    lin.bias.copy_(p_ @ c + b_)
