"""
tests/test_erasure.py
=====================
LEACE and INLP (src/unlearning/erasure.py + the method classes).

These tests pin the implementations to the PAPERS' stated guarantees, not to
"it runs":

  LEACE (Belrose et al. 2023):
    * Thm 3.4  -- after erasure the class-conditional means coincide EXACTLY
                  and Cov(r(X), Z) = 0, to float64 roundoff.
    * Dfn 2.3  -- a retrained linear probe scores AUROC ~ 0.5 (guardedness),
                  while an MLP still recovers planted NONLINEAR structure:
                  the guarantee is linear-only and the test would catch an
                  implementation that accidentally erased more than that.
    * Thm 4.2  -- least-squares optimality: LEACE moves the data no more than
                  the obvious alternative eraser (orthogonal projection of
                  the mean-difference direction), strictly less on
                  anisotropic data.
    * P is an oblique projection (P^2 = P), so erasing twice = erasing once.
    * Cross-check against the authors' reference implementation
      (pip: concept-erasure) when it is installed; skipped otherwise.

  INLP (Ravfogel et al. 2020):
    * drives a RETRAINED linear probe to chance on separable data,
    * removes exactly one rank per iteration,
    * produces a symmetric idempotent projection.

  Fold-into-fc_mu:
    * the folded model's encode() equals post-hoc erasure of the original
      encode() (float32 tolerance), std path untouched, and NO parameter
      outside fc_mu changes -- the fold is a reparameterisation, not training.
"""

import numpy as np
import pytest
import torch

from src.unlearning.erasure import erase, fit_inlp, fit_leace, fold_affine_into_fc_mu
from src.unlearning.methods import METHODS, INLPErase, LeaceErase
from src.evaluation.probing import linear_probe, mlp_probe

from test_unlearning_loop import make_args

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Synthetic data with a planted, linearly decodable concept
# ---------------------------------------------------------------------------

def planted_data(n=4000, d=8, gap=2.0, seed=0, aniso=True):
    """Anisotropic Gaussian blobs whose means differ along a skewed direction.

    Anisotropy matters: on isotropic data LEACE coincides with the orthogonal
    mean-difference projection and the strict-optimality test would be vacuous.
    """
    rng = np.random.RandomState(seed)
    if aniso:
        a = rng.randn(d, d)
        cov_half = a @ np.diag(np.linspace(0.2, 3.0, d)) @ np.linalg.inv(a)
    else:
        cov_half = np.eye(d)
    x = rng.randn(n, d) @ cov_half.T
    z = (rng.rand(n) < 0.4).astype(int)
    shift = np.zeros(d)
    shift[0], shift[1] = gap, -0.5 * gap
    x = x + np.outer(z, shift)
    return torch.from_numpy(x), torch.from_numpy(z.astype(float))


def xor_data(n=4000, seed=0, leak=1.0):
    """Concept = XOR of the signs of dims 0/1 (nonlinear), plus a linear leak
    on dim 2.  LEACE must kill the leak; the XOR structure must survive."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 6)
    z = ((x[:, 0] * x[:, 1]) > 0).astype(int)
    x[:, 2] += leak * z
    return torch.from_numpy(x), torch.from_numpy(z.astype(float))


# ---------------------------------------------------------------------------
# LEACE -- the paper's guarantees
# ---------------------------------------------------------------------------

class TestLeaceGuarantees:
    def test_class_conditional_means_coincide_exactly(self):
        x, z = planted_data()
        p, b = fit_leace(x, z)
        xe = erase(x, p, b)
        y = z.bool()
        gap = (xe[y].mean(0) - xe[~y].mean(0)).norm().item()
        assert gap < 1e-8, f"Thm 3.4 violated: residual class-mean gap {gap:.3e}"

    def test_cross_covariance_is_zero(self):
        x, z = planted_data(seed=1)
        p, b = fit_leace(x, z)
        xe = erase(x, p, b)
        xc = xe - xe.mean(0)
        zc = (z - z.mean()).unsqueeze(1)
        sigma_xz = (xc.T @ zc / (len(x) - 1)).abs().max().item()
        assert sigma_xz < 1e-9, f"Cov(r(X), Z) not zero: {sigma_xz:.3e}"

    def test_linear_probe_falls_to_chance_mlp_recovers_nonlinear(self):
        x, z = xor_data()
        y = z.numpy().astype(int)

        # Pre-erasure the leak makes the concept linearly decodable.
        xn = x.numpy()
        pre = linear_probe(xn[:3000], y[:3000], xn[3000:], y[3000:])
        assert pre["auroc"] > 0.75, "test setup broken: no linear leak planted"

        p, b = fit_leace(x, z)
        xe = erase(x, p, b).numpy()
        post_lin = linear_probe(xe[:3000], y[:3000], xe[3000:], y[3000:])
        # One-sided on purpose: the guarantee is "cannot BEAT chance".  A
        # retrained probe fitting pure noise can land below 0.5 on held-out
        # data (observed: 0.438); that is not concept recovery.
        assert post_lin["auroc"] < 0.55, (
            f"linear guardedness violated: retrained linear AUROC "
            f"{post_lin['auroc']:.3f} beats chance")

        # The XOR structure is nonlinear and must survive -- LEACE guards
        # against linear probes ONLY, and erasing more would be a bug too.
        post_mlp = mlp_probe(xe[:3000], y[:3000], xe[3000:], y[3000:],
                             hidden=(32,), balance=False)
        assert post_mlp["auroc"] > 0.8, (
            f"nonlinear structure destroyed (MLP AUROC {post_mlp['auroc']:.3f}) "
            "-- eraser removed more than the linear guarantee allows")

    def test_idempotent(self):
        x, z = planted_data(seed=2)
        p, b = fit_leace(x, z)
        once = erase(x, p, b)
        twice = erase(once, p, b)
        assert torch.allclose(once, twice, atol=1e-8)

    def test_rank_erased_is_one_for_binary_concept(self):
        x, z = planted_data(seed=3)
        p, _ = fit_leace(x, z)
        rank_erased = p.shape[0] - float(torch.diagonal(p).sum())
        assert abs(rank_erased - 1.0) < 1e-6

    def test_exactly_uncorrelated_concept_yields_identity(self):
        # Duplicate every point with both labels: sample Sigma_XZ is exactly 0,
        # so the paper's eraser must be the identity map.
        rng = np.random.RandomState(4)
        base = torch.from_numpy(rng.randn(500, 5))
        x = torch.cat([base, base])
        z = torch.cat([torch.zeros(500), torch.ones(500)])
        p, b = fit_leace(x, z)
        assert torch.allclose(p, torch.eye(5, dtype=torch.float64), atol=1e-10)
        assert b.abs().max().item() < 1e-10

    def test_least_squares_optimality_vs_orthogonal_projection(self):
        """Thm 4.2: among affine maps with Cov(r(X),Z)=0, LEACE moves X least.

        The comparison eraser projects out the mean-difference direction
        orthogonally (with its own optimal bias).  For a binary concept that
        also achieves guardedness -- Sigma_XZ is proportional to the mean
        difference -- so it is a valid competitor, and on anisotropic data it
        must move the data strictly more than LEACE's oblique projection.
        """
        x, z = planted_data(n=6000, seed=5, aniso=True)
        y = z.bool()

        p, b = fit_leace(x, z)
        cost_leace = (erase(x, p, b) - x).pow(2).sum(1).mean().item()

        d = (x[y].mean(0) - x[~y].mean(0)).double()
        u = (d / d.norm()).unsqueeze(1)
        p_orth = torch.eye(x.shape[1], dtype=torch.float64) - u @ u.T
        mean_x = x.mean(0)
        b_orth = mean_x - p_orth @ mean_x
        xe_orth = x @ p_orth.T + b_orth

        # Sanity: the competitor really does guard (means coincide).
        gap = (xe_orth[y].mean(0) - xe_orth[~y].mean(0)).norm().item()
        assert gap < 1e-8

        cost_orth = (xe_orth - x).pow(2).sum(1).mean().item()
        assert cost_leace <= cost_orth + 1e-12
        assert cost_leace < 0.99 * cost_orth, (
            f"expected strictly smaller movement on anisotropic data: "
            f"LEACE {cost_leace:.6f} vs orthogonal {cost_orth:.6f}")

    def test_matches_reference_implementation(self):
        """Cross-check against the authors' own package (EleutherAI
        concept-erasure).  Skipped when not installed."""
        ce = pytest.importorskip("concept_erasure")
        x, z = planted_data(n=3000, seed=6)

        # shrinkage / constrain_cov_trace are the package's additions beyond
        # the paper's Thm 4.3; disabled so both sides compute the same object.
        eraser = ce.LeaceEraser.fit(x, z, method="leace",
                                    shrinkage=False, constrain_cov_trace=False)
        ref = eraser(x)

        p, b = fit_leace(x, z)
        ours = erase(x, p, b)
        dev = (ours - ref).abs().max().item()
        assert dev < 1e-10, f"deviation from reference implementation: {dev:.3e}"


# ---------------------------------------------------------------------------
# INLP
# ---------------------------------------------------------------------------

class TestInlp:
    def test_drives_retrained_linear_probe_to_chance(self):
        x, z = planted_data(n=4000, d=6, seed=7)
        y = z.numpy().astype(int)
        p, b, _ = fit_inlp(x, z, n_iter=3)
        xe = erase(x, p, b).numpy()
        post = linear_probe(xe[:3000], y[:3000], xe[3000:], y[3000:])
        assert abs(post["auroc"] - 0.5) < 0.06, (
            f"retrained linear AUROC {post['auroc']:.3f} after 3 INLP rounds")

    def test_removes_one_rank_per_iteration(self):
        x, z = planted_data(n=3000, d=6, seed=8)
        for k in (1, 2, 3):
            p, _, _ = fit_inlp(x, z, n_iter=k)
            rank = float(torch.diagonal(p).sum())
            assert abs(rank - (6 - k)) < 1e-6, f"k={k}: trace(P)={rank}"

    def test_projection_is_symmetric_and_idempotent(self):
        x, z = planted_data(n=2000, d=5, seed=9)
        p, _, _ = fit_inlp(x, z, n_iter=2)
        assert torch.allclose(p, p.T, atol=1e-10)
        assert torch.allclose(p @ p, p, atol=1e-10)

    def test_classifier_auroc_history_decays(self):
        x, z = planted_data(n=4000, d=6, seed=10)
        _, _, aurocs = fit_inlp(x, z, n_iter=4)
        assert aurocs[0] > 0.85, "first-round classifier should see the concept"
        assert aurocs[-1] < 0.7, f"AUROC path did not decay: {aurocs}"


# ---------------------------------------------------------------------------
# Folding into the model
# ---------------------------------------------------------------------------

class TestFoldIntoModel:
    def _mus(self, model, x):
        # eval() is load-bearing: a train-mode forward pass updates BatchNorm
        # running statistics, which would make "no parameter outside fc_mu
        # changed" fail for reasons unrelated to the fold -- the exact bug
        # class bn_stats_frozen exists for.
        model.eval()
        with torch.no_grad():
            mu, std = model.encode(x)
        return mu, std

    def test_folded_encode_equals_posthoc_erasure(self, tiny_vae_bce):
        model = tiny_vae_bce
        x = torch.rand(32, 1, 64, 64).round()
        mu0, std0 = self._mus(model, x)

        z = (torch.arange(32) % 3 == 0).double()
        p, b = fit_leace(mu0.double(), z)
        fold_affine_into_fc_mu(model, p, b)

        mu1, std1 = self._mus(model, x)
        want = erase(mu0.double(), p, b).float()
        assert torch.allclose(mu1, want, atol=1e-4), (
            f"fold is not exact: max dev {(mu1 - want).abs().max().item():.3e}")
        assert torch.allclose(std0, std1), "fc_std path must be untouched"

    def test_fold_touches_only_fc_mu(self, tiny_vae_bce):
        model = tiny_vae_bce
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.rand(16, 1, 64, 64).round()
        mu0, _ = self._mus(model, x)
        p, b = fit_leace(mu0.double(), (torch.arange(16) % 2).double())
        fold_affine_into_fc_mu(model, p, b)

        for k, v in model.state_dict().items():
            if k.startswith("fc_mu."):
                assert not torch.equal(before[k], v), f"{k} should have changed"
            else:
                assert torch.equal(before[k], v), f"{k} must not change"


# ---------------------------------------------------------------------------
# Method classes + shared loop
# ---------------------------------------------------------------------------

class TestMethodIntegration:
    def test_registered(self):
        assert isinstance(METHODS["leace"], LeaceErase)
        assert isinstance(METHODS["inlp"], INLPErase)
        for name in ("leace", "inlp"):
            assert METHODS[name].one_shot, f"{name} must be one-shot"
            assert METHODS[name].train_mode is False

    def test_run_names(self):
        leace, inlp = METHODS["leace"], METHODS["inlp"]
        assert leace.run_name(make_args("leace")) == "leace_ep1"
        assert leace.run_name(make_args("leace", fit_samples=500)) == "leace_ep1_fs500"
        assert inlp.run_name(make_args("inlp", n_iter=4)) == "k4_ep1"
        assert inlp.run_name(make_args("inlp", n_iter=4, inlp_seed=2)) == "k4_ep1_s2"

    def test_leace_through_the_loop_equalises_train_means(self, tiny_spec):
        from src.unlearning.loop import run_unlearning

        out_dir = run_unlearning(tiny_spec, METHODS["leace"],
                                 make_args("leace"), CPU)
        assert out_dir is not None and (out_dir / "final.pt").exists()

        ckpt = torch.load(out_dir / "final.pt", map_location="cpu",
                          weights_only=True)
        from src.models.vae import ConvVAE
        model = ConvVAE(ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        forget_loader, retain_loader = tiny_spec.concept_loaders()

        def mean_mu(loader):
            mus = []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    mus.append(model.encode(x)[0])
            return torch.cat(mus).mean(0)

        gap = (mean_mu(forget_loader) - mean_mu(retain_loader)).norm().item()
        # float32 model + tiny N: exact-in-float64 becomes ~1e-3 here.
        assert gap < 5e-3, f"folded LEACE left class-mean gap {gap:.4f}"

    def test_inlp_through_the_loop(self, tiny_spec):
        from src.unlearning.loop import run_unlearning

        out_dir = run_unlearning(tiny_spec, METHODS["inlp"],
                                 make_args("inlp"), CPU)
        assert out_dir is not None and (out_dir / "final.pt").exists()
        assert out_dir.name == "k2_ep1"


def _probe_auroc(xe, z):
    """Held-out linear-probe AUROC on an (already erased) representation."""
    xn = xe.numpy() if hasattr(xe, "numpy") else xe
    y = z.numpy().astype(int) if hasattr(z, "numpy") else z.astype(int)
    n = len(xn) // 2
    return linear_probe(xn[:n], y[:n], xn[n:], y[n:])["auroc"]


class TestPartialErasureDose:
    """--erase-strength interpolates identity -> LEACE.

    The endpoints must be exact (a=0 is a no-op, a=1 is the theorem's map), and
    the middle must be honestly labelled: partial erasure is NOT linearly
    guarded, and reporting it as though it were would misuse Thm 3.4.
    """

    @staticmethod
    def _interp(p, b, a):
        """Mirrors LeaceErase._fit's interpolation."""
        eye = torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
        return (1.0 - a) * eye + a * p, a * b

    def test_strength_zero_is_exactly_the_identity(self):
        x, z = planted_data()
        p, b = fit_leace(x, z)
        p0, b0 = self._interp(p, b, 0.0)
        assert torch.allclose(p0, torch.eye(p.shape[0], dtype=p.dtype), atol=1e-12)
        assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-12)
        # and therefore leaves the representation untouched
        assert torch.allclose(erase(x, p0, b0), x, atol=1e-10)

    def test_strength_one_is_exactly_full_leace(self):
        x, z = planted_data()
        p, b = fit_leace(x, z)
        p1, b1 = self._interp(p, b, 1.0)
        assert torch.allclose(p1, p, atol=1e-12)
        assert torch.allclose(b1, b, atol=1e-12)

    def test_residual_cross_covariance_scales_as_one_minus_a(self):
        """The dose is linear in the thing the guarantee is about.

        Cov(r_a(x), z) = (1-a) Cov(x, z), so `a` reads directly as "fraction of
        the linear concept signal removed" rather than as an opaque knob.
        """
        x, z = planted_data()
        p, b = fit_leace(x, z)
        zc = (z - z.mean(0)).reshape(-1, 1)

        def xcov(xe):
            xc = xe - xe.mean(0)
            return float((xc.T @ zc).abs().max()) / max(len(xe) - 1, 1)

        base = xcov(x)
        for a in (0.25, 0.5, 0.75):
            pa, ba = self._interp(p, b, a)
            assert xcov(erase(x, pa, ba)) == pytest.approx((1 - a) * base, rel=1e-6)

    def test_partial_erasure_is_not_guarded(self):
        """A probe must still beat chance at a<1 -- otherwise the dose curve
        would have nothing to trace and `a` would be meaningless."""
        x, z = planted_data()
        p, b = fit_leace(x, z)
        pa, ba = self._interp(p, b, 0.5)
        auroc = _probe_auroc(erase(x, pa, ba), z)
        assert auroc > 0.55, f"half-erased data should stay linearly detectable, got {auroc}"

    def test_dose_is_monotone_in_detectability(self):
        """More erasure must not make the concept easier to read off."""
        x, z = planted_data()
        p, b = fit_leace(x, z)
        aurocs = []
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            pa, ba = self._interp(p, b, a)
            aurocs.append(_probe_auroc(erase(x, pa, ba), z))
        assert all(aurocs[i] >= aurocs[i + 1] - 0.02 for i in range(len(aurocs) - 1)), aurocs
        assert aurocs[0] > aurocs[-1] + 0.05, aurocs
