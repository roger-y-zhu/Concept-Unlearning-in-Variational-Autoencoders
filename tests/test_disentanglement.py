"""
tests/test_disentanglement.py
===============================
MIG and DCI must give the answer the disentanglement literature defines them
to give on the two textbook cases (a clean one-dim-per-factor code, and a
maximally entangled one), or numbers computed from them on real checkpoints
are not trustworthy comparisons to that literature.
"""

import numpy as np
import pytest

from src.evaluation.disentanglement import mutual_info_gap, dci_disentanglement


def _rng(seed=0):
    return np.random.default_rng(seed)


def disentangled_latents(n=4000, seed=0):
    """dim0 <-> factorA, dim1 <-> factorB, dims 2/3 pure noise.

    Small additive noise on the informative dims so the mapping is not
    literally the identity (which would trivially inflate MI estimates) but
    still overwhelmingly informative.
    """
    g = _rng(seed)
    a = g.integers(0, 5, n).astype(float)
    b = g.integers(0, 5, n).astype(float)
    mu = np.stack([
        a + g.normal(0, 0.05, n),
        b + g.normal(0, 0.05, n),
        g.normal(0, 1, n),
        g.normal(0, 1, n),
    ], axis=1)
    return mu, {"factorA": a, "factorB": b}


def entangled_latents(n=4000, seed=0):
    """Both factors spread equally over BOTH of two informative dims (a
    45-degree rotation of the disentangled case) -- the textbook "bad" code."""
    g = _rng(seed)
    a = g.integers(0, 5, n).astype(float)
    b = g.integers(0, 5, n).astype(float)
    d0 = (a + b) / 2 + g.normal(0, 0.05, n)
    d1 = (a - b) / 2 + g.normal(0, 0.05, n)
    mu = np.stack([d0, d1, g.normal(0, 1, n), g.normal(0, 1, n)], axis=1)
    return mu, {"factorA": a, "factorB": b}


class TestMIG:
    def test_disentangled_code_scores_high(self):
        mu, factors = disentangled_latents()
        out = mutual_info_gap(mu, factors, n_bins=10)
        assert out["mig"] > 0.7, out

    def test_entangled_code_scores_low(self):
        mu, factors = entangled_latents()
        out = mutual_info_gap(mu, factors, n_bins=10)
        # both factors are smeared roughly evenly over dims 0 and 1, so the
        # top-2 gap for each factor should be small
        assert out["mig"] < 0.35, out

    def test_disentangled_beats_entangled(self):
        """The comparative claim is what the metric is FOR -- test it directly
        rather than trusting two absolute thresholds to imply it."""
        mu_d, f_d = disentangled_latents(seed=1)
        mu_e, f_e = entangled_latents(seed=1)
        assert mutual_info_gap(mu_d, f_d)["mig"] > mutual_info_gap(mu_e, f_e)["mig"]

    def test_top_dim_is_the_informative_one(self):
        mu, factors = disentangled_latents()
        out = mutual_info_gap(mu, factors, n_bins=10)
        assert out["per_factor"]["factorA"]["top_dim"] == 0
        assert out["per_factor"]["factorB"]["top_dim"] == 1

    def test_pure_noise_factor_scores_near_zero(self):
        """A factor independent of every latent dim must not score high by
        the top-2-gap noise alone."""
        g = _rng(2)
        mu = g.normal(0, 1, (4000, 4))
        v = g.integers(0, 5, 4000).astype(float)   # independent of mu
        out = mutual_info_gap(mu, {"junk": v}, n_bins=10)
        assert out["mig"] < 0.15, out

    def test_mig_is_bounded(self):
        mu, factors = disentangled_latents()
        out = mutual_info_gap(mu, factors, n_bins=10)
        for v in out["per_factor"].values():
            assert -1e-6 <= v["mig"] <= 1.0 + 1e-6


class TestDCI:
    def test_disentangled_code_scores_high(self):
        mu, factors = disentangled_latents()
        out = dci_disentanglement(mu, factors, n_estimators=30)
        assert out["disentanglement"] > 0.7, out

    def test_entangled_code_scores_low(self):
        mu, factors = entangled_latents()
        out = dci_disentanglement(mu, factors, n_estimators=30)
        assert out["disentanglement"] < 0.5, out

    def test_disentangled_beats_entangled(self):
        mu_d, f_d = disentangled_latents(seed=1)
        mu_e, f_e = entangled_latents(seed=1)
        d1 = dci_disentanglement(mu_d, f_d, n_estimators=30)["disentanglement"]
        d2 = dci_disentanglement(mu_e, f_e, n_estimators=30)["disentanglement"]
        assert d1 > d2

    def test_unused_dimension_does_not_crash_or_inflate(self):
        """A constant dimension has an undefined importance distribution, not
        a perfectly-disentangled one -- it must score 0 and carry 0 weight."""
        mu, factors = disentangled_latents()
        mu = np.concatenate([mu, np.zeros((len(mu), 1))], axis=1)  # dead dim
        out = dci_disentanglement(mu, factors, n_estimators=30)
        assert out["per_dim_disentanglement"][-1] == 0.0
        assert out["rho"][-1] == pytest.approx(0.0, abs=1e-6)
        assert out["disentanglement"] > 0.7   # unaffected by the dead dim

    def test_rho_sums_to_one(self):
        mu, factors = disentangled_latents()
        out = dci_disentanglement(mu, factors, n_estimators=30)
        assert sum(out["rho"]) == pytest.approx(1.0, abs=1e-6)

    def test_informative_dims_get_high_disentanglement(self):
        mu, factors = disentangled_latents()
        out = dci_disentanglement(mu, factors, n_estimators=30)
        assert out["per_dim_disentanglement"][0] > 0.8
        assert out["per_dim_disentanglement"][1] > 0.8
