"""
tests/test_separation.py
=========================
The point of src/evaluation/separation.py is that the notebooks' raw
`direction_norm` is not scale-invariant, so a model that merely rescales its
latent space scores as though it unlearned something.  These tests check that
specific property -- that cohens_d and mahalanobis are unmoved by exactly the
transformations that move direction_norm -- rather than just that the numbers
come out.
"""

import numpy as np
import pytest

from src.evaluation.separation import concept_separation, oracle_gap


def two_blobs(n=400, d=5, sep=1.5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(2 * n, d)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    X[y == 1, 0] += sep
    return X, y


class TestScaleInvariance:
    def test_isotropic_scaling_moves_direction_norm_but_not_the_effect_sizes(self):
        """The exact failure mode the module exists for: shrink the whole latent
        space by 10x and direction_norm drops 10x while nothing is actually
        harder to decode."""
        X, y = two_blobs()
        big = concept_separation(X, y)
        small = concept_separation(0.1 * X, y)

        assert small["direction_norm"] == pytest.approx(0.1 * big["direction_norm"], rel=1e-9)
        assert small["cohens_d"] == pytest.approx(big["cohens_d"], rel=1e-9)
        assert small["mahalanobis"] == pytest.approx(big["mahalanobis"], rel=1e-6)

    def test_mahalanobis_survives_an_anisotropic_map_that_cohens_d_does_not(self):
        """Mahalanobis whitens by the pooled covariance, so ANY invertible
        linear map leaves it alone.  cohens_d only divides by a scalar, so it
        moves -- documenting the difference is the point of having both."""
        X, y = two_blobs()
        rng = np.random.RandomState(1)
        A = rng.randn(X.shape[1], X.shape[1])
        A += X.shape[1] * np.eye(X.shape[1])       # keep it well-conditioned

        base = concept_separation(X, y)
        mapped = concept_separation(X @ A.T, y)

        assert mapped["mahalanobis"] == pytest.approx(base["mahalanobis"], rel=1e-4)
        assert mapped["cohens_d"] != pytest.approx(base["cohens_d"], rel=1e-3)

    def test_translation_changes_nothing(self):
        X, y = two_blobs()
        shifted = concept_separation(X + 37.0, y)
        base = concept_separation(X, y)
        assert shifted["direction_norm"] == pytest.approx(base["direction_norm"], rel=1e-9)
        assert shifted["mahalanobis"] == pytest.approx(base["mahalanobis"], rel=1e-6)


class TestSeparationBasics:
    def test_no_separation_scores_near_zero(self):
        rng = np.random.RandomState(0)
        X = rng.randn(600, 5)
        y = np.concatenate([np.zeros(300), np.ones(300)]).astype(int)
        out = concept_separation(X, y)
        assert out["cohens_d"] < 0.15
        assert out["mahalanobis"] < 0.5

    def test_more_separated_blobs_score_higher(self):
        near = concept_separation(*two_blobs(sep=0.5))
        far = concept_separation(*two_blobs(sep=5.0))
        assert far["cohens_d"] > near["cohens_d"]
        assert far["mahalanobis"] > near["mahalanobis"]

    def test_counts_and_forget_val(self):
        X, y = two_blobs(n=100)
        out = concept_separation(X, y, forget_val=1)
        assert out["n_forget"] == 100 and out["n_retain"] == 100
        # raw dSprites shape classes: 0=square 1=ellipse 2=heart
        y3 = np.where(y == 1, 2, 0)
        assert concept_separation(X, y3, forget_val=2)["n_forget"] == 100

    def test_single_class_raises_rather_than_returning_a_meaningless_number(self):
        X, y = two_blobs()
        with pytest.raises(ValueError):
            concept_separation(X, np.zeros_like(y))

    def test_sigma_ratio_reports_posterior_width(self):
        X, y = two_blobs()
        std = np.ones_like(X)
        std[y == 1] *= 0.5
        out = concept_separation(X, y, std=std)
        assert out["sigma_ratio"] == pytest.approx(0.5, rel=1e-9)
        assert concept_separation(X, y)["sigma_ratio"] is None

    def test_collapsed_latent_dimension_does_not_blow_up_mahalanobis(self):
        """An aggressive unlearning run really does zero out latent dimensions;
        the ridge must keep the inverse finite instead of producing inf/nan."""
        X, y = two_blobs()
        X[:, 2] = 0.0
        out = concept_separation(X, y)
        assert np.isfinite(out["mahalanobis"])


class TestOracleGap:
    def test_matching_the_oracle_scores_zero(self):
        m = {"direction_norm": 0.47, "cohens_d": 0.191}
        assert oracle_gap(m, m) == {"direction_norm": 0.0, "cohens_d": 0.0}

    def test_undershooting_past_the_oracle_is_penalised(self):
        """Synthetic, not measured: a model that separates the concept LESS
        than the oracle does (undershoots past it) must score as farther away,
        not closer -- 'lower is better' would rank it best, which is exactly
        the failure mode oracle_gap exists to prevent. (On the real dSprites
        checkpoints nothing has undershot the oracle this far yet -- see
        src/evaluation/separation.py's docstring for what IS measured there,
        and its note on a since-corrected oracle-checkpoint mixup -- but the
        framing must not depend on that staying true.)"""
        oracle = {"direction_norm": 0.365}
        close = {"direction_norm": 0.30}     # short of the oracle
        overshot = {"direction_norm": 0.05}  # well past it
        assert oracle_gap(overshot, oracle)["direction_norm"] > \
               oracle_gap(close, oracle)["direction_norm"]

    def test_missing_values_are_skipped_not_scored_as_perfect(self):
        m = {"cohens_d": 0.2, "sigma_ratio": None}
        o = {"cohens_d": 0.2, "sigma_ratio": 0.9}
        gap = oracle_gap(m, o)
        assert "sigma_ratio" not in gap
        assert gap["cohens_d"] == 0.0

    def test_non_numeric_keys_are_ignored(self):
        m = {"cohens_d": 0.2, "n_forget": 100}
        o = {"cohens_d": 0.1, "n_forget": 90}
        assert oracle_gap(m, o, keys=["cohens_d"]) == {"cohens_d": pytest.approx(0.1)}
