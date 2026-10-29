"""
tests/test_evaluation.py
=========================
Do the evaluation metrics actually measure what they claim to, on toy examples
where the correct answer is known in advance?

Every test here constructs synthetic latent vectors with a designed property
(identical representations, orthogonal rotation, separated clusters, random
noise) and checks the metric lands where that property demands -- not just
that it runs without crashing.
"""

import numpy as np
import pytest

from src.evaluation.cka import linear_cka, kernel_cka, cka_matrix
from src.evaluation.probing import linear_probe, mlp_probe, encode_dataset
from src.evaluation.knn import knn_concept_cohesion
from src.evaluation.clustering import cluster_metrics


# ---------------------------------------------------------------------------
# Toy latent spaces
# ---------------------------------------------------------------------------

def random_orthogonal(d, seed=0):
    """A genuine d x d orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(d, d))
    return q


def two_blobs(n_per_class=200, d=6, sep=6.0, seed=0):
    """Two well-separated Gaussian blobs -- a concept that IS linearly decodable
    and DOES cluster, for testing the "metric detects a real signal" direction."""
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n_per_class, d) - sep / 2
    x1 = rng.randn(n_per_class, d) + sep / 2
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return X, y


def pure_noise(n_per_class=200, d=6, seed=0):
    """Labels uncorrelated with the features -- a concept that is NOT decodable,
    for testing the "metric reports null when there's nothing there" direction."""
    rng = np.random.RandomState(seed)
    X = rng.randn(2 * n_per_class, d)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

class TestCKA:
    def test_identical_representations_score_one(self):
        X, _ = two_blobs(seed=0)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_transform_is_invariant(self):
        """The entire point of CKA over CCA: rotating the representation must
        not change the score."""
        X, _ = two_blobs(seed=0)
        Q = random_orthogonal(X.shape[1], seed=1)
        assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)

    def test_isotropic_scaling_is_invariant(self):
        X, _ = two_blobs(seed=0)
        assert linear_cka(X, 37.0 * X) == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_representations_score_low(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 6)
        Y = rng.randn(300, 6)  # independent noise, same shape, no relationship
        assert linear_cka(X, Y) < 0.3

    def test_kernel_cka_identical_representations_score_near_one(self):
        X, _ = two_blobs(n_per_class=100, seed=0)
        assert kernel_cka(X, X, sample_n=None) == pytest.approx(1.0, abs=1e-3)

    def test_cka_matrix_diagonal_is_one_and_symmetric(self):
        X, _ = two_blobs(seed=0)
        Y, _ = two_blobs(seed=1)
        result = cka_matrix({"a": X, "b": Y, "c": X})
        M = result["matrix"]
        assert np.allclose(np.diag(M), 1.0)
        assert np.allclose(M, M.T)
        # "a" and "c" are literally the same array
        assert M[0, 2] == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_row_counts_raise(self):
        X = np.random.randn(10, 4)
        Y = np.random.randn(11, 4)
        with pytest.raises(AssertionError):
            linear_cka(X, Y)


# ---------------------------------------------------------------------------
# Probing (linear + MLP -- see test_adversarial.py for the separate,
# active adversarial-recovery attack)
# ---------------------------------------------------------------------------

class TestProbing:
    def test_linear_probe_recovers_separable_concept(self):
        X, y = two_blobs(seed=0)
        Xte, yte = two_blobs(seed=1)
        result = linear_probe(X, y, Xte, yte)
        assert result["auroc"] > 0.95

    def test_linear_probe_near_chance_on_unrelated_labels(self):
        X, y = pure_noise(n_per_class=300, seed=0)
        Xte, yte = pure_noise(n_per_class=300, seed=1)
        result = linear_probe(X, y, Xte, yte)
        assert 0.35 < result["auroc"] < 0.65

    def test_mlp_probe_recovers_separable_concept(self):
        X, y = two_blobs(seed=0)
        Xte, yte = two_blobs(seed=1)
        result = mlp_probe(X, y, Xte, yte, max_iter=300)
        assert result["auroc"] > 0.90

    def test_probes_return_proba_for_roc_curves(self):
        """Regression guard: evaluate_baseline.py plots an ROC curve from
        result['proba']; both probes must expose it."""
        X, y = two_blobs(n_per_class=50, seed=0)
        lin = linear_probe(X, y, X, y)
        mlp = mlp_probe(X, y, X, y, max_iter=200)
        assert "proba" in lin and len(lin["proba"]) == len(y)
        assert "proba" in mlp and len(mlp["proba"]) == len(y)

    def test_mlp_probe_balancing_recovers_minority_class(self):
        """At realistic CelebA-scale imbalance (~6.5% positive), an unbalanced
        MLP can win by predicting the majority class -- high accuracy, near-zero
        recall.  balance=True must fix this."""
        rng = np.random.RandomState(0)
        n = 3000
        X = rng.randn(n, 6)
        y = np.zeros(n, dtype=int)
        pos = rng.choice(n, int(n * 0.065), replace=False)
        y[pos] = 1
        X[pos, 0] += 2.0  # inject a real, learnable signal

        split = n // 2
        unbalanced = mlp_probe(X[:split], y[:split], X[split:], y[split:],
                               balance=False, max_iter=200)
        balanced = mlp_probe(X[:split], y[:split], X[split:], y[split:],
                             balance=True, max_iter=200)

        # The core claim: balancing recovers substantially more of the
        # minority class than leaving the 6.5% imbalance untouched. Not
        # pinned to an absolute threshold (MLP training has run-to-run
        # variance), but the gap must be large and in the right direction.
        assert balanced["recall"] > unbalanced["recall"] + 0.15

    def test_encode_dataset_labels_none_for_unlabeled_loader(self):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from src.models.vae import ConvVAE, VAEConfig

        class PlainImageDataset(Dataset):
            """Returns a bare tensor per item, like DSpritesDataset(include_labels=False)
            -- unlike TensorDataset, which always wraps items in a 1-tuple."""
            def __init__(self, imgs):
                self.imgs = imgs
            def __len__(self):
                return len(self.imgs)
            def __getitem__(self, idx):
                return self.imgs[idx]

        model = ConvVAE(VAEConfig(latent_dim=4, base_channels=4, in_channels=1))
        loader = DataLoader(PlainImageDataset(torch.rand(8, 1, 64, 64)), batch_size=4)
        mu, labels = encode_dataset(model, loader, torch.device("cpu"))
        assert mu.shape == (8, 4)
        assert labels is None


# ---------------------------------------------------------------------------
# k-NN concept cohesion
# ---------------------------------------------------------------------------

class TestKNN:
    def test_perfectly_separated_clusters_have_full_cohesion(self):
        X, y = two_blobs(n_per_class=100, sep=20.0, seed=0)  # blobs far enough
        # apart that no cross-cluster point is ever a k-NN
        result = knn_concept_cohesion(X, y, k=10)
        assert result["cohesion"] == pytest.approx(1.0, abs=1e-6)
        assert result["lift_norm"] == pytest.approx(1.0, abs=1e-6)

    def test_random_labels_have_near_baseline_cohesion(self):
        X, y = pure_noise(n_per_class=300, seed=0)
        result = knn_concept_cohesion(X, y, k=15)
        # cohesion should track the base rate (0.5), not be inflated
        assert abs(result["cohesion"] - result["baseline"]) < 0.15

    def test_lift_is_cohesion_minus_baseline(self):
        X, y = two_blobs(n_per_class=100, sep=10.0, seed=0)
        result = knn_concept_cohesion(X, y, k=10)
        assert result["lift"] == pytest.approx(result["cohesion"] - result["baseline"], abs=1e-9)


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

class TestClustering:
    def test_separated_clusters_have_high_silhouette_and_purity(self):
        X, y = two_blobs(n_per_class=150, sep=15.0, seed=0)
        result = cluster_metrics(X, y)
        assert result["silhouette"] > 0.7
        assert result["kmeans_purity"] > 0.95

    def test_overlapping_clusters_have_low_silhouette(self):
        X, y = two_blobs(n_per_class=150, sep=0.1, seed=0)  # barely separated
        result = cluster_metrics(X, y)
        assert result["silhouette"] < 0.3
