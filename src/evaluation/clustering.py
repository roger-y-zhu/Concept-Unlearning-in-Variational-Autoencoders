import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def _purity(km_labels, true_labels):
    """Fraction of majority-class samples across all k-means clusters."""
    total = len(true_labels)
    correct = 0
    for c in np.unique(km_labels):
        mask = km_labels == c
        correct += np.bincount(true_labels[mask]).max()
    return correct / total


def cluster_metrics(mu, labels, sample_n=10_000, seed=0):
    """
    Measure concept cluster separation in latent space using ground-truth labels.

    Args:
        mu:       (N, D) latent mu vectors
        labels:   (N,) binary concept labels (0=negative, 1=positive)
        sample_n: subsample size for silhouette score (O(n^2) cost)
        seed:     RNG seed for subsampling and k-means

    Returns dict:
        silhouette      higher is better  [-1, 1]
        davies_bouldin  lower is better   [0, inf)
        kmeans_purity   higher is better  [0, 1]
    """
    rng = np.random.RandomState(seed)

    scaler = StandardScaler()
    X = scaler.fit_transform(mu)

    # subsample for silhouette
    n = len(X)
    if n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        X_sub, y_sub = X[idx], labels[idx]
    else:
        X_sub, y_sub = X, labels

    sil = silhouette_score(X_sub, y_sub)
    db  = davies_bouldin_score(X_sub, y_sub)

    # k-means purity on full data
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    km_labels = km.fit_predict(X)
    purity = _purity(km_labels, labels)

    return {
        "silhouette":     float(sil),
        "davies_bouldin": float(db),
        "kmeans_purity":  float(purity),
    }