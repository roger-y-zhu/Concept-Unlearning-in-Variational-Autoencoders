import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def knn_concept_cohesion(mu, labels, k=15, sample_n=None, seed=0):
    """
    Measure local neighbourhood purity for the concept class.

    For each concept-positive sample, computes the fraction of its k nearest
    neighbours that are also concept-positive, then compares to the random
    baseline (global positive class proportion).

    Args:
        mu:       (N, D) latent mu vectors
        labels:   (N,) binary concept labels (0=negative, 1=positive)
        k:        number of nearest neighbours
        sample_n: subsample before computing (None = use all)
        seed:     RNG seed for subsampling

    Returns dict:
        cohesion   mean same-label neighbour fraction for positive samples
        baseline   positive class proportion (random expectation)
        lift       cohesion - baseline
        lift_norm  lift / (1 - baseline), in [0, 1]
    """
    rng = np.random.RandomState(seed)

    scaler = StandardScaler()
    X = scaler.fit_transform(mu)

    if sample_n is not None and len(X) > sample_n:
        idx = rng.choice(len(X), sample_n, replace=False)
        X, labels = X[idx], labels[idx]

    baseline = float(labels.mean())

    # fit on all points, query only positive samples for efficiency
    pos_mask = labels == 1
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(X)

    # k+1 because the first result is the point itself (distance=0)
    _, indices = nn.kneighbors(X[pos_mask])
    neighbour_labels = labels[indices[:, 1:]]  # (n_pos, k), skip self

    cohesion = float(neighbour_labels.mean())
    lift = cohesion - baseline
    lift_norm = float(lift / (1 - baseline)) if baseline < 1.0 else 0.0

    return {
        "cohesion":  cohesion,
        "baseline":  baseline,
        "lift":      lift,
        "lift_norm": lift_norm,
    }