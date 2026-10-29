import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def encode_dataset(model, loader, device, return_std=False):
    """
    Encode all batches through the VAE encoder.

    Loader must yield either:
      - plain images (no labels)
      - (images, labels) tuples       ← CelebA labeled
      - (images, latents, classes)    ← dSprites with include_labels=True

    Returns: mu (N, latent_dim) and labels (N,) as numpy arrays.
    Labels are None if loader yields plain images.

    ``return_std=True`` returns (mu, std, labels) instead.  Worth using: a VAE's
    representation is the distribution q(z|x) = N(mu, sigma^2), and every
    unlearning method here constrains only mu -- RMU's forget term is a
    distance on mu, Feature Unlearning does not touch the encoder at all -- so
    sigma is free to drift.  It does: measured on dSprites, mean sigma on the
    forget class relative to the retain class is 0.81 (baseline), 0.89
    (oracle), 0.71 (RMU).  A probe run on mu alone cannot see any concept
    information that has moved into the posterior width.  See sample_z.
    """
    model.eval()
    mus, stds, lbls = [], [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                # dSprites: (img, latents_values, latents_classes)
                imgs = batch[0].to(device)
                cls = batch[2]              # latents_classes, shape (B, 6)
                lbl = cls[:, 1].numpy()     # shape factor: 0=square, 1=ellipse, 2=heart
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # CelebA labeled: (img, label)
                imgs = batch[0].to(device)
                lbl = batch[1].numpy()
            else:
                imgs = batch.to(device)
                lbl = None

            mu, std = model.encode(imgs)
            mus.append(mu.cpu().numpy())
            if return_std:
                stds.append(std.cpu().numpy())
            if lbl is not None:
                lbls.append(lbl)

    mu_matrix = np.concatenate(mus, axis=0)
    labels = np.concatenate(lbls, axis=0) if lbls else None
    if return_std:
        return mu_matrix, np.concatenate(stds, axis=0), labels
    return mu_matrix, labels


def sample_z(mu, std, seed=0):
    """Draw one z ~ N(mu, std^2) per row -- the representation the DECODER
    actually receives.

    Probing mu answers "is the concept in the posterior mean"; probing z
    answers "is the concept in what the model actually passes downstream".
    They can disagree in principle: a method that shrinks the concept's
    separation in mu while leaving sigma large has not removed the
    information, only buried it under sampling noise.

    MEASURED RESULT ON dSprites -- they do not disagree here.  Linear probe
    AUROC on mu vs on a single z draw, 8192 test images:

        baseline  0.6213 / 0.6196      rmu       0.5868 / 0.5873
        oracle    0.6628 / 0.6621      feat_unl  0.6213 / 0.6196

    Differences are under 0.002 everywhere, so sigma is not hiding concept
    information that mu-based reporting misses, and the mu-only metrics are
    safe to report as-is on this dataset.  This is a null result worth
    stating rather than a caveat worth carrying: it was checked, not assumed.
    Re-check on CelebA before extending the claim there -- sigma_f/sigma_r
    varies by method (0.71-0.89 on dSprites, see
    src/evaluation/separation.py), so the null is not automatic.
    """
    rng = np.random.RandomState(seed)
    return mu + std * rng.randn(*mu.shape)


def _balanced_subsample(X, y, seed=0):
    """
    Undersample the majority class so both classes are equally represented.

    MLPClassifier supports neither ``class_weight`` nor ``sample_weight``, so
    balancing has to happen in the data.  This matters for CelebA, where
    Eyeglasses is only ~6.5% of the set and an unbalanced MLP collapses onto
    the majority class (93.5% "accuracy" by predicting all-negative).
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_keep = counts.min()
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), n_keep, replace=False) for c in classes
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _score(clf, X_test, y_test):
    """Shared metric block for both probes.  Includes raw scores for ROC curves."""
    proba  = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return {
        "accuracy":  float(clf.score(X_test, y_test)),
        "auroc":     float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        # Kept so callers can draw ROC curves without refitting.
        # Note: excluded from the cached metric JSONs, which store scalars only.
        "proba":     proba,
    }


def linear_probe(mu_train, y_train, mu_test, y_test, seed=0, max_iter=1000):
    """
    Fit a class-weighted logistic regression on latent codes and evaluate.

    Returns dict with accuracy, auroc, precision, recall, f1, and proba.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(mu_train)
    X_test = scaler.transform(mu_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed,
                             class_weight="balanced")
    clf.fit(X_train, y_train)
    return _score(clf, X_test, y_test)


def mlp_probe(mu_train, y_train, mu_test, y_test, seed=0, hidden=(64,),
              max_iter=1000, balance=True):
    """
    Fit an MLP on latent codes and evaluate — a nonlinear PASSIVE probe.

    This is a stronger version of linear_probe, not the "adversarial recovery"
    method from the project proposal (S3.3.4). Both ask whether the concept is
    decodable from a frozen model's latents right now; adversarial recovery
    (src/evaluation/adversarial.py) is a different, active question -- can an
    attacker find a nearby latent point that DECODES to the concept, and how
    far do they have to search. Do not conflate the two: a model can score
    low here (concept not linearly/nonlinearly decodable) while still being
    highly vulnerable to adversarial_recovery (concept reachable nearby), or
    vice versa.

    ``balance=True`` (default) undersamples the majority class in the *training*
    half only, so the probe cannot win by predicting the majority label.  The
    test half keeps its natural class distribution, so the reported metrics
    still describe real-world performance.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(mu_train)
    X_test = scaler.transform(mu_test)

    if balance:
        X_train, y_train = _balanced_subsample(X_train, y_train, seed=seed)

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter,
                        random_state=seed)
    clf.fit(X_train, y_train)
    return _score(clf, X_test, y_test)