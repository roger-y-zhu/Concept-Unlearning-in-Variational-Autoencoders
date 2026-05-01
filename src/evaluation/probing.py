import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def encode_dataset(model, loader, device):
    """
    Encode all batches through the VAE encoder.

    Loader must yield either:
      - plain images (no labels)
      - (images, labels) tuples       ← CelebA labeled
      - (images, latents, classes)    ← dSprites with include_labels=True

    Returns: mu (N, latent_dim) and labels (N,) as numpy arrays.
    Labels are None if loader yields plain images.
    """
    model.eval()
    mus, lbls = [], []

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

            mu, _ = model.encode(imgs)
            mus.append(mu.cpu().numpy())
            if lbl is not None:
                lbls.append(lbl)

    mu_matrix = np.concatenate(mus, axis=0)
    labels = np.concatenate(lbls, axis=0) if lbls else None
    return mu_matrix, labels


def linear_probe(mu_train, y_train, mu_test, y_test, seed=0, max_iter=1000):
    """
    Fit a logistic regression on latent codes and evaluate.

    Returns dict with accuracy and auroc.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(mu_train)
    X_test = scaler.transform(mu_test)

    clf = LogisticRegression(max_iter=max_iter, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    proba = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "proba": proba.tolist(),
    }



def mlp_probe(mu_train, y_train, mu_test, y_test, seed=0, hidden=(64,), max_iter=500):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(mu_train)
    X_test = scaler.transform(mu_test)

    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    proba = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, proba)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "proba": proba.tolist(),
    }