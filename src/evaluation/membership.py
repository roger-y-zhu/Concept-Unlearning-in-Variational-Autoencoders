"""
src/evaluation/membership.py
==============================
Population-level loss-threshold membership inference (Yeom et al. 2018) --
the evaluation axis flagged as the project's biggest gap in
docs/EVALUATION_LANDSCAPE.md (Hayes et al. 2024's U-MIA framing, MUSE's
PrivLeak). Everything else in this project asks "is the CONCEPT (shape=heart)
decodable". This asks a different, privacy-relevant question: "was this
SPECIFIC image in the data the model was fitted on".

THE ATTACK
----------
A model that memorised its training set reconstructs training ("member")
images with systematically lower loss than images it never trained on
("non-member"). The attacker's score is simply per-sample negative ELBO;
success is measured as the AUROC of separating members from non-members by
that score. No shadow models needed -- this is the population-level attack
Hayes et al. show UNDERSTATES risk relative to a per-example attacker, so a
low score here is not proof of safety, only a floor.

WHY THIS IS DSPRITES-ONLY
--------------------------
The attack needs a genuine non-member set: forget-concept images that were
NEVER in any model's gradient updates. dSprites has one -- get_concept_loaders
partitions hearts into a train split (used for original training AND every
method's unlearning objective) and a held-out test split. CelebA's forget set
has no such partition: `get_concept_loaders`'s own docstring states the forget
loader "covers the full forget set... there is nothing to hold out there".
This is a structural property of the CelebA setup, not a bug to fix -- it means
this axis cannot be computed on CelebA at all, and that scope limit should be
stated plainly rather than worked around.

WHAT THE OWNED VS RETRAINED MODELS MEAN AS REFERENCE POINTS
-------------------------------------------------------------
Baseline was trained on train-split hearts, so a real memorisation signal is
expected there: this is the ceiling, not a null result. The oracle never saw
ANY hearts (train or test), so train-vs-test hearts should be indistinguishable
under it (AUROC near 0.5) -- exactly the floor this metric can reach, and the
correct comparison point for every unlearning method's residual leak.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def per_sample_negative_elbo(model, loader, device, *, kl_weight: float = 1.0) -> np.ndarray:
    """Per-sample loss = recon (summed over pixels) + kl_weight * KL.

    Mirrors ConvVAE.forward's formulas exactly (see that docstring for the
    derivation) but without the batch-mean reduction, and decodes from mu
    rather than a reparameterised sample so the score is deterministic --
    consistent with how the rest of this project's evaluation code treats the
    posterior (e.g. adversarial recovery attacks decode(mu), never a sample).
    """
    model.eval()
    out = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        mu, std = model.encode(x)
        x_hat = model.decode(mu)

        if model.cfg.recon_loss == "bce":
            recon = F.binary_cross_entropy_with_logits(
                x_hat, x, reduction="none").sum(dim=tuple(range(1, x.dim())))
        else:
            recon = F.mse_loss(
                x_hat, x, reduction="none").sum(dim=tuple(range(1, x.dim())))

        kl = 0.5 * torch.sum(
            mu.pow(2) + std.pow(2) - 1.0 - torch.log(std.pow(2) + model.cfg.eps),
            dim=1,
        )
        out.append((recon + kl_weight * kl).cpu().numpy())
    return np.concatenate(out)


def loss_threshold_mia(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> dict:
    """AUROC of separating members from non-members by NEGATIVE loss.

    Members are expected to have LOWER loss (better fit -> the model has seen
    them), so the attacker's score is -loss; label 1 = member.

    Returns
    -------
    dict with:
        auroc         attack success. 0.5 = no detectable membership signal
                      (the target for a well-forgotten set); the ceiling is
                      whatever the baseline (never-unlearned) model scores.
        member_loss_mean / nonmember_loss_mean
        n_member / n_nonmember
    """
    from sklearn.metrics import roc_auc_score

    m = np.asarray(member_losses, dtype=np.float64)
    nm = np.asarray(nonmember_losses, dtype=np.float64)
    scores = np.concatenate([-m, -nm])
    labels = np.concatenate([np.ones(len(m)), np.zeros(len(nm))])
    if len(np.unique(labels)) < 2 or len(m) == 0 or len(nm) == 0:
        return {"auroc": float("nan"), "member_loss_mean": float(m.mean()) if len(m) else float("nan"),
                "nonmember_loss_mean": float(nm.mean()) if len(nm) else float("nan"),
                "n_member": int(len(m)), "n_nonmember": int(len(nm))}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "member_loss_mean": float(m.mean()),
        "nonmember_loss_mean": float(nm.mean()),
        "n_member": int(len(m)),
        "n_nonmember": int(len(nm)),
    }
