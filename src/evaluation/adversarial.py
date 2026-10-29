"""
src/evaluation/adversarial.py
==============================
Adversarial recovery (project proposal S3.3.4) -- the strongest test in the
evaluation framework, and the one that was previously unimplemented.

This is NOT the MLP probe. `probing.mlp_probe` is a passive classifier fit on
frozen latent codes -- it asks "is the concept linearly/nonlinearly decodable
right now". Adversarial recovery is an active attack: starting from a
retain-set image's latent code, search for a nearby point that decodes to the
forget concept. A model where a tiny perturbation suffices has the forget
region sitting right next to retain data in latent space, whatever a passive
probe says; a model where no perturbation within a large budget succeeds has
genuinely pushed the concept out of reach.

Two pieces:
    ConceptClassifier / train_concept_classifier   the attacker's oracle,
        trained once per dataset on REAL images (not decoder output) so every
        model under test is attacked by the same fixed judge.
    adversarial_recovery                            the PGD-style latent
        search itself, run per model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The attacker's classifier -- trained once on real images, reused for every
# model under test so the standard of "looks like the concept" stays fixed.
# ---------------------------------------------------------------------------

class ConceptClassifier(nn.Module):
    """Small binary CNN classifier for 64x64 images. Lighter than ConvVAE's
    encoder -- this is a much easier task than reconstruction."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(c * 8 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (B,). Positive = concept present."""
        return self.fc(self.conv(x).flatten(1)).squeeze(-1)


class ConceptMLPClassifier(nn.Module):
    """No convolutions at all -- flatten and use an MLP.

    Reseeding a judge does not test architecture. All the reseeded judges share
    one CNN, so a shared inductive bias would survive reseeding and still look
    like agreement. This is deliberately the most different thing that can still
    learn the task: no weight sharing, no locality, no translation equivariance.

    Trained on downsampled input to keep the first layer tractable; the point is
    a different bias, not a competitive classifier.
    """

    def __init__(self, in_channels: int = 3, size: int = 16, hidden: int = 512):
        super().__init__()
        self.size = size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * size * size, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, self.size)
        return self.net(x).squeeze(-1)


class ConceptGAPClassifier(nn.Module):
    """CNN whose readout is global average pooling rather than a flattened map.

    Between the two extremes: convolutional like ConceptClassifier, but the
    classifier head cannot use absolute spatial position. If a result depends on
    the judge keying on WHERE something is rather than WHAT it is, this and the
    flatten-head judge will disagree.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(c * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).mean(dim=(2, 3))     # global average pool
        return self.fc(h).squeeze(-1)


class ConsensusJudge(nn.Module):
    """Several judges, success only where a MAJORITY agree.

    WHY THIS IS THE RIGOROUS JUDGE
    ------------------------------
    The attack is PGD against a classifier, so it can always succeed by finding
    an adversarial example -- a latent point whose decoding fools that one
    classifier without looking like the concept to anything else. Measured on
    CelebA: attacks that beat the CNN judge transfer to a held-out MLP judge at
    0.75 for the baseline but only **0.035** for feature_unlearn, against a
    ceiling of 0.69 on REAL concept images. So a single-judge success rate mixes
    genuine concept reachability with classifier fragility, in proportions that
    differ per model -- which is exactly the way a metric can rank models
    wrongly while looking self-consistent.

    Taking the MEDIAN probability makes fooling one judge insufficient: the
    attack must move at least a majority. Adversarial examples are famously
    judge-specific, genuine concept content is not, so the median is a cheap
    and principled filter between them.

    Returns a logit (the median of the judges' raw logits) so it drops into any
    code expecting a classifier's raw output.

    WHY THE MEDIAN IS TAKEN ON LOGITS, NOT CLAMPED PROBABILITIES
    -----------------------------------------------------------
    An earlier version computed ``log(clamp(median(sigmoid(z_i)), eps, 1-eps))``.
    Because sigmoid is monotonic, ``median(sigmoid(z_i)) == sigmoid(median(z_i))``,
    so for the success test (median prob > threshold) that form is *identical* to
    returning ``median(z_i)`` -- but it destroys the gradient the PGD attack needs.
    On a trivially-separable concept (MPI3D object_color=red: judge AUROC 1.000,
    probabilities pinned at 0/1) every judge saturates, the median probability is
    ~0, the clamp pins it to eps, and the log-odds becomes a CONSTANT -> zero
    gradient. Measured: the consensus attack on the MPI3D baseline floored at ~0.04
    success regardless of step count (60 vs 200) or radius (8 vs 32), while the
    concept is plainly reachable (a single-CNN attack hits 0.6-1.0). Taking the
    median of the raw logits gives the *same* majority-agreement decision (median
    logit > 0 iff a majority of judges say "present") with a smooth, non-saturating
    gradient. sigmoid(output) still equals the (unclamped) median probability, so
    every success test `sigmoid(logits) > threshold` is unchanged.
    """

    def __init__(self, judges, eps: float = 1e-6):
        super().__init__()
        self.judges = nn.ModuleList(judges)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([j(x) for j in self.judges], dim=0)
        return logits.median(dim=0).values


#: Judge architectures. "cnn" is the original; the others exist to test whether
#: a conclusion depends on the judge's inductive bias rather than on the models.
JUDGE_ARCHS = {
    "cnn": ConceptClassifier,
    "mlp": ConceptMLPClassifier,
    "gap": ConceptGAPClassifier,
}


def _unpack_labeled_batch(batch, forget_col: int, forget_val: int):
    """Mirrors probing.encode_dataset's batch-shape convention.

    dSprites' LabeledCelebA-style loaders already normalise labels to {0, 1}
    (see LabeledCelebADataset), so `== forget_val` is correct uniformly --
    no special-casing needed for forget_val == 1.
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        imgs = batch[0]
        y = (batch[2][:, forget_col] == forget_val).long()
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        imgs = batch[0]
        y = (batch[1] == forget_val).long()
    else:
        raise ValueError("Loader must yield (img, labels) or (img, latents, classes); "
                         "adversarial recovery needs ground-truth concept labels.")
    return imgs, y


def train_concept_classifier(
    loader,
    device,
    *,
    in_channels: int = 3,
    forget_col: int = 1,
    forget_val: int = 1,
    epochs: int = 5,
    lr: float = 1e-3,
    arch: str = "cnn",
) -> nn.Module:
    """Train the attacker's classifier on REAL images with ground-truth labels.

    Trained once per dataset/concept and reused across every model under test,
    so "looks like the concept" is a fixed external standard rather than
    something that could drift with the model being attacked.
    """
    # arch is a robustness knob, not a tuning knob: varying it tests whether a
    # conclusion depends on the judge's inductive bias.  See JUDGE_ARCHS.
    model = JUDGE_ARCHS[arch](in_channels=in_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            imgs, y = _unpack_labeled_batch(batch, forget_col, forget_val)
            imgs, y = imgs.to(device), y.to(device).float()
            logits = model(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_concept_classifier(model: ConceptClassifier, loader, device,
                            *, forget_col: int = 1, forget_val: int = 1) -> dict:
    """AUROC of the trained classifier -- sanity-check it before trusting an attack."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs, ys = [], []
    for batch in loader:
        imgs, y = _unpack_labeled_batch(batch, forget_col, forget_val)
        p = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return {"auroc": float(roc_auc_score(ys, probs)) if len(set(ys.tolist())) > 1 else float("nan")}


# ---------------------------------------------------------------------------
# Validity gate -- run this BEFORE trusting any adversarial_recovery number
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruction_concept_scores(
    model,
    classifier: ConceptClassifier,
    loader,
    device,
    *,
    to_img=lambda x: x,
    forget_col: int = 1,
    forget_val: int = 1,
    max_batches: int | None = None,
) -> dict:
    """Does the classifier fire on DECODER OUTPUT at all?

    adversarial_recovery trains its judge on real images but applies it to
    decoded ones, which are blurry and off-distribution.  If the classifier
    never crosses success_threshold even on decode(mu(x_forget)) under the
    BASELINE model -- a model that has not unlearned anything and reconstructs
    the concept as well as it ever will -- then success_rate is 0 for every
    model and the metric is vacuous rather than informative.  A vacuous 0 and
    a meaningful 0 look identical in the output dict, which is why this has to
    be checked separately rather than inferred afterwards.

    Run it on the baseline, read `attack_is_meaningful`, and only then sweep.

    Returns
    -------
    dict with:
        p_forget_mean / p_forget_median   classifier probability on decoded
        p_retain_mean / p_retain_median   forget- and retain-class images
        auroc                  separability of the two on decoder output
        suggested_threshold    Youden-J-optimal cut on THIS distribution, for
                               use as adversarial_recovery's success_threshold
                               when the default 0.5 is unreachable
        attack_is_meaningful   True if some decoded forget image already
                               exceeds 0.5 -- i.e. the default threshold is
                               reachable in principle
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    model.eval()
    classifier.eval()
    probs, ys = [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs, y = _unpack_labeled_batch(batch, forget_col, forget_val)
        imgs = imgs.to(device)
        mu, _ = model.encode(imgs)
        recon = to_img(model.decode(mu))
        probs.append(torch.sigmoid(classifier(recon)).cpu().numpy())
        ys.append(y.numpy())

    probs, ys = np.concatenate(probs), np.concatenate(ys).astype(bool)
    if not ys.any() or ys.all():
        raise ValueError("need both concept-positive and concept-negative samples; "
                         f"got {int(ys.sum())} positive of {len(ys)}")

    pf, pr = probs[ys], probs[~ys]
    auroc = float(roc_auc_score(ys, probs))
    fpr, tpr, thr = roc_curve(ys, probs)
    # roc_curve prepends an infinite threshold (max(score)+1 on older sklearn),
    # which wins the Youden argmax whenever the classes are inseparable -- and
    # inseparable is precisely the case this function exists to detect.  Clip
    # to the observed range so the caller always gets a usable cut.
    suggested = float(np.clip(thr[int(np.argmax(tpr - fpr))],
                              float(probs.min()), float(probs.max())))

    return {
        "p_forget_mean":       float(pf.mean()),
        "p_forget_median":     float(np.median(pf)),
        "p_retain_mean":       float(pr.mean()),
        "p_retain_median":     float(np.median(pr)),
        "auroc":               auroc,
        "suggested_threshold": suggested,
        "attack_is_meaningful": bool((pf > 0.5).any()),
        "n":                   int(len(probs)),
    }


# ---------------------------------------------------------------------------
# The attack itself
# ---------------------------------------------------------------------------

def adversarial_recovery(
    model,
    classifier: ConceptClassifier,
    source_loader,
    device,
    *,
    to_img=lambda x: x,
    max_steps: int = 50,
    step_size: float = 0.1,
    success_threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict:
    """Latent-space PGD search for the nearest point that decodes to the concept.

    For each retain-set image, encode to mu, then take signed-gradient ascent
    steps on the classifier's logit for decode(z) -- checking for success after
    every step so the reported distance is the SMALLEST perturbation that
    flips the classifier, not just whether max_steps was enough.

    No radius cap is applied during the search (the question is "how far",
    not "does it fit in a fixed budget"); max_steps * step_size is the
    effective ceiling, and a sample that never succeeds is censored at that
    ceiling rather than treated as an unbounded success.

    Returns
    -------
    dict with:
        success_rate     fraction of samples where a flip was found
        distances        list of L2 latent distances at first success
                          (None for samples that never flipped)
        mean_distance    mean over successes only (None if no successes)
        n                number of samples attacked
        z_success        list of latent tensors at first success, or None --
                          lets a caller independently re-verify a reported
                          success by re-decoding and re-classifying it.
    """
    model.eval()
    distances: list[float | None] = []
    z_success: list[torch.Tensor | None] = []

    for bi, batch in enumerate(source_loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)

        with torch.no_grad():
            mu0, _ = model.encode(imgs)
        z = mu0.clone().detach().requires_grad_(True)

        found = torch.zeros(z.size(0), dtype=torch.bool, device=device)
        best_dist = torch.full((z.size(0),), float("nan"), device=device)
        best_z = mu0.clone().detach()

        for _ in range(max_steps):
            recon = to_img(model.decode(z))
            logits = classifier(recon)
            probs = torch.sigmoid(logits)

            newly = (~found) & (probs > success_threshold)
            if newly.any():
                dist = (z - mu0).norm(dim=1)
                best_dist = torch.where(newly, dist.detach(), best_dist)
                best_z[newly] = z.detach()[newly]
                found = found | newly
            if found.all():
                break

            # Ascend the logit directly (maximise it), rather than defining a
            # separate "loss" to descend -- that indirection is exactly what
            # caused an earlier version of this function to update z in the
            # WRONG direction (descending the logit while believing it was
            # ascending it). See tests/test_adversarial.py's monotonicity and
            # exact-flip-verification tests, which catch this class of bug.
            objective = logits.sum()
            if not objective.requires_grad:
                # classifier's output didn't depend on z at all (e.g. a
                # genuinely constant classifier, or fully saturated
                # activations) -- no direction to move in. Stop rather than
                # crash; remaining samples stay censored.
                break
            (grad,) = torch.autograd.grad(objective, z, allow_unused=True)
            if grad is None:
                break
            with torch.no_grad():
                z = z + step_size * grad.sign()
            z = z.detach().requires_grad_(True)

        for i in range(mu0.size(0)):
            if found[i]:
                distances.append(float(best_dist[i].item()))
                z_success.append(best_z[i].clone())
            else:
                distances.append(None)
                z_success.append(None)

    n = len(distances)
    successes = [d for d in distances if d is not None]
    return {
        "success_rate":  len(successes) / n if n else float("nan"),
        "distances":     distances,
        "mean_distance": float(np.mean(successes)) if successes else None,
        "n":             n,
        "z_success":     z_success,
    }


# ---------------------------------------------------------------------------
# Radius-constrained variant -- the one to trust for DISTANCE claims
# ---------------------------------------------------------------------------

def adversarial_recovery_within_radius(
    model,
    classifier: ConceptClassifier,
    source_loader,
    device,
    *,
    to_img=lambda x: x,
    radii=(0.5, 1.0, 2.0, 4.0, 8.0),
    steps_per_radius: int = 100,
    success_threshold: float = 0.5,
    max_batches: int | None = None,
    restarts: int = 1,
    seed: int = 0,
    step_per_radius: float | None = None,
) -> dict:
    """Is a concept-decoding point reachable within L2 radius `r` of mu?

    WHY THIS EXISTS ALONGSIDE `adversarial_recovery`
    ------------------------------------------------
    `adversarial_recovery` walks an unconstrained sign-gradient trajectory and
    reports the distance at first success. That distance is **step-size
    confounded**, badly enough to invert conclusions:

      * `z + step_size * grad.sign()` moves `step_size * sqrt(d)` in L2,
        because every entry of `grad.sign()` is +-1. A step size that is
        sensible at d=8 travels 5.7x further per step at d=256, so two
        datasets are not on a common scale unless step_size is divided by
        sqrt(d).
      * Even after fixing that, success depends on the step size itself rather
        than on any geometric property. Measured on dSprites at matched reach
        (~15 in latent L2): coarse steps (0.283/step) succeed on 71% of retain
        points, fine steps (0.05/step) on 9%. Successes occur after ~4 steps in
        BOTH cases -- large sign-steps jump out of the plateau the decoder's
        logit surface puts around mu, small ones descend into it and stall. So
        the reported "distance" was mostly `4 * step_size`.

    Constraining the search to a ball of radius `r` and scaling the step size
    to that radius removes the confound: each radius gets an optimiser sized to
    it, and the reported quantity is the well-posed one -- *can the attacker
    reach the concept without leaving the ball* -- rather than *where did this
    particular trajectory happen to land*. This is the standard
    minimum-norm-adversarial-example formulation.

    `min_radius` per sample is the smallest grid radius that succeeded, so it
    is quantised to `radii` but is not an artefact of the optimiser's stride.

    `step_per_radius` (default None) overrides the stride: the step is `step_per_radius * r`
    instead of `2.5 * r / steps_per_radius`. The default keeps the headline behaviour; the override
    lets a sensitivity analysis change the number of steps without changing the stride.

    Returns
    -------
    dict with:
        success_at_r   {radius: fraction of samples reachable within it}
        min_radius     per-sample smallest successful radius (None if none)
        radii          the grid used
        n              number of samples attacked
    """
    model.eval()
    classifier.eval()
    radii = tuple(sorted(float(r) for r in radii))

    per_sample_min: list[float | None] = []

    for bi, batch in enumerate(source_loader):
        if max_batches is not None and bi >= max_batches:
            break
        imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)

        with torch.no_grad():
            mu0, _ = model.encode(imgs)

        best = torch.full((mu0.size(0),), float("nan"), device=device)
        found_any = torch.zeros(mu0.size(0), dtype=torch.bool, device=device)

        for r in radii:
            # Standard PGD heuristic: enough stride to cross the ball a couple
            # of times over the step budget, so the optimiser is matched to r.
            step = (2.5 * r / steps_per_radius) if step_per_radius is None else step_per_radius * r
            hit = torch.zeros(mu0.size(0), dtype=torch.bool, device=device)

            for attempt in range(max(1, restarts)):
                if attempt == 0:
                    delta = torch.zeros_like(mu0)
                else:
                    g = torch.Generator(device="cpu").manual_seed(seed + attempt)
                    d0 = torch.randn(mu0.shape, generator=g).to(device)
                    d0 = d0 / d0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    scale = torch.rand(mu0.size(0), 1, generator=g).to(device) ** (1.0 / mu0.size(1))
                    delta = d0 * scale * r

                z = (mu0 + delta).detach().requires_grad_(True)

                for _ in range(steps_per_radius):
                    logits = classifier(to_img(model.decode(z)))
                    with torch.no_grad():
                        hit |= torch.sigmoid(logits) > success_threshold
                    if bool(hit.all()):
                        break
                    objective = logits.sum()
                    if not objective.requires_grad:
                        break
                    (grad,) = torch.autograd.grad(objective, z, allow_unused=True)
                    if grad is None:
                        break
                    with torch.no_grad():
                        z = z + step * grad.sign()
                        # Project back into the ball -- this is what makes the
                        # radius, not the stride, the binding constraint.
                        d = z - mu0
                        nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        z = mu0 + d * torch.clamp(r / nrm, max=1.0)
                    z = z.detach().requires_grad_(True)

                if bool(hit.all()):
                    break

            newly = hit & (~found_any)
            best = torch.where(newly, torch.full_like(best, r), best)
            found_any |= hit
            if bool(found_any.all()):
                break  # every sample already has its minimal radius

        for i in range(mu0.size(0)):
            per_sample_min.append(float(best[i].item()) if bool(found_any[i]) else None)

    n = len(per_sample_min)
    return {
        "success_at_r": {
            str(r): (sum(1 for m in per_sample_min if m is not None and m <= r) / n
                     if n else float("nan"))
            for r in radii
        },
        "min_radius": per_sample_min,
        "radii":      list(radii),
        "n":          n,
    }
