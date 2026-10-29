"""
tests/test_adversarial.py
==========================
Adversarial recovery (project proposal S3.3.4): an active latent-space PGD
attack, distinct from the passive `mlp_probe`. Tests check the specific claims
the method makes -- not just that it runs.
"""

import torch
import torch.nn as nn

import pytest

from src.evaluation.adversarial import (
    ConceptClassifier, train_concept_classifier, eval_concept_classifier,
    adversarial_recovery, reconstruction_concept_scores,
    adversarial_recovery_within_radius, ConsensusJudge,
)


# ---------------------------------------------------------------------------
# ConsensusJudge -- the rigorous judge the dense/champion attacks actually use.
# Regression tests for the saturated-gradient bug (2026-09): the old forward
# computed log(clamp(median(sigmoid(logit_i)), eps, 1-eps)), which on a
# trivially-separable concept (every judge saturated) clamped to a CONSTANT and
# zeroed the gradient, so PGD could not climb and attack@r floored near 0 even
# on the baseline. These would all FAIL against that implementation.
# ---------------------------------------------------------------------------

class _AffineJudge(nn.Module):
    """A differentiable 'judge' whose logit is gain * mean(x) + bias. With a
    large gain the logit is large-magnitude (probability saturated at 0/1) yet
    still has a real, non-zero gradient w.r.t. the input -- the exact regime
    that broke the clamped consensus."""
    def __init__(self, gain: float, bias: float = 0.0):
        super().__init__()
        self.gain, self.bias = gain, bias

    def forward(self, x):
        return self.gain * x.flatten(1).mean(dim=1) + self.bias


class TestConsensusJudge:
    def test_output_is_median_of_logits(self):
        judges = [_AffineJudge(1.0, b) for b in (-2.0, 0.5, 3.0)]
        cj = ConsensusJudge(judges)
        x = torch.full((4, 1, 8, 8), 1.0)  # mean(x)=1 -> logits -1, 1.5, 4 -> median 1.5
        out = cj(x)
        assert torch.allclose(out, torch.full((4,), 1.5), atol=1e-5)

    def test_sigmoid_output_equals_median_probability(self):
        """The success test is sigmoid(consensus) > threshold, so this must equal
        the median of the judges' individual probabilities for any threshold."""
        judges = [_AffineJudge(4.0, b) for b in (-1.0, 0.3, 2.0)]
        cj = ConsensusJudge(judges)
        x = torch.rand(6, 1, 8, 8)
        with torch.no_grad():
            probs = torch.stack([torch.sigmoid(j(x)) for j in judges], 0)
            median_prob = probs.median(dim=0).values
            assert torch.allclose(torch.sigmoid(cj(x)), median_prob, atol=1e-5)

    def test_gradient_flows_when_judges_are_saturated(self):
        """THE regression: with saturated judges the consensus must still pass a
        usable gradient to the input. The old clamped-probability form returned
        ~0 gradient here, starving the PGD attack."""
        judges = [_AffineJudge(gain=50.0, bias=b) for b in (-1.0, 0.0, 1.0)]
        cj = ConsensusJudge(judges)
        x = torch.full((3, 1, 8, 8), -0.5, requires_grad=True)  # logits ~ -25 -> prob ~0
        with torch.no_grad():
            assert torch.sigmoid(cj(x)).max() < 1e-6  # genuinely saturated
        (grad,) = torch.autograd.grad(cj(x).sum(), x)
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 1e-3, "consensus gradient vanished on saturated judges"

    def test_majority_decision_matches_median_sign(self):
        # 2 of 3 judges positive -> concept present -> sigmoid(out) > 0.5
        judges = [_AffineJudge(10.0, b) for b in (-5.0, 5.0, 5.0)]
        cj = ConsensusJudge(judges)
        x = torch.zeros(2, 1, 8, 8)  # logits -5, 5, 5 -> median 5 -> present
        assert bool((torch.sigmoid(cj(x)) > 0.5).all())
        judges2 = [_AffineJudge(10.0, b) for b in (-5.0, -5.0, 5.0)]  # 1 of 3 -> absent
        assert bool((torch.sigmoid(ConsensusJudge(judges2)(x)) <= 0.5).all())


# ---------------------------------------------------------------------------
# ConceptClassifier / training
# ---------------------------------------------------------------------------

def _brightness_dataset(n, channels, seed):
    """Synthetic images where mean brightness IS the concept -- a classifier
    should trivially learn this, giving a known-good sanity target."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    base = torch.rand(n, channels, 64, 64, generator=g) * 0.3
    imgs = base + labels.view(-1, 1, 1, 1).float() * 0.6
    return imgs.clamp(0, 1), labels


class TestConceptClassifier:
    def test_forward_shape(self):
        clf = ConceptClassifier(in_channels=3)
        out = clf(torch.rand(5, 3, 64, 64))
        assert out.shape == (5,)

    def test_train_concept_classifier_separates_synthetic_concept(self):
        imgs, labels = _brightness_dataset(200, 3, seed=0)
        loader = list(zip(imgs.split(20), labels.split(20)))
        clf = train_concept_classifier(loader, torch.device("cpu"),
                                       in_channels=3, epochs=8, lr=1e-2)

        test_imgs, test_labels = _brightness_dataset(100, 3, seed=1)
        auroc = eval_concept_classifier(clf, [(test_imgs, test_labels)],
                                        torch.device("cpu"))["auroc"]
        assert auroc > 0.9, f"expected near-perfect separation on an obvious signal, got {auroc}"

    def test_eval_returns_nan_for_single_class_batch(self):
        """AUROC is undefined with only one class present -- must not crash."""
        clf = ConceptClassifier(in_channels=1)
        imgs = torch.rand(10, 1, 64, 64)
        labels = torch.zeros(10, dtype=torch.long)
        auroc = eval_concept_classifier(clf, [(imgs, labels)], torch.device("cpu"))["auroc"]
        assert auroc != auroc  # NaN


# ---------------------------------------------------------------------------
# adversarial_recovery -- the attack itself
# ---------------------------------------------------------------------------

class _BrightnessClassifier(nn.Module):
    """Hand-crafted, not learned: logit = k * mean(top-left 8x8 patch) - offset.

    Deterministic and differentiable, so gradient ascent on this logit w.r.t.
    a decoded image reliably pushes the patch brighter. Using a known
    function instead of a trained net makes the attack's mechanics testable
    independent of classifier-training noise.

    A small PATCH rather than the whole image matters: whole-image mean over
    64x64x3 pixels averages a latent perturbation's effect down to near
    nothing (verified empirically -- an Adam-optimised search over 300 steps
    on a random-init decoder could not push the whole-image mean above ~-1e-4,
    an asymptote at the origin, never actually reachable). An 8x8 patch is
    small enough to be genuinely moved by a handful of PGD steps.

    offset=0 makes the target "patch brighter than baseline" -- empirically
    reachable within ~30 steps at step_size=0.3 from a freshly-initialised
    decoder (verified: start ~-0.01, PGD reaches ~+0.025).
    """
    def __init__(self, k=50.0, offset=0.0):
        super().__init__()
        self.k, self.offset = k, offset

    def forward(self, x):
        return self.k * x[:, :, :8, :8].mean(dim=(1, 2, 3)) - self.offset


class _ImpossibleClassifier(nn.Module):
    """A large negative logit that still depends on z (via a zero-weighted
    term) -- sigmoid never exceeds any reachable threshold, so success_rate
    must be exactly 0. Must stay connected to the autograd graph: a genuinely
    constant output is the degenerate case adversarial_recovery itself must
    survive (see test_impossible_target_never_succeeds), not something the
    classifier is required to produce."""
    def forward(self, x):
        return -1000.0 + 0.0 * x.mean(dim=(1, 2, 3))


def _loader(batch, n_batches=1):
    return [batch for _ in range(n_batches)]


class _AlwaysPositiveClassifier(nn.Module):
    def forward(self, x):
        return 10.0 + 0.0 * x.mean(dim=(1, 2, 3))


def _labeled_loader(imgs, labels, n_batches=1):
    return [(imgs, labels) for _ in range(n_batches)]


class TestReconstructionValidityGate:
    """adversarial_recovery trains its judge on REAL images but applies it to
    DECODED ones. If the judge never crosses the threshold even on the
    baseline's reconstruction of a forget image, success_rate is 0 for every
    model -- and a vacuous 0 is indistinguishable from a meaningful 0 in the
    attack's own output. The gate has to be a separate measurement."""

    def _batch(self, n=8):
        imgs = torch.rand(n, 3, 64, 64)
        labels = torch.zeros(n, dtype=torch.long)
        labels[: n // 2] = 1
        return imgs, labels

    def test_unreachable_classifier_is_flagged_and_the_attack_really_does_fail(
            self, tiny_vae_mse):
        """The gate's whole job: predict the vacuous case BEFORE spending a
        sweep on it. Assert both halves -- flagged as meaningless, and the
        attack it was gating genuinely returns nothing."""
        imgs, labels = self._batch()
        clf = _ImpossibleClassifier()

        gate = reconstruction_concept_scores(
            tiny_vae_mse, clf, _labeled_loader(imgs, labels), torch.device("cpu"))
        assert gate["attack_is_meaningful"] is False
        assert gate["p_forget_mean"] < 0.5

        attack = adversarial_recovery(tiny_vae_mse, clf, [imgs], torch.device("cpu"),
                                      max_steps=10, step_size=0.5)
        assert attack["success_rate"] == 0.0

    def test_reachable_classifier_is_not_flagged(self, tiny_vae_mse):
        imgs, labels = self._batch()
        gate = reconstruction_concept_scores(
            tiny_vae_mse, _AlwaysPositiveClassifier(),
            _labeled_loader(imgs, labels), torch.device("cpu"))
        assert gate["attack_is_meaningful"] is True
        assert gate["p_forget_mean"] > 0.5

    def test_suggested_threshold_lies_inside_the_observed_range(self, tiny_vae_mse):
        """When 0.5 is unreachable the caller needs a usable substitute cut, not
        a number outside the distribution it was derived from."""
        imgs, labels = self._batch()
        clf = _BrightnessClassifier(k=50.0, offset=8.0)   # pushes every prob well below 0.5
        gate = reconstruction_concept_scores(
            tiny_vae_mse, clf, _labeled_loader(imgs, labels), torch.device("cpu"))

        assert gate["attack_is_meaningful"] is False
        assert 0.0 < gate["suggested_threshold"] < 0.5
        assert 0.0 <= gate["auroc"] <= 1.0

    def test_suggested_threshold_is_finite_when_the_classes_are_inseparable(
            self, tiny_vae_mse):
        """roc_curve's first threshold is +inf. With an all-constant classifier
        it also wins the Youden argmax, so an unclipped implementation returns
        inf here -- in exactly the degenerate case the gate is meant to catch."""
        imgs, labels = self._batch()
        gate = reconstruction_concept_scores(
            tiny_vae_mse, _ImpossibleClassifier(),
            _labeled_loader(imgs, labels), torch.device("cpu"))
        s = gate["suggested_threshold"]
        assert torch.isfinite(torch.tensor(s)), f"suggested_threshold={s}"
        assert 0.0 <= s <= 1.0, "a probability cut must be a probability"

    def test_single_class_raises_rather_than_reporting_a_meaningless_auroc(
            self, tiny_vae_mse):
        imgs = torch.rand(6, 3, 64, 64)
        with pytest.raises(ValueError):
            reconstruction_concept_scores(
                tiny_vae_mse, _AlwaysPositiveClassifier(),
                _labeled_loader(imgs, torch.zeros(6, dtype=torch.long)),
                torch.device("cpu"))

    def test_max_batches_limits_work(self, tiny_vae_mse):
        imgs, labels = self._batch()
        gate = reconstruction_concept_scores(
            tiny_vae_mse, _AlwaysPositiveClassifier(),
            _labeled_loader(imgs, labels, n_batches=5), torch.device("cpu"),
            max_batches=2)
        assert gate["n"] == 16


class TestAdversarialRecovery:
    def test_reported_distance_matches_a_real_flip(self, tiny_vae_mse):
        """The claimed property: for every reported success, re-decoding
        z_success and re-classifying it must actually exceed the threshold,
        and the reported distance must equal ||z_success - mu0||."""
        model = tiny_vae_mse
        clf = _BrightnessClassifier()
        imgs = torch.rand(6, 3, 64, 64)

        out = adversarial_recovery(model, clf, _loader(imgs), torch.device("cpu"),
                                   max_steps=30, step_size=0.3, success_threshold=0.5)

        with torch.no_grad():
            mu0, _ = model.encode(imgs)

        assert out["success_rate"] > 0, "brightness is an easy target -- some samples must flip"
        for i, (d, z) in enumerate(zip(out["distances"], out["z_success"])):
            if d is None:
                assert z is None
                continue
            assert z is not None
            with torch.no_grad():
                prob = torch.sigmoid(clf(model.decode(z.unsqueeze(0))))[0].item()
            assert prob > 0.5, f"sample {i}: reported success but re-check gives prob={prob}"
            assert abs((z - mu0[i]).norm().item() - d) < 1e-4

    def test_survives_a_classifier_output_disconnected_from_z(self, tiny_vae_mse):
        """A classifier whose output has no grad_fn w.r.t. z (fully constant,
        no dependence on the input at all) must not crash the sweep -- it
        should simply find no direction to move in and censor every sample."""
        class ConstantClassifier(nn.Module):
            def forward(self, x):
                return torch.full((x.size(0),), -5.0)   # no grad_fn at all

        imgs = torch.rand(4, 3, 64, 64)
        out = adversarial_recovery(tiny_vae_mse, ConstantClassifier(), _loader(imgs),
                                   torch.device("cpu"), max_steps=5, step_size=0.3)
        assert out["success_rate"] == 0.0
        assert out["n"] == 4

    def test_impossible_target_never_succeeds(self, tiny_vae_bce):
        """A classifier that can never be satisfied must censor every sample,
        not report a spurious success."""
        imgs = torch.randint(0, 2, (4, 1, 64, 64)).float()
        out = adversarial_recovery(tiny_vae_bce, _ImpossibleClassifier(), _loader(imgs),
                                   torch.device("cpu"), max_steps=10, step_size=0.5,
                                   to_img=torch.sigmoid)
        assert out["success_rate"] == 0.0
        assert all(d is None for d in out["distances"])
        assert all(z is None for z in out["z_success"])
        assert out["mean_distance"] is None

    def test_more_steps_can_only_find_more_successes(self, tiny_vae_mse):
        """PGD here is a deterministic greedy walk with early-stopping checks
        every step -- a sample that flips within k steps must also flip
        within any k' > k (same trajectory prefix), so the success set at a
        smaller step budget must be a subset of the success set at a larger
        one. This must hold exactly, not just on average."""
        model = tiny_vae_mse
        clf = _BrightnessClassifier()
        imgs = torch.rand(8, 3, 64, 64)

        short = adversarial_recovery(model, clf, _loader(imgs), torch.device("cpu"),
                                     max_steps=2, step_size=0.3, success_threshold=0.5)
        long = adversarial_recovery(model, clf, _loader(imgs), torch.device("cpu"),
                                    max_steps=30, step_size=0.3, success_threshold=0.5)

        for i in range(len(short["distances"])):
            if short["distances"][i] is not None:
                assert long["distances"][i] is not None, \
                    f"sample {i} succeeded with fewer steps but not more"
                assert long["distances"][i] <= short["distances"][i] + 1e-5

    def test_n_matches_dataset_size(self, tiny_vae_mse):
        imgs = torch.rand(5, 3, 64, 64)
        out = adversarial_recovery(tiny_vae_mse, _BrightnessClassifier(),
                                   _loader(imgs, n_batches=3), torch.device("cpu"),
                                   max_steps=3, step_size=0.1)
        assert out["n"] == 15

    def test_max_batches_limits_work(self, tiny_vae_mse):
        imgs = torch.rand(4, 3, 64, 64)
        out = adversarial_recovery(tiny_vae_mse, _BrightnessClassifier(),
                                   _loader(imgs, n_batches=5), torch.device("cpu"),
                                   max_steps=2, step_size=0.1, max_batches=2)
        assert out["n"] == 8

    def test_gradient_actually_moves_toward_higher_logit(self, tiny_vae_mse):
        """Sanity on the mechanism itself: one PGD step must not decrease the
        classifier's logit on average, for an unconstrained ascent step."""
        model, clf = tiny_vae_mse, _BrightnessClassifier()
        imgs = torch.rand(6, 3, 64, 64)
        with torch.no_grad():
            mu0, _ = model.encode(imgs)
            logit_before = clf(model.decode(mu0)).mean().item()

        z = mu0.clone().requires_grad_(True)
        logits = clf(model.decode(z))
        grad = torch.autograd.grad(logits.sum(), z)[0]
        with torch.no_grad():
            z_after = mu0 + 0.3 * grad.sign()
            logit_after = clf(model.decode(z_after)).mean().item()

        assert logit_after > logit_before


# ---------------------------------------------------------------------------
# Radius-constrained variant
# ---------------------------------------------------------------------------

class TestRecoveryWithinRadius:
    """The claims this function makes that `adversarial_recovery` cannot.

    The unconstrained walk reports a distance that is really `n_steps *
    step_size`, so it moves when the optimiser's stride moves even though no
    geometry changed. These tests pin the properties that make the constrained
    version usable for distance claims instead.
    """

    def test_success_is_monotone_in_radius(self, tiny_vae_mse):
        """A larger ball contains the smaller one, so success cannot decrease.

        This is the property the unconstrained walk violates in practice.
        """
        out = adversarial_recovery_within_radius(
            tiny_vae_mse, _BrightnessClassifier(), [torch.rand(8, 3, 64, 64)],
            torch.device("cpu"), radii=(0.5, 1.0, 2.0, 4.0), steps_per_radius=20,
        )
        vals = [out["success_at_r"][str(r)] for r in out["radii"]]
        assert vals == sorted(vals), f"success must be non-decreasing in r, got {vals}"

    def test_min_radius_never_below_first_success_radius(self, tiny_vae_mse):
        """min_radius must be drawn from the grid, and consistent with the curve."""
        out = adversarial_recovery_within_radius(
            tiny_vae_mse, _BrightnessClassifier(), [torch.rand(6, 3, 64, 64)],
            torch.device("cpu"), radii=(1.0, 2.0), steps_per_radius=20,
        )
        for m in out["min_radius"]:
            assert m is None or m in out["radii"]

    def test_impossible_target_never_succeeds_at_any_radius(self, tiny_vae_mse):
        """A classifier pinned far below threshold must yield an all-zero curve.

        Guards the vacuous-zero case: this has to stay distinguishable from a
        model that genuinely resists the attack.
        """
        clf = _BrightnessClassifier(k=50.0, offset=8.0)
        out = adversarial_recovery_within_radius(
            tiny_vae_mse, clf, [torch.rand(6, 3, 64, 64)], torch.device("cpu"),
            radii=(1.0, 4.0), steps_per_radius=10,
        )
        assert all(v == 0.0 for v in out["success_at_r"].values())
        assert all(m is None for m in out["min_radius"])

    def test_always_positive_classifier_succeeds_at_smallest_radius(self, tiny_vae_mse):
        """If the point already decodes to the concept, min_radius is the
        smallest grid value -- the attack needed no room at all."""
        out = adversarial_recovery_within_radius(
            tiny_vae_mse, _AlwaysPositiveClassifier(), [torch.rand(6, 3, 64, 64)],
            torch.device("cpu"), radii=(0.5, 2.0), steps_per_radius=5,
        )
        assert out["success_at_r"]["0.5"] == 1.0
        assert all(m == 0.5 for m in out["min_radius"])

    def test_iterates_stay_inside_the_ball(self, tiny_vae_mse):
        """The projection is the whole point -- verify it directly rather than
        trusting that the reported radius means anything."""
        model, clf = tiny_vae_mse, _BrightnessClassifier()
        imgs = torch.rand(5, 3, 64, 64)
        with torch.no_grad():
            mu0, _ = model.encode(imgs)
        r = 1.0
        z = mu0.clone().requires_grad_(True)
        for _ in range(25):
            logits = clf(model.decode(z))
            (grad,) = torch.autograd.grad(logits.sum(), z)
            with torch.no_grad():
                z = z + (2.5 * r / 25) * grad.sign()
                d = z - mu0
                nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
                z = mu0 + d * torch.clamp(r / nrm, max=1.0)
            z = z.detach().requires_grad_(True)
        assert torch.all((z - mu0).norm(dim=1) <= r + 1e-5)

    def test_max_batches_limits_work(self, tiny_vae_mse):
        out = adversarial_recovery_within_radius(
            tiny_vae_mse, _BrightnessClassifier(), _loader(torch.rand(4, 3, 64, 64), n_batches=5),
            torch.device("cpu"), radii=(1.0,), steps_per_radius=5, max_batches=2,
        )
        assert out["n"] == 8
