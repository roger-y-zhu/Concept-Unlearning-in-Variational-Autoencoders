"""
tests/test_fisher_scrub.py
===========================
Fisher scrubbing (Golatkar et al. 2020, arXiv:1911.04933).

The method is one closed-form line --
    S(w) = w + noise_scale * (F + eps)^(-1/4) * n
-- so the tests here go after the two things that line can silently get wrong
and that a "does it run" smoke test would never catch:

  1. F must be the mean of SQUARED PER-SAMPLE gradients, not the square of the
     batch-mean gradient. The wrong one still produces a plausible tensor of
     the right shape and positive values.
  2. The exponent must be -1/4, not -1/2. Proposition 4 gives the noise
     COVARIANCE as sqrt(lambda sigma_h^2) B^(-1/2); the standard deviation --
     which is what multiplies a N(0,I) draw -- carries half that power. Both
     choices "work" and scale noise inversely with importance; only one
     matches the paper. test_noise_scales_as_fisher_to_the_minus_quarter
     pins the exponent quantitatively and fails for -1/2.
"""

import json

import pytest
import torch
import torch.nn as nn

from src.unlearning.loop import run_unlearning
from src.unlearning.methods import FisherScrub, _fmt_num

from test_unlearning_loop import make_args

CPU = torch.device("cpu")


@pytest.fixture
def deterministic_vae(tiny_vae_bce, monkeypatch):
    """A VAE whose reparameterisation is the identity.

    The FIM is defined by gradients of the loss; with sampling left on, two
    evaluations of the same input give different gradients and nothing can be
    compared exactly. Pinning z = mu removes the sampling noise without
    touching any of the weight-gradient paths under test.
    """
    monkeypatch.setattr(type(tiny_vae_bce), "reparameterise",
                        lambda self, mu, std: mu)
    return tiny_vae_bce


def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# The FIM estimate
# ---------------------------------------------------------------------------

class TestDiagonalFisher:
    def test_matches_a_hand_rolled_per_sample_computation(self, deterministic_vae):
        model = deterministic_vae
        xs = torch.rand(4, 1, 64, 64).round()

        fisher, seen = FisherScrub.diagonal_fisher(model, [xs], CPU, n_samples=4)
        assert seen == 4

        params = _params(model)
        manual = [torch.zeros_like(p) for p in params]
        for i in range(4):
            loss = model(xs[i:i + 1], kl_weight=1.0).loss
            for acc, g in zip(manual, torch.autograd.grad(loss, params)):
                acc += g.pow(2)
        for acc in manual:
            acc /= 4

        for got, want in zip(fisher, manual):
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-8)

    def test_is_not_the_squared_batch_mean_gradient(self, deterministic_vae):
        """The bug this guards: squaring the batch-averaged gradient lets
        opposing per-sample gradients cancel BEFORE squaring, understating
        exactly the specialised weights the method needs to find. Jensen makes
        the relationship strict and directional, so this is checkable rather
        than merely "different".
        """
        model = deterministic_vae
        xs = torch.rand(6, 1, 64, 64).round()

        fisher, _ = FisherScrub.diagonal_fisher(model, [xs], CPU, n_samples=6)

        params = _params(model)
        batched = model(xs, kl_weight=1.0).loss          # already mean-over-batch
        wrong = [g.pow(2) for g in torch.autograd.grad(batched, params)]

        # mean(g^2) >= (mean g)^2 elementwise, by Jensen
        for right, bad in zip(fisher, wrong):
            assert (right >= bad - 1e-9).all()

        # and it must be a strict difference somewhere, not a tie everywhere
        gaps = [(r - b).abs().max().item() for r, b in zip(fisher, wrong)]
        assert max(gaps) > 1e-8, "per-sample and batched estimates are identical"

    def test_is_non_negative_and_shaped_like_the_parameters(self, deterministic_vae):
        model = deterministic_vae
        fisher, _ = FisherScrub.diagonal_fisher(
            model, [torch.rand(3, 1, 64, 64).round()], CPU, n_samples=3)
        for f, p in zip(fisher, _params(model)):
            assert f.shape == p.shape
            assert (f >= 0).all()
            assert torch.isfinite(f).all()

    def test_sample_budget_is_respected_and_can_stop_mid_batch(self, deterministic_vae):
        """n_samples smaller than one batch must stop early, not round up."""
        _, seen = FisherScrub.diagonal_fisher(
            deterministic_vae, [torch.rand(8, 1, 64, 64).round()], CPU, n_samples=3)
        assert seen == 3

    def test_budget_larger_than_the_dataset_uses_what_exists(self, deterministic_vae):
        _, seen = FisherScrub.diagonal_fisher(
            deterministic_vae, [torch.rand(2, 1, 64, 64).round()], CPU, n_samples=500)
        assert seen == 2

    def test_empty_loader_raises_rather_than_dividing_by_zero(self, deterministic_vae):
        with pytest.raises(RuntimeError, match="no samples"):
            FisherScrub.diagonal_fisher(deterministic_vae, [], CPU, n_samples=10)


# ---------------------------------------------------------------------------
# The perturbation
# ---------------------------------------------------------------------------

class _TwoBlock(nn.Module):
    """Two parameter blocks, so a synthetic FIM can give each a known
    importance and the resulting noise magnitudes can be compared."""
    def __init__(self, n=8000):
        super().__init__()
        self.lo = nn.Parameter(torch.zeros(n))
        self.hi = nn.Parameter(torch.zeros(n))


class TestApplyScrub:
    def test_noise_scales_as_fisher_to_the_minus_quarter(self):
        """The exponent, pinned quantitatively.

        With F_lo=1e-4 and F_hi=1e0 the predicted RMS ratio is
        (F_hi/F_lo)^(1/4) = 10. Under the plausible-but-wrong F^(-1/2) it
        would be 100, so this test separates the two.
        """
        F_LO, F_HI = 1e-4, 1.0
        model = _TwoBlock()
        fisher = [torch.full_like(model.lo, F_LO), torch.full_like(model.hi, F_HI)]

        FisherScrub.apply_scrub(model, fisher, CPU, noise_scale=1.0, eps=0.0, seed=0)

        rms_lo = model.lo.detach().pow(2).mean().sqrt().item()
        rms_hi = model.hi.detach().pow(2).mean().sqrt().item()
        assert rms_lo / rms_hi == pytest.approx((F_HI / F_LO) ** 0.25, rel=0.05)

    def test_important_weights_move_less_than_unimportant_ones(self):
        """The method's whole premise, stated directionally."""
        model = _TwoBlock(n=4000)
        fisher = [torch.full_like(model.lo, 1e-6), torch.full_like(model.hi, 1e2)]
        FisherScrub.apply_scrub(model, fisher, CPU, noise_scale=1e-2, eps=0.0, seed=0)
        assert model.lo.detach().abs().mean() > model.hi.detach().abs().mean()

    def test_noise_scale_zero_is_an_exact_no_op(self, deterministic_vae):
        model = deterministic_vae
        before = [p.detach().clone() for p in _params(model)]
        fisher, _ = FisherScrub.diagonal_fisher(
            model, [torch.rand(2, 1, 64, 64).round()], CPU, n_samples=2)
        FisherScrub.apply_scrub(model, fisher, CPU, noise_scale=0.0, eps=1e-8, seed=0)
        for p, b in zip(_params(model), before):
            assert torch.equal(p.detach(), b)

    def test_larger_noise_scale_perturbs_more(self):
        model_small, model_big = _TwoBlock(n=2000), _TwoBlock(n=2000)
        fisher = [torch.full((2000,), 1.0), torch.full((2000,), 1.0)]
        s = FisherScrub.apply_scrub(model_small, fisher, CPU,
                                    noise_scale=1e-3, eps=0.0, seed=0)
        b = FisherScrub.apply_scrub(model_big, fisher, CPU,
                                    noise_scale=1e-1, eps=0.0, seed=0)
        assert b["rms_delta"] > s["rms_delta"] * 50

    def test_zero_fisher_entries_stay_finite_via_eps(self):
        """F is exactly 0 for weights the retain set never exercises;
        (F)^(-1/4) is then inf and every weight becomes NaN."""
        model = _TwoBlock(n=100)
        fisher = [torch.zeros(100), torch.zeros(100)]
        FisherScrub.apply_scrub(model, fisher, CPU, noise_scale=1e-6,
                                eps=1e-8, seed=0)
        assert torch.isfinite(model.lo.detach()).all()
        assert torch.isfinite(model.hi.detach()).all()

    def test_same_seed_is_reproducible_and_different_seeds_are_not(self):
        fisher = [torch.full((500,), 1.0), torch.full((500,), 1.0)]

        def run(seed):
            m = _TwoBlock(n=500)
            FisherScrub.apply_scrub(m, fisher, CPU, noise_scale=1.0, eps=0.0, seed=seed)
            return m.lo.detach().clone()

        assert torch.equal(run(0), run(0))
        assert not torch.equal(run(0), run(1))


# ---------------------------------------------------------------------------
# Run-directory naming
# ---------------------------------------------------------------------------

class TestRunName:
    def test_small_scales_do_not_become_scientific_notation(self):
        """f"{1e-05}" is '1e-05'; the 'e' and '-' break the [0-9.]+ slot that
        nb_utils.discover_sweep_keys parses, so the run would be invisible to
        every sweep notebook."""
        assert _fmt_num(1e-5) == "0.00001"
        assert _fmt_num(1e-3) == "0.001"
        assert _fmt_num(0.0) == "0"
        # Check the parameter slot itself, not the whole name -- "_ep1"
        # legitimately contains an 'e'.
        name = FisherScrub().run_name(make_args("fisher", noise_scale=1e-6))
        assert name.startswith("sig0.000001_ep1")
        import re
        assert re.match(r"sig[0-9.]+_ep[0-9]+", name), name

    def test_distinct_scales_get_distinct_directories(self):
        m = FisherScrub()
        names = {m.run_name(make_args("fisher", noise_scale=v))
                 for v in (1e-5, 1e-4, 1e-3, 1e-2)}
        assert len(names) == 4

    def test_non_default_sample_count_and_seed_are_part_of_the_identity(self):
        """Two runs differing only in FIM sample count or noise seed are
        different runs; sharing a directory would make the second silently
        skip via loop.py's final.pt marker."""
        m = FisherScrub()
        base = m.run_name(make_args("fisher"))
        assert m.run_name(make_args("fisher", fisher_samples=50)) != base
        assert m.run_name(make_args("fisher", fisher_seed=3)) != base
        assert m.run_name(make_args("fisher", fisher_seed=3)).endswith("_s3")


# ---------------------------------------------------------------------------
# One-shot integration with the shared loop
# ---------------------------------------------------------------------------

class TestOneShotLoopIntegration:
    def test_declares_itself_one_shot(self):
        assert FisherScrub().one_shot is True

    def test_no_optimiser_step_touches_the_scrubbed_weights(self, tiny_spec):
        """The core one-shot guarantee. With noise_scale=0 the scrub is an
        exact no-op, so the final weights must equal the baseline BIT FOR BIT.
        Any drift means an optimiser ran -- and AdamW's decoupled weight decay
        moves weights even when every gradient is zero, so this would not stay
        a rounding error."""
        out = run_unlearning(tiny_spec, FisherScrub(),
                             make_args("fisher", noise_scale=0.0), CPU)
        assert out is not None
        base = torch.load(tiny_spec.baseline_ckpt, map_location="cpu",
                          weights_only=True)["model_state_dict"]
        got = torch.load(out / "final.pt", map_location="cpu",
                         weights_only=True)["model_state_dict"]
        for k in base:
            assert torch.equal(base[k], got[k]), f"{k} moved during a one-shot run"

    def test_a_real_scrub_does_change_the_weights(self, tiny_spec):
        """Guards the opposite failure: a no-op that only *looks* correct
        because nothing ever happened."""
        out = run_unlearning(tiny_spec, FisherScrub(),
                             make_args("fisher", noise_scale=1e-2), CPU)
        base = torch.load(tiny_spec.baseline_ckpt, map_location="cpu",
                          weights_only=True)["model_state_dict"]
        got = torch.load(out / "final.pt", map_location="cpu",
                         weights_only=True)["model_state_dict"]
        assert any(not torch.equal(base[k], got[k]) for k in base)

    def test_epochs_is_ignored_and_history_has_one_entry(self, tiny_spec):
        out = run_unlearning(tiny_spec, FisherScrub(),
                             make_args("fisher", epochs=7), CPU)
        hist = json.loads((out / "history.json").read_text())
        assert len(hist["val_loss"]) == 1, "one-shot must not loop over epochs"
        assert hist["selection"]["one_shot"] is True

    def test_checkpoint_records_no_optimiser_state(self, tiny_spec):
        out = run_unlearning(tiny_spec, FisherScrub(), make_args("fisher"), CPU)
        ckpt = torch.load(out / "final.pt", map_location="cpu", weights_only=True)
        assert ckpt["optimizer_state_dict"] is None
        assert ckpt["epoch"] == 1

    def test_reports_both_forget_and_retain_losses(self, tiny_spec):
        out = run_unlearning(tiny_spec, FisherScrub(), make_args("fisher"), CPU)
        hist = json.loads((out / "history.json").read_text())
        for key in ("forget_loss", "retain_loss", "val_loss"):
            assert len(hist[key]) == 1
            assert hist[key][0] == hist[key][0], f"{key} is NaN"

    def test_iterative_methods_still_build_an_optimiser(self, tiny_spec):
        """Regression guard on the loop change: the one_shot branch must not
        leak into the four gradient-based methods."""
        from src.unlearning.methods import NegGrad
        out = run_unlearning(tiny_spec, NegGrad(), make_args("neg_grad"), CPU)
        ckpt = torch.load(out / "final.pt", map_location="cpu", weights_only=True)
        assert ckpt["optimizer_state_dict"] is not None
