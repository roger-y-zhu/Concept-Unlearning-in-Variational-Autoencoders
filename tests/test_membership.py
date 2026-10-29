"""
tests/test_membership.py
==========================
The loss-threshold MIA (src/evaluation/membership.py) makes one directional
claim -- members score lower loss than non-members under a model that
memorised them -- and these tests pin that claim plus the degenerate cases a
population attack must not silently mishandle.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.vae import ConvVAE, VAEConfig
from src.evaluation.membership import per_sample_negative_elbo, loss_threshold_mia


def _vae(recon_loss="bce", in_channels=1):
    torch.manual_seed(0)
    cfg = VAEConfig(latent_dim=4, base_channels=4, in_channels=in_channels,
                    recon_loss=recon_loss)
    return ConvVAE(cfg)


class TestLossThresholdMIA:
    def test_perfect_separation_gives_auroc_one(self):
        member = np.full(50, 1.0)      # low loss -- the model fit these
        nonmember = np.full(50, 10.0)  # high loss -- never seen
        out = loss_threshold_mia(member, nonmember)
        assert out["auroc"] == pytest.approx(1.0)

    def test_identical_distributions_give_chance(self):
        rng = np.random.default_rng(0)
        member = rng.normal(5, 1, 2000)
        nonmember = rng.normal(5, 1, 2000)
        out = loss_threshold_mia(member, nonmember)
        assert abs(out["auroc"] - 0.5) < 0.05

    def test_reversed_gap_gives_auroc_below_half(self):
        """If non-members somehow score LOWER loss than members, the attack
        must report that honestly (auroc < 0.5), not clip or invert silently."""
        member = np.full(50, 10.0)
        nonmember = np.full(50, 1.0)
        out = loss_threshold_mia(member, nonmember)
        assert out["auroc"] < 0.1

    def test_reports_loss_means(self):
        out = loss_threshold_mia(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
        assert out["member_loss_mean"] == pytest.approx(1.5)
        assert out["nonmember_loss_mean"] == pytest.approx(5.5)

    def test_empty_member_set_is_nan_not_a_crash(self):
        out = loss_threshold_mia(np.array([]), np.array([1.0, 2.0]))
        assert out["auroc"] != out["auroc"]  # NaN
        assert out["n_member"] == 0


class TestPerSampleNegativeELBO:
    def test_matches_batch_mean_up_to_reduction(self):
        """Per-sample values must average to the same thing forward()'s
        batch-mean loss reports, or the two are measuring different formulas."""
        model = _vae("bce", in_channels=1)
        x = torch.randint(0, 2, (8, 1, 64, 64)).float()
        per_sample = per_sample_negative_elbo(model, [x], torch.device("cpu"))
        with torch.no_grad():
            mu, std = model.encode(x)
            out = model(x, kl_weight=1.0)
        assert per_sample.shape == (8,)
        # forward() reparameterises (adds sampling noise) and per_sample decodes
        # from mu deterministically, so this checks magnitude agreement (same
        # formula, same order of magnitude) rather than exact equality.
        assert abs(per_sample.mean() - out.loss.item()) < 0.05 * abs(out.loss.item())

    def test_deterministic_across_calls(self):
        """Decoding from mu (not a reparameterised sample) means calling twice
        on the same input must give bit-identical scores -- required for the
        attack's AUROC to be reproducible."""
        model = _vae("bce", in_channels=1)
        x = torch.randint(0, 2, (8, 1, 64, 64)).float()
        a = per_sample_negative_elbo(model, [x], torch.device("cpu"))
        b = per_sample_negative_elbo(model, [x], torch.device("cpu"))
        assert np.array_equal(a, b)

    def test_mse_path_runs(self):
        model = _vae("mse", in_channels=3)
        x = torch.rand(6, 3, 64, 64) * 2 - 1
        out = per_sample_negative_elbo(model, [x], torch.device("cpu"))
        assert out.shape == (6,)
        assert np.all(out >= 0)  # recon(MSE, sum) and KL are both nonneg


class TestMemorisationSignalEndToEnd:
    def test_a_model_trained_only_on_members_shows_a_gap(self):
        """The integration claim: fit a tiny VAE on ONE set of images, then
        check it scores those images lower loss than a disjoint held-out set
        it never saw -- the actual mechanism the attack depends on, not just
        the AUROC arithmetic above."""
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        member_x = torch.randint(0, 2, (64, 1, 64, 64), generator=g).float()
        nonmember_x = torch.randint(0, 2, (64, 1, 64, 64), generator=g).float()

        model = _vae("bce", in_channels=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(60):
            out = model(member_x, kl_weight=1.0)
            opt.zero_grad(); out.loss.backward(); opt.step()

        member_loss = per_sample_negative_elbo(model, [member_x], torch.device("cpu"))
        nonmember_loss = per_sample_negative_elbo(model, [nonmember_x], torch.device("cpu"))
        result = loss_threshold_mia(member_loss, nonmember_loss)
        assert result["auroc"] > 0.7, (
            f"expected a clear memorisation signal after overfitting to the "
            f"member set, got auroc={result['auroc']}"
        )
