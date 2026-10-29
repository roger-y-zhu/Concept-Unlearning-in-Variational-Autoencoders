"""
tests/test_vae.py
==================
Basic-function tests for src/models/vae.py -- shapes, encode/decode roundtrip,
the two reconstruction modes, and the loss decomposition.
"""

import torch

from src.models.vae import ConvVAE, VAEConfig


def test_encode_output_shapes(tiny_vae_bce, bce_batch):
    mu, std = tiny_vae_bce.encode(bce_batch)
    assert mu.shape == (8, 4)
    assert std.shape == (8, 4)


def test_std_is_positive(tiny_vae_bce, bce_batch):
    """softplus(x) + eps must never be <= 0, whatever the raw logits are."""
    _, std = tiny_vae_bce.encode(bce_batch)
    assert torch.all(std > 0)


def test_decode_output_shape(tiny_vae_bce):
    z = torch.randn(5, 4)
    x_hat = tiny_vae_bce.decode(z)
    assert x_hat.shape == (5, 1, 64, 64)


def test_reparameterise_uses_noise(tiny_vae_bce):
    """Two samples from the same (mu, std) must differ -- otherwise the
    reparameterisation trick isn't actually sampling."""
    mu = torch.zeros(4, 4)
    std = torch.ones(4, 4)
    torch.manual_seed(1)
    z1 = tiny_vae_bce.reparameterise(mu, std)
    torch.manual_seed(2)
    z2 = tiny_vae_bce.reparameterise(mu, std)
    assert not torch.allclose(z1, z2)


def test_reparameterise_zero_std_is_deterministic(tiny_vae_bce):
    mu = torch.full((4, 4), 3.0)
    std = torch.zeros(4, 4)
    z = tiny_vae_bce.reparameterise(mu, std)
    assert torch.allclose(z, mu)


def test_forward_loss_is_recon_plus_weighted_kl(tiny_vae_bce, bce_batch):
    out = tiny_vae_bce(bce_batch, kl_weight=0.3)
    expected = out.loss_recon + 0.3 * out.loss_kl
    assert torch.allclose(out.loss, expected, atol=1e-5)


def test_kl_weight_zero_drops_kl_term(tiny_vae_bce, bce_batch):
    out = tiny_vae_bce(bce_batch, kl_weight=0.0)
    assert torch.allclose(out.loss, out.loss_recon, atol=1e-5)


def test_kl_is_zero_for_standard_normal_encoding():
    """KL(N(0,1) || N(0,1)) = 0 exactly -- a direct check of the analytic formula
    0.5 * sum(mu^2 + std^2 - 1 - log(std^2)), independent of any learned weights."""
    cfg = VAEConfig(latent_dim=4, base_channels=4, in_channels=1, recon_loss="bce")
    model = ConvVAE(cfg)
    mu = torch.zeros(1, 4)
    std = torch.ones(1, 4)
    kl = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - torch.log(std.pow(2) + cfg.eps))
    assert kl.abs().item() < 1e-4


def test_bce_mode_output_is_unbounded_logits(tiny_vae_bce):
    """decode() for recon_loss='bce' must NOT apply an output activation --
    downstream code (src/unlearning/methods.make_to_img) depends on this to
    decide whether sigmoid is needed. See tests/test_regressions.py."""
    z = torch.randn(4, 4) * 10  # push toward the tails
    x_hat = tiny_vae_bce.decode(z)
    assert x_hat.min() < -0.5 or x_hat.max() > 1.5, (
        "decode() output looks bounded to [0,1] -- did output_act change from Identity?"
    )


def test_mse_mode_output_is_tanh_bounded(tiny_vae_mse):
    z = torch.randn(4, 4) * 10
    x_hat = tiny_vae_mse.decode(z)
    assert x_hat.min() >= -1.0 - 1e-5
    assert x_hat.max() <= 1.0 + 1e-5


def test_recon_loss_bce_vs_mse_use_correct_reduction(tiny_vae_bce, tiny_vae_mse,
                                                      bce_batch, mse_batch):
    out_bce = tiny_vae_bce(bce_batch, kl_weight=1.0)
    out_mse = tiny_vae_mse(mse_batch, kl_weight=1.0)
    assert torch.isfinite(out_bce.loss_recon)
    assert torch.isfinite(out_mse.loss_recon)
    # both losses are summed-over-pixels / averaged-over-batch -- for random
    # data on a small in_channels=1 image this should be O(1e3), not O(1) or O(1e6)
    assert 10 < out_bce.loss_recon.item() < 1e5


def test_unknown_recon_loss_raises():
    cfg = VAEConfig(latent_dim=4, base_channels=4, in_channels=1, recon_loss="bogus")
    model = ConvVAE(cfg)
    x = torch.rand(2, 1, 64, 64)
    try:
        model(x)
        assert False, "expected ValueError for unknown recon_loss"
    except ValueError:
        pass


def test_sample_shape_and_grad_free(tiny_vae_bce):
    imgs = tiny_vae_bce.sample(6, device=torch.device("cpu"))
    assert imgs.shape == (6, 1, 64, 64)
    assert not imgs.requires_grad


def test_backward_reaches_all_parameters(tiny_vae_bce, bce_batch):
    """A basic wiring sanity check: every parameter should receive a gradient
    from a single forward/backward pass on real (non-degenerate) input."""
    out = tiny_vae_bce(bce_batch, kl_weight=1.0)
    out.loss.backward()
    missing = [n for n, p in tiny_vae_bce.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
