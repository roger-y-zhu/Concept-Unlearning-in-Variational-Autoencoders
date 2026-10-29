"""beta-TCVAE TC-decomposition sanity tests (Stage 1, ledger: second-model plan)."""
import math
import torch

from src.models.vae import btc_kl_decomposition, ConvVAE, VAEConfig


def _terms(z, mu, logvar, N):
    return [t.item() for t in btc_kl_decomposition(z, mu, logvar, N)]


def test_tc_higher_when_latents_correlated():
    """TC must INCREASE when latent dims are correlated vs independent -- the property
    beta-TCVAE penalises. MWS is a biased estimator (absolute magnitude carries a
    constant offset), so this is a RELATIVE check; the exact-partition guarantee is
    test_decomposition_sums_to_mc_kl."""
    torch.manual_seed(0)
    N, M, D = 4096, 512, 4
    logvar = torch.zeros(M, D)                      # var=1: overlapping posteriors
    std = (0.5 * logvar).exp()
    mu_indep = torch.randn(M, D)
    z_indep = mu_indep + std * torch.randn(M, D)
    _, tc_i, _ = _terms(z_indep, mu_indep, logvar, N)
    mu_corr = mu_indep.clone()
    mu_corr[:, 2] = mu_corr[:, 0]
    mu_corr[:, 3] = mu_corr[:, 1]                    # dims 2,3 now copy 0,1
    z_corr = mu_corr + std * torch.randn(M, D)
    _, tc_c, _ = _terms(z_corr, mu_corr, logvar, N)
    assert math.isfinite(tc_i) and math.isfinite(tc_c)
    # Two latent dims made exact copies of two others at var=1 => aggregate covariance
    # [[2,1],[1,2]] per copied pair => closed-form TC difference ~= 2*0.5*(2ln2-ln3) ~= 0.29
    # nats. MWS carries a constant bias that cancels in the difference, so test THAT.
    diff = tc_c - tc_i
    assert 0.12 < diff < 0.50, f"TC diff {diff:.3f} should match closed-form ~0.29 (indep={tc_i:.2f} corr={tc_c:.2f})"


def test_decomposition_sums_to_mc_kl():
    """mi + tc + dim_kl must equal the Monte-Carlo KL E[log q(z|x) - log p(z)]."""
    torch.manual_seed(1)
    N, M, D = 20000, 256, 6
    mu = torch.randn(M, D); logvar = torch.randn(M, D) * 0.3
    std = (0.5 * logvar).exp(); z = mu + std * torch.randn(M, D)
    mi, tc, dk = btc_kl_decomposition(z, mu, logvar, N)
    log2pi = math.log(2 * math.pi)
    logqzx = (-0.5 * (log2pi + logvar + (z - mu) ** 2 / logvar.exp())).sum(1)
    logpz = (-0.5 * (log2pi + z ** 2)).sum(1)
    mc_kl = (logqzx - logpz).mean().item()
    assert abs((mi + tc + dk).item() - mc_kl) < 1e-3, "mi+tc+dim_kl must equal MC KL"


def test_btc_forward_runs_and_is_finite():
    """btc_forward produces a finite scalar loss with gradients; forward() untouched."""
    torch.manual_seed(2)
    m = ConvVAE(VAEConfig(latent_dim=8, base_channels=16, in_channels=1, recon_loss="bce"))
    x = torch.rand(16, 1, 64, 64)
    out = m.btc_forward(x, kl_weight=1.0, tc_beta=6.0, dataset_size=10000)
    assert torch.isfinite(out.loss) and out.loss.requires_grad
    out.loss.backward()
    # plain ELBO path still works and is independent
    out2 = m(x, kl_weight=1.0)
    assert torch.isfinite(out2.loss)
