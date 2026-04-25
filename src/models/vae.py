"""Convolutional VAE for 64×64 images.

Supports both:
  - dSprites  : 1-channel binary images  → BCE reconstruction loss
  - CelebA    : 3-channel RGB images     → MSE reconstruction loss

Architecture
------------
Encoder: 4 strided Conv2d layers → flatten → Linear → (μ, σ)
Decoder: Linear → reshape → 4 ConvTranspose2d layers → output

The implementation follows modern practices described in:
  Heidenreich (2024) "Modern PyTorch VAEs: A Detailed Implementation Guide"
  https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/

Key choices:
  - Softplus + ε for std parameterisation (stable gradients)
  - KL computed analytically: KL(N(μ,σ²) ‖ N(0,I)) = 0.5 Σ(μ² + σ² − 1 − log σ²)
  - Reconstruction loss summed over pixels, averaged over batch
  - β-VAE support via kl_weight argument to forward()
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class VAEOutput:
    z: torch.Tensor           # sampled latent vector   (B, latent_dim)
    mu: torch.Tensor          # encoder mean            (B, latent_dim)
    std: torch.Tensor         # encoder std             (B, latent_dim)
    x_hat: torch.Tensor       # decoder output          (B, C, H, W)
    loss: torch.Tensor        # total ELBO loss         (scalar)
    loss_recon: torch.Tensor  # reconstruction term     (scalar)
    loss_kl: torch.Tensor     # KL term                 (scalar)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VAEConfig:
    latent_dim: int = 10
    base_channels: int = 32      # doubled at each encoder stage
    in_channels: int = 1         # 1 for dSprites, 3 for CelebA
    recon_loss: str = "bce"      # "bce" (binary) or "mse" (continuous)
    eps: float = 1e-6            # numerical stability floor for std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvVAE(nn.Module):
    """Convolutional VAE for 64×64 images.

    Encoder spatial path: 64 → 32 → 16 → 8 → 4  (stride-2 convolutions)
    Bottleneck:           4×4×(base_channels×8) → latent_dim
    Decoder spatial path: 4 → 8 → 16 → 32 → 64  (transposed convolutions)

    For dSprites (binary): use recon_loss="bce", in_channels=1
    For CelebA   (RGB):    use recon_loss="mse", in_channels=3
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder_conv = nn.Sequential(
            # (B, C_in, 64, 64) → (B, c, 32, 32)
            nn.Conv2d(cfg.in_channels, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 2c, 16, 16)
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 4c, 8, 8)
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 8c, 4, 4)
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._flat_dim = c * 8 * 4 * 4  # e.g. 32*8*16 = 4096

        self.fc_mu  = nn.Linear(self._flat_dim, cfg.latent_dim)
        self.fc_std = nn.Linear(self._flat_dim, cfg.latent_dim)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.fc_decode = nn.Linear(cfg.latent_dim, self._flat_dim)

        self.decoder_conv = nn.Sequential(
            # (B, 8c, 4, 4) → (B, 4c, 8, 8)
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            # → (B, 2c, 16, 16)
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            # → (B, c, 32, 32)
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            # → (B, C_in, 64, 64)
            nn.ConvTranspose2d(c, cfg.in_channels, kernel_size=4, stride=2, padding=1),
        )

        # Final activation:
        #   BCE path → Identity; sigmoid is applied inside the loss function
        #   MSE path → Tanh maps decoder output to [-1, 1],
        #              matching CelebA's per-channel normalisation
        self.output_act = nn.Tanh() if cfg.recon_loss == "mse" else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map images to (μ, σ) in latent space."""
        h = self.encoder_conv(x).flatten(1)
        mu  = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + self.cfg.eps
        return mu, std

    def reparameterise(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(μ, σ²) using the reparameterisation trick."""
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vector to image space."""
        h = self.fc_decode(z).view(-1, self.cfg.base_channels * 8, 4, 4)
        return self.output_act(self.decoder_conv(h))

    def forward(self, x: torch.Tensor, kl_weight: float = 1.0) -> VAEOutput:
        """Full forward pass with ELBO computation.

        Args:
            x:         Input images (B, C, 64, 64).
                         dSprites: C=1, values in {0, 1}
                         CelebA:   C=3, values in [-1, 1] after normalisation
            kl_weight: β coefficient for the KL term (β-VAE / KL annealing).

        Returns:
            VAEOutput with all tensors and scalar losses.
        """
        mu, std = self.encode(x)
        z = self.reparameterise(mu, std)
        x_hat = self.decode(z)

        loss_recon = self._recon_loss(x, x_hat)

        # KL: analytical for N(μ,σ²) vs N(0,I)
        loss_kl = 0.5 * torch.sum(
            mu.pow(2) + std.pow(2) - 1.0 - torch.log(std.pow(2) + self.cfg.eps),
            dim=1,
        ).mean()

        loss = loss_recon + kl_weight * loss_kl

        return VAEOutput(
            z=z, mu=mu, std=std, x_hat=x_hat,
            loss=loss, loss_recon=loss_recon, loss_kl=loss_kl,
        )

    def _recon_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss, summed over pixels and averaged over batch.

        BCE for binary inputs (dSprites), MSE for continuous inputs (CelebA).
        Summing over spatial/channel dimensions keeps the magnitude comparable
        across different latent dimensionalities.
        """
        if self.cfg.recon_loss == "bce":
            # x_hat contains pre-sigmoid logits — numerically stable
            return F.binary_cross_entropy_with_logits(
                x_hat, x, reduction="sum"
            ) / x.size(0)
        elif self.cfg.recon_loss == "mse":
            return F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        else:
            raise ValueError(
                f"Unknown recon_loss '{self.cfg.recon_loss}'. Use 'bce' or 'mse'."
            )

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n images by sampling z ~ N(0, I)."""
        z = torch.randn(n, self.cfg.latent_dim, device=device)
        return self.decode(z)
