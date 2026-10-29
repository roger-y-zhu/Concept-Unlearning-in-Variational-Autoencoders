"""RealNVP normalizing flow for 64x64 images -- the Stage-2 "second model" (a genuinely
different architecture family from the VAE: invertible, same-dimensional latent, EXACT
likelihood via change-of-variables).

Design
------
- Affine coupling layers (Dinh et al. 2017, RealNVP) with alternating checkerboard (and,
  for multi-channel inputs, channel-wise) binary masks.
- Each coupling's scale/translation (s, t) is produced by a small conv net whose final
  layer is ZERO-INITIALISED, so every coupling starts as the identity and training is
  stable. s is passed through tanh for boundedness.
- DEQUANTIZATION + LOGIT PREPROCESSING. Image pixels are quantized (dSprites is binary,
  2 levels), but a continuous flow cannot place finite density on discrete values -- doing
  so drives log-density to +inf and the NLL to -inf/NaN. So before the couplings we map
      x in [0,1] --dequantize--> v in [0,1) --constrain w=a+(1-2a)v--> logit(w) = y in R
  with the exact Jacobian tracked. The N(0,I) base then lives on R, matching y. The
  deterministic path (dequant noise = 0.5) is an exact bijection on [0,1], so encode/decode
  stay exactly invertible (geometry/attack apparatus); stochastic uniform dequant is used
  ONLY for the training objective (a standard variational dequantization bound).
- Standard-normal base density over all C*H*W dimensions.
- log p(x) = log N(z) + logdet_preprocess + sum_couplings logdet,  z = f(x);  x = f^{-1}(z).

Interface parallels ConvVAE where the evaluation apparatus needs it:
  encode(x) -> (z_flat, zeros)   # deterministic invertible latent (flattened); zero "std"
  decode(z) -> x in [0,1]        # exact inverse (deterministic path), differentiable (PGD)
  log_prob(x) -> (B,)            # exact per-example log-likelihood (nats, up to a constant)
  forward(x) -> FlowOutput       # carries .loss (mean NLL) for the trainer
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    in_channels: int = 1
    image_size: int = 64
    n_coupling: int = 8          # number of affine coupling layers
    hidden_channels: int = 64
    eps: float = 1e-6
    levels: int = 256            # pixel quantization levels (dSprites binary -> 2)
    alpha: float = 0.05          # logit boundary padding: w = alpha + (1-2 alpha) v

    def __post_init__(self):
        # full-dimensional invertible latent; exposed for trainer logging / interface parity
        self.latent_dim = self.in_channels * self.image_size * self.image_size


@dataclass
class FlowOutput:
    z: torch.Tensor
    x_hat: torch.Tensor          # reconstruction = decode(encode(x)); ~x by invertibility
    loss: torch.Tensor           # mean negative log-likelihood (nats/image)
    loss_recon: torch.Tensor     # NLL (alias, so trainer logging works)
    loss_kl: torch.Tensor        # 0 (no KL in a flow; kept for a uniform interface)


def _checkerboard(c, h, w, parity):
    m = torch.zeros(c, h, w)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == parity:
                m[:, i, j] = 1.0
    return m


def _channel_mask(c, h, w, parity):
    m = torch.zeros(c, h, w)
    half = c // 2
    if parity == 0:
        m[:half] = 1.0
    else:
        m[half:] = 1.0
    return m


class _STNet(nn.Module):
    """Conv net producing (s, t), each with `ch` channels. Final conv zero-initialised."""
    def __init__(self, ch, hidden):
        super().__init__()
        self.c1 = nn.Conv2d(ch, hidden, 3, padding=1)
        self.c2 = nn.Conv2d(hidden, hidden, 1)
        self.c3 = nn.Conv2d(hidden, 2 * ch, 3, padding=1)
        nn.init.zeros_(self.c3.weight); nn.init.zeros_(self.c3.bias)  # identity at init
        # a learned per-channel scale on the tanh'd s (helps expressivity; starts at 0)
        self.scale = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = self.c3(h)
        s_raw, t = h.chunk(2, dim=1)
        s = torch.tanh(s_raw) * torch.exp(self.scale)
        return s, t


class _AffineCoupling(nn.Module):
    def __init__(self, cfg: FlowConfig, mask):
        super().__init__()
        self.register_buffer("mask", mask.unsqueeze(0))   # (1,C,H,W)
        self.net = _STNet(cfg.in_channels, cfg.hidden_channels)

    def forward(self, x):
        m = self.mask
        s, t = self.net(x * m)
        s = s * (1 - m); t = t * (1 - m)          # only transform the un-masked half
        z = x * torch.exp(s) + t                  # masked entries: s=t=0 -> unchanged
        logdet = s.flatten(1).sum(1)              # (B,)
        return z, logdet

    def inverse(self, z):
        m = self.mask
        s, t = self.net(z * m)
        s = s * (1 - m); t = t * (1 - m)
        x = (z - t) * torch.exp(-s)
        return x


class RealNVP(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        c, hw = cfg.in_channels, cfg.image_size
        layers = []
        for i in range(cfg.n_coupling):
            if c > 1 and i % 2 == 1:
                mask = _channel_mask(c, hw, hw, (i // 2) % 2)
            else:
                mask = _checkerboard(c, hw, hw, i % 2)
            layers.append(_AffineCoupling(cfg, mask))
        self.couplings = nn.ModuleList(layers)
        self._log2pi = math.log(2.0 * math.pi)

    # -- dequantization + logit preprocessing (exact Jacobian) ------------
    def _preprocess(self, x, stochastic: bool):
        """x in [0,1] -> (y in R, pre_logdet (B,)). Maps quantized pixels off their discrete
        grid (uniform dequant) and through a logit so the density lives on R. The
        deterministic variant (stochastic=False) is an exact bijection on [0,1]."""
        L, a = self.cfg.levels, self.cfg.alpha
        noise = torch.rand_like(x) if stochastic else torch.full_like(x, 0.5)
        v = (x * (L - 1) + noise) / L                 # dequantize -> [0,1)
        w = a + (1.0 - 2.0 * a) * v                   # constrain -> [a, 1-a)
        y = torch.log(w) - torch.log1p(-w)            # logit -> R
        D = x[0].numel()                              # C*H*W
        const = D * (math.log((L - 1) / L) + math.log(1.0 - 2.0 * a))
        pre_logdet = const - (torch.log(w) + torch.log1p(-w)).flatten(1).sum(1)
        return y, pre_logdet

    def _postprocess(self, y):
        """Inverse of the deterministic _preprocess: y in R -> x in [0,1]."""
        L, a = self.cfg.levels, self.cfg.alpha
        w = torch.sigmoid(y)
        v = (w - a) / (1.0 - 2.0 * a)
        x = (v * L - 0.5) / (L - 1)
        return x.clamp(0.0, 1.0)

    # -- core transforms --------------------------------------------------
    def encode(self, x):
        """x (B,C,H,W) in [0,1] -> (z_flat (B, C*H*W), std). Deterministic invertible latent;
        a zero std is returned so this matches the ConvVAE (mu, std) interface the evaluation
        apparatus expects (probes/geometry/attack read `mu`; the flow latent has no noise)."""
        z, _ = self._preprocess(x, stochastic=False)
        for c in self.couplings:
            z, _ = c(z)
        zf = z.flatten(1)
        return zf, torch.zeros_like(zf)

    def decode(self, z):
        """z (flat (B, C*H*W) or (B,C,H,W)) -> x in [0,1], the exact inverse (differentiable,
        so PGD can attack through it)."""
        cfg = self.cfg
        x = z if z.dim() == 4 else z.reshape(-1, cfg.in_channels, cfg.image_size, cfg.image_size)
        for c in reversed(self.couplings):
            x = c.inverse(x)
        return self._postprocess(x)

    def _forward_logdet(self, x):
        """Deterministic x -> (z, total logdet of the full bijection x->z)."""
        z, logdet = self._preprocess(x, stochastic=False)
        for c in self.couplings:
            z, ld = c(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x):
        """Exact per-example log-likelihood (nats, up to the global dequant constant), via
        change of variables on the deterministic path."""
        z, logdet = self._forward_logdet(x)
        log_pz = (-0.5 * (z ** 2 + self._log2pi)).flatten(1).sum(1)
        return log_pz + logdet

    def forward(self, x, kl_weight: float = 1.0) -> FlowOutput:
        """kl_weight is accepted for interface parity with ConvVAE but unused (a flow has no
        KL term); the objective is the negative log-likelihood under variational
        dequantization (stochastic noise while training, deterministic at eval)."""
        y, logdet = self._preprocess(x, stochastic=self.training)
        z = y
        for c in self.couplings:
            z, ld = c(z)
            logdet = logdet + ld
        log_pz = (-0.5 * (z ** 2 + self._log2pi)).flatten(1).sum(1)
        nll = -(log_pz + logdet).mean()
        # NOTE: do NOT decode here -- a full inverse pass per training step would double the
        # cost for an x_hat the trainer never reads. Callers that want a reconstruction call
        # decode(z) explicitly.
        return FlowOutput(z=z, x_hat=z, loss=nll,
                          loss_recon=nll.detach(), loss_kl=nll.new_zeros(()))

    @torch.no_grad()
    def sample(self, n, device):
        c = self.cfg
        z = torch.randn(n, c.in_channels, c.image_size, c.image_size, device=device)
        return self.decode(z)
