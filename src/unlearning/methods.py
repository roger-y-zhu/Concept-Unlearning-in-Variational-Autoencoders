"""
src/unlearning/methods.py
=========================
One class per unlearning method.  Each owns its hyperparameters, its run-directory
naming, any one-off setup, and its per-batch loss.  Nothing else.

The shared training loop lives in loop.py and the dataset differences live in
spec.py, so adding a method means writing one class here and registering it in
METHODS -- not forking a script per dataset.

Interface
---------
    add_args(parser)                  declare CLI hyperparameters
    run_name(args) -> str             run directory name, e.g. "lam0.0003_ep13"
    prepare(...)   -> dict            one-off setup (feature direction, RMU vector,
                                      parameter freezing); returns a context dict
    train_mode     -> bool            True  = model.train()  (BatchNorm stats update)
                                      False = model.eval()   (stats frozen)
    step(...)      -> (loss, metrics) the per-batch objective

A NOTE ON train_mode -- this is load-bearing
--------------------------------------------
Any method that compares the live model against a *frozen* copy must run the
live model in .eval().  Otherwise the live model uses BatchNorm batch statistics
while the frozen reference uses running statistics, so a term that should be
exactly zero at initialisation (e.g. "keep retain outputs unchanged") is not.

Measured on dSprites with identical weights:
    model.eval()  -> retain anchor 2.5e-12   correct
    model.train() -> retain anchor 208.0     wrecked every feature-unlearn run

Methods that only optimise the ELBO (NegGrad+, Finetune) have no frozen
reference and should keep train_mode = True.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def bn_stats_frozen(model):
    """Run a forward pass without letting it update BatchNorm running stats.

    Needed for any forward pass over the FORGET set in a train_mode method.
    BatchNorm buffers are not parameters and are in no method's objective, but
    a forward pass in train mode updates them from the batch -- so pushing
    concept-homogeneous batches (100% hearts) through the model rewrites the
    running statistics toward the forget distribution.  That is a side channel
    by which the forget set damages the model outside the loss entirely.

    It is not a small effect.  Measured on the dSprites baseline with NO
    optimiser, NO backward pass and NO weight change whatsoever (max |dW| was
    exactly 0), pushing 100 forget batches through in train mode moved retain
    val from 35.71 to 57.94.  The same 100 batches in eval mode leave it at
    35.82, and 100 RETAIN batches in train mode leave it at 37.02 -- so the
    damage is specific to homogeneous forget batches, not to forward passes.

    This also explains why NegGrad+ retain damage was almost independent of
    lambda across a 10x range (val 41.52 / 41.55 / 41.56 at lambda = 1e-5 /
    3e-5 / 1e-4): most of it was never coming from the gradient ascent.
    """
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.SyncBatchNorm))]
    was_training = [m.training for m in bns]
    try:
        for m in bns:
            m.eval()
        yield
    finally:
        for m, t in zip(bns, was_training):
            m.train(t)

ENCODER_PREFIXES = ("encoder_conv", "fc_mu", "fc_std")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def freeze_encoder(model) -> None:
    for name, p in model.named_parameters():
        if name.startswith(ENCODER_PREFIXES):
            p.requires_grad_(False)


def make_to_img(recon_loss: str):
    """Raw decoder output -> image space.

    bce (dSprites) emits unbounded logits (observed range [-256, +135]); MSE on
    those is dominated by saturated pixels, so apply sigmoid first.
    mse (CelebA) already ends in Tanh and is bounded to [-1, 1].
    """
    return torch.sigmoid if recon_loss == "bce" else (lambda raw: raw)


def img_mse(a, b):
    """MSE summed over pixels, averaged over batch -- matches the ELBO recon scale."""
    return F.mse_loss(a, b, reduction="sum") / a.size(0)


def img_l1(a, b):
    """L1 summed over pixels, averaged over batch.

    Moon et al. use the L1 norm for both the reconstruction and unlearning
    terms; img_mse is the project's original (L2) choice and remains the
    default so existing sweeps stay comparable.  See --recon-norm.
    """
    return F.l1_loss(a, b, reduction="sum") / a.size(0)


IMG_NORMS = {"l2": img_mse, "l1": img_l1}


def latent_mse(a, b):
    """MSE summed over latent dims, averaged over batch."""
    return F.mse_loss(a, b, reduction="sum") / a.size(0)


def make_to_unit(recon_loss: str):
    """make_to_img output -> [0, 1].

    SSIM's luminance term assumes non-negative intensities, so the tanh path
    ([-1, 1]) has to be rescaled before MS-SSIM.  The sigmoid path is already
    in range.  Only used for the perceptual term; the pixel loss stays in the
    model's native image space so its scale matches the ELBO.
    """
    return (lambda v: v) if recon_loss == "bce" else (lambda v: (v + 1.0) * 0.5)


# --- MS-SSIM -----------------------------------------------------------------
# Moon et al. add a perceptual term on top of the pixel loss specifically because
# pixel-level metrics produce blurry outputs.  Implemented here rather than
# pulled from pytorch-msssim to avoid a dependency for ~30 lines.
#
# Note on scales: the standard 5-scale weights need >=161px.  At 64x64 only
# three scales survive an 11px window (64 -> 32 -> 16), so `scales` defaults to
# 3 and the weights are renormalised.

_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(win_size, sigma, channels, device, dtype):
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.expand(channels, 1, win_size, win_size).contiguous()


def _ssim_maps(a, b, window, data_range):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ch = a.size(1)
    mu_a = F.conv2d(a, window, groups=ch)
    mu_b = F.conv2d(b, window, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = F.conv2d(a * a, window, groups=ch) - mu_a2
    var_b = F.conv2d(b * b, window, groups=ch) - mu_b2
    cov   = F.conv2d(a * b, window, groups=ch) - mu_ab
    cs   = (2 * cov + c2) / (var_a + var_b + c2)
    ssim = ((2 * mu_ab + c1) / (mu_a2 + mu_b2 + c1)) * cs
    return ssim.mean(dim=(1, 2, 3)), cs.mean(dim=(1, 2, 3))


def ms_ssim(a, b, *, data_range=1.0, win_size=11, scales=3, eps=1e-8):
    """Multi-scale SSIM in [0, 1], averaged over the batch.  Higher = more similar."""
    weights = torch.tensor(_MS_SSIM_WEIGHTS[:scales], device=a.device, dtype=a.dtype)
    weights = weights / weights.sum()
    window  = _gaussian_window(win_size, 1.5, a.size(1), a.device, a.dtype)

    levels = []
    for i in range(scales):
        ssim_i, cs_i = _ssim_maps(a, b, window, data_range)
        levels.append(ssim_i if i == scales - 1 else cs_i)
        if i < scales - 1:
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)

    stacked = torch.stack(levels, dim=0).clamp_min(eps)     # (scales, B)
    return (stacked ** weights.unsqueeze(1)).prod(dim=0).mean()


def translate_out(mu, direction, t: float = 0.0):
    """Set mu's coefficient along `direction` to `t`.

    Moon et al. eq.:  z_hat = z - (proj_{z_e}(z) - t) z_e, where t is the
    midpoint between the mean projections of feature-positive and -negative
    samples -- i.e. translate to the class boundary, not to the origin.
    t = 0 recovers a plain projection-out (project_out).
    """
    coef = (mu @ direction) / (direction * direction).sum()
    return mu - (coef - t).unsqueeze(1) * direction.unsqueeze(0)


def project_out(mu, direction):
    """Remove the component of mu along `direction`:  mu - proj(mu, direction)."""
    return translate_out(mu, direction, 0.0)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class UnlearnMethod:
    name: str
    train_mode: bool = True
    description: str = ""

    # True = the method is a single closed-form intervention, not an iterative
    # objective.  loop.py then builds NO optimiser, forces exactly one pass,
    # and runs step() under no_grad -- the pass measures the model prepare()
    # already modified.  Fisher scrubbing is the only such method here.
    one_shot: bool = False

    # Which direction of the reported `forget_loss` means "more forgetting".
    # Used by loop.py to select best_tradeoff.pt -- the checkpoint that forgets
    # most while keeping retain val inside budget.  Selecting on retain val
    # alone (best.pt) is monotonically optimised by doing nothing, so it always
    # returns the LEAST-unlearned epoch; Kurmanji et al. tune instead for "high
    # forget-error while not destroying retain-error".
    #   "higher" -- forget_loss is a negative-ELBO being pushed UP  (NegGrad+)
    #   "lower"  -- forget_loss is a distance to a target being pushed DOWN
    #               (Feature Unlearning, RMU)
    forget_better: str = "higher"

    def add_args(self, p) -> None:
        raise NotImplementedError

    def run_name(self, args) -> str:
        raise NotImplementedError

    def prepare(self, model, frozen, forget_loader, retain_loader, device, args) -> dict:
        """One-off setup.  Return a context dict passed to every step()."""
        return {}

    def trainable_params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    def step(self, model, frozen, ctx, x_r, x_f, args):
        """Return (loss_tensor, {metric_name: float}) for one batch."""
        raise NotImplementedError

    def hparams(self, args) -> dict:
        """Hyperparameters as a flat dict -- recorded in metrics.json."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NegGrad+ / Finetune -- same objective, different default lambda range
# ---------------------------------------------------------------------------

class _ElboAscent(UnlearnMethod):
    """Loss = retain_ELBO - lambda * forget_ELBO.

    lambda = 0 degenerates to pure retain-only finetuning, and the forget
    forward pass is skipped entirely.

    Relation to the paper (Kurmanji et al. 2023, arXiv:2302.09880)
    -------------------------------------------------------------
    NegGrad+ is defined there as a convex combination

        L = beta * L_retain - (1 - beta) * L_forget,        beta in [0, 1]

    which is this objective scaled by beta, with

        lambda = (1 - beta) / beta

    so the two differ only by an overall factor absorbed into the learning
    rate.  Their tuned values are beta = 0.95 (small-scale) and beta = 0.9999
    (large-scale), i.e. lambda = 5.3e-2 and 1.0e-4 -- and they report that
    "small beta result in explosion quickly".

    NOTE on the default below: lambda = 1.0 corresponds to beta = 0.5, which is
    an order of magnitude past their most aggressive setting and sits squarely
    in the regime they describe as exploding.  It is kept only because existing
    result directories are named after it; the Optuna range in
    experiments/unlearn/tune.py (1e-4 .. 5e-3) is the range worth trusting.
    """

    train_mode    = True
    forget_better = "higher"          # forget_loss is a negative ELBO, pushed up
    default_lambda: float
    default_epochs: int

    def add_args(self, p):
        p.add_argument("--lambda-forget", type=float, default=self.default_lambda,
                       help="Weight on forget ELBO (0 = retain-only finetuning)")
        p.add_argument("--epochs", type=int, default=self.default_epochs)

    def run_name(self, args):
        return f"lam{args.lambda_forget}_ep{args.epochs}"

    def hparams(self, args):
        return {"lambda_forget": args.lambda_forget, "epochs": args.epochs}

    def step(self, model, frozen, ctx, x_r, x_f, args):
        r_out = model(x_r, kl_weight=1.0)
        # Every forward pass over the forget set runs with BN buffers frozen --
        # see bn_stats_frozen. Gradients still flow to the weights; only the
        # running-statistic updates are suppressed. Without this the forget set
        # damages retain performance through BatchNorm rather than through the
        # objective, which made lambda almost irrelevant to the measured harm.
        if args.lambda_forget > 0:
            with bn_stats_frozen(model):
                f_out = model(x_f, kl_weight=1.0)
            loss  = r_out.loss - args.lambda_forget * f_out.loss
            f_val = f_out.loss.item()
        else:
            loss = r_out.loss
            # lambda = 0 is pure retain-only finetuning: the forget set must not
            # enter the objective.  It is still MEASURED, under no_grad, because
            # measurement is not optimisation -- without a forget_loss series in
            # history.json the shared budget rule (selection.tradeoff_epoch) has
            # nothing to rank epochs by, and pure finetuning drops out of every
            # cross-method comparison for a purely bookkeeping reason.  The extra
            # cost is one forward pass per step and no backward.
            #
            # bn_stats_frozen is load-bearing here, not decoration: no_grad
            # suppresses the backward graph but does NOT stop BatchNorm from
            # updating its running statistics, so measuring the forget ELBO
            # without it would silently inflict the full +22-nat BN damage on
            # pure finetuning -- a method whose whole point is that it never
            # touches the forget set.
            with torch.no_grad(), bn_stats_frozen(model):
                f_val = model(x_f, kl_weight=1.0).loss.item()
        return loss, {"retain_loss": r_out.loss.item(), "forget_loss": f_val}


class NegGrad(_ElboAscent):
    name           = "neg_grad"
    description    = "NegGrad+ : gradient ascent on the forget set"
    default_lambda = 1.0
    default_epochs = 5


class Finetune(_ElboAscent):
    name           = "finetune"
    description    = "Finetune : same objective, gentler lambda range"
    default_lambda = 0.0
    default_epochs = 10


# ---------------------------------------------------------------------------
# Feature Unlearning (Moon et al. 2023)
# ---------------------------------------------------------------------------

class FeatureUnlearn(UnlearnMethod):
    """Distil the decoder toward de-concepted reconstructions.

    z_e = mean_mu(forget) - mean_mu(retain) is computed once from the frozen
    encoder.  Forget latents have their z_e component projected out, and the
    decoder is trained to reproduce what the frozen decoder makes of the
    translated latent.  Retain outputs are anchored to the frozen decoder.

    The encoder is frozen by default -- Moon et al. fine-tune the decoder only,
    and this is also what makes "feature unlearning does not touch the encoder"
    a true statement rather than an approximate one.

    WHAT THAT MEANS FOR THE LATENT-GEOMETRY METRICS -- read before reporting
    -----------------------------------------------------------------------
    Frozen encoder + train_mode = False means no encoder parameter AND no
    BatchNorm buffer is ever written.  Verified on a real run (dSprites,
    alpha20.0_ep10_n1_ms0.5_pt1 vs baseline): max |delta| over encoder params
    and buffers = 0.0 exactly, while the decoder moved by 0.33.

    So mu(x) is BIT-IDENTICAL to the baseline's, and therefore so is every
    metric computed from mu: linear/MLP probe AUROC, kNN cohesion, cluster
    purity, silhouette, concept direction norm, and CKA-to-baseline (= 1.0
    exactly, not approximately).

    "Decoder-level unlearning leaves the latent geometry intact" is therefore
    an IDENTITY imposed by the method's design, not an empirical finding.  Two
    honest ways to report it:
      * state it as the design premise, and put the empirical weight on
        decoder-side evidence (src/evaluation/adversarial.py, reconstruction
        fidelity on forget images);
      * run --train-encoder, the only configuration where the question is
        non-trivial and the natural controlled twin for RMU (same distillation
        shape, both touching the encoder).

    Fidelity to the paper (arXiv:2303.05699)
    ----------------------------------------
    Their objective is

        L = alpha * (L_unlearn + L_percep) + L_recon
        L_recon   = (1 - sim(z,z_e)) * || g(z) - f(z) ||_1
        L_unlearn =      sim(z,z_e)  * || g(z) - f(z_hat) ||_1
        L_percep  =      sim(z,z_e)  * (1 - MS-SSIM(g(z), f(z_hat)))
        z_hat     = z - (proj_{z_e}(z) - t) z_e

    Three knobs below close the gap, all defaulting to this project's original
    behaviour so completed sweeps stay comparable:

      --recon-norm l1   use the paper's L1 instead of L2            (default l2)
      --ms-ssim W       add W * (1 - MS-SSIM) to the forget term    (default 0)
      --project-to threshold
                        translate to the class boundary t rather than to the
                        origin (t = 0)                              (default zero)

    SCALE WARNING on --ms-ssim
    --------------------------
    Eq. 5 weights L_percep at exactly 1 relative to L_unlearn, so "faithful"
    looks like --ms-ssim 1.0.  But the two terms are not on the same scale
    here: the pixel term is SUMMED over pixels, while (1 - MS-SSIM) is bounded
    in [0, 1].  Measured from real histories at the first epoch:

        dSprites  L_unlearn ~ 34.7  ->  --ms-ssim 0.5 contributes <= 1.4%
        CelebA    L_unlearn ~ 63.2  ->  --ms-ssim 0.5 contributes <= 0.8%

    Even at the paper's implied weight of 1.0 the perceptual term is under 3%
    of the forget objective, so an "MS-SSIM only" ablation at these weights
    will return a null -- and that null is a fact about loss scaling, not
    about perceptual loss.  To give the term real leverage, either normalise
    the pixel term to a per-pixel mean (which would break comparability with
    the ELBO scale, hence not the default) or sweep --ms-ssim into O(10-100).
    The same ambiguity exists in the paper, which suggests Moon et al.'s
    ||.||_1 may be a mean rather than a sum.

    Two deviations are deliberate and not flagged:
      * They sample z from the prior and gate each term by a learned classifier
        sim(z,z_e); we encode real images and use the ground-truth forget/retain
        labels, which is the hard version of the same gate and is available to
        us because our concept labels are exact.
      * Their reported alpha is 3 (MNIST VAE) and 300 (CelebA VDVAE), far above
        the 0.25-20 this project has swept.  Loss normalisation differs, so the
        numbers are not directly comparable, but the direction is worth heeding.
    """

    name          = "feature_unlearn"
    description   = "Feature Unlearning (Moon et al. 2023), decoder-only"
    train_mode    = False        # see module docstring -- BatchNorm
    forget_better = "lower"      # forget_loss = distance to the de-concepted target

    def add_args(self, p):
        p.add_argument("--alpha", type=float, default=1.0,
                       help="Weight on the forget term relative to the retain term")
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--elbo-anchor", type=float, default=0.0,
                       help="Weight on retain ELBO (0 = faithful to Moon et al.; "
                            ">0 pins the model to a valid VAE)")
        p.add_argument("--train-encoder", action="store_true",
                       help="Unfreeze the encoder (ablation)")
        p.add_argument("--recon-norm", choices=("l2", "l1"), default="l2",
                       help="Pixel norm. l1 matches Moon et al.; l2 is this "
                            "project's original choice (default)")
        p.add_argument("--ms-ssim", type=float, default=0.0,
                       help="Weight on the paper's MS-SSIM perceptual term "
                            "(0 = off, the original behaviour)")
        p.add_argument("--project-to", choices=("zero", "threshold"), default="zero",
                       help="Translate forget latents to projection 0 (default) or "
                            "to the positive/negative midpoint t (Moon et al.)")

    def run_name(self, args):
        name = f"alpha{args.alpha}_ep{args.epochs}"
        # Suffixes are appended only when a flag departs from the original
        # behaviour, so existing result directories keep their exact names.
        # Each must end in a digit to stay parseable by
        # nb_utils.discover_sweep_keys.
        if getattr(args, "recon_norm", "l2") == "l1":
            name += "_n1"
        if getattr(args, "ms_ssim", 0.0) > 0:
            name += f"_ms{args.ms_ssim}"
        if getattr(args, "project_to", "zero") == "threshold":
            name += "_pt1"
        # train_encoder and elbo_anchor used to be invisible here, so a
        # --train-encoder run wrote to the same directory as its frozen-encoder
        # twin and was then SKIPPED by loop.py's final.pt marker -- silently
        # returning the wrong model.  They are part of the run identity.
        if getattr(args, "train_encoder", False):
            name += "_te1"
        if getattr(args, "elbo_anchor", 0.0) > 0:
            name += f"_ea{args.elbo_anchor}"
        return name

    def hparams(self, args):
        return {"alpha": args.alpha, "epochs": args.epochs,
                "recon_norm":    getattr(args, "recon_norm", "l2"),
                "ms_ssim":       getattr(args, "ms_ssim", 0.0),
                "project_to":    getattr(args, "project_to", "zero"),
                "train_encoder": bool(getattr(args, "train_encoder", False)),
                "elbo_anchor":   getattr(args, "elbo_anchor", 0.0)}

    def prepare(self, model, frozen, forget_loader, retain_loader, device, args):
        if not getattr(args, "train_encoder", False):
            freeze_encoder(model)

        @torch.no_grad()
        def _mean_mu(loader):
            acc, n = None, 0
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
                mu, _ = frozen.encode(x)
                acc = mu.sum(0) if acc is None else acc + mu.sum(0)
                n += x.size(0)
            return acc / n

        frozen.eval()
        mean_f, mean_r = _mean_mu(forget_loader), _mean_mu(retain_loader)
        z_e = mean_f - mean_r

        # t = midpoint of the two class means' projections onto z_e.  With
        # z_e = mean_f - mean_r the projections are ||z_e||^-2 <mean, z_e>, so
        # this is exactly the paper's "average projection value of the positive
        # and negative samples".
        if getattr(args, "project_to", "zero") == "threshold":
            denom = (z_e * z_e).sum()
            t = float(0.5 * ((mean_f @ z_e) / denom + (mean_r @ z_e) / denom))
        else:
            t = 0.0

        info = f"z_e norm = {z_e.norm().item():.4f} | translate to proj = {t:.4f}"
        if getattr(args, "ms_ssim", 0.0) > 0:
            info += f" | MS-SSIM weight {args.ms_ssim}"
        return {"z_e": z_e, "t": t,
                "to_img":  make_to_img(model.cfg.recon_loss),
                "to_unit": make_to_unit(model.cfg.recon_loss),
                "pix":     IMG_NORMS[getattr(args, "recon_norm", "l2")],
                "info": info}

    def step(self, model, frozen, ctx, x_r, x_f, args):
        to_img, z_e, pix = ctx["to_img"], ctx["z_e"], ctx["pix"]

        mu_f, _  = model.encode(x_f)
        recon_f  = to_img(model.decode(mu_f))
        with torch.no_grad():
            target_f = to_img(frozen.decode(translate_out(mu_f, z_e, ctx["t"])))
        loss_f = pix(recon_f, target_f)
        if getattr(args, "ms_ssim", 0.0) > 0:
            to_unit = ctx["to_unit"]
            loss_f = loss_f + args.ms_ssim * (
                1.0 - ms_ssim(to_unit(recon_f), to_unit(target_f), data_range=1.0))

        mu_r, _  = model.encode(x_r)
        recon_r  = to_img(model.decode(mu_r))
        with torch.no_grad():
            target_r = to_img(frozen.decode(mu_r))
        loss_r = pix(recon_r, target_r)

        loss = args.alpha * loss_f + loss_r
        if getattr(args, "elbo_anchor", 0.0) > 0:
            loss = loss + args.elbo_anchor * model(x_r, kl_weight=1.0).loss

        return loss, {"forget_loss": loss_f.item(), "retain_loss": loss_r.item()}


# ---------------------------------------------------------------------------
# RMU -- Representation Misdirection for Unlearning
# ---------------------------------------------------------------------------

class RMU(UnlearnMethod):
    """Steer forget representations toward a fixed random direction.

        loss_forget = || mu(x_f)      -  c * u        ||^2
        loss_retain = || mu(x_r)      -  mu_frozen(x_r) ||^2
        loss        = loss_forget + alpha * loss_retain

    u is a random unit vector in latent space, fixed for the whole run.

    Unlike Feature Unlearning this deliberately trains the ENCODER -- that is the
    entire point.  The two methods otherwise share a shape (distillation against
    a frozen reference), which makes them a controlled pair for the thesis
    question: does an encoder-level intervention move the latent geometry when a
    decoder-level one does not?

    Still runs in .eval() mode: the retain term compares against the frozen
    encoder, so BatchNorm must use running statistics in both.

    Calibrating --alpha (measured: dSprites, c=1, 1 epoch, baseline val 36.9)
    -----------------------------------------------------------------------
        alpha      retain val      forget loss
            1         1781.5              2.28     model destroyed
           10          128.2              5.34
          100           37.5              6.98     retain intact, ~no forgetting
         1000           35.8              7.56     ~no forgetting

    The forget term starts near 7.25 (||mu_f||^2 + c^2, mu_f norm ~2.5), so
    alpha >= 100 means the encoder barely moves and nothing is unlearned, while
    alpha = 1 collapses the model.  The usable trade-off is roughly 10-100.

    The collapse mechanism is worth understanding: this loss never touches the
    DECODER, so the decoder receives no gradient at all.  A large encoder shift
    therefore leaves the decoder calibrated for a latent distribution that no
    longer exists, and reconstruction quality falls off a cliff.  The retain
    anchor is the only thing holding the encoder in place.

    (That part is faithful, incidentally: in the paper everything downstream of
    layer l is likewise never updated.  The collapse is a property of the
    method, not of this port.)

    Fidelity to the paper (arXiv:2403.03218 and the reference implementation
    at github.com/centerforaisafety/wmdp)
    ----------------------------------------------------------------------
    Matching: loss shape L_f + alpha*L_r, target c*u with ||u|| = 1, retain
    anchored to a frozen reference, AdamW at lr 5e-5.

    Diverging:
      * STEERING MAGNITUDE c -- the biggest gap, and the least explored.
        Section 4 states the mechanism outright: "increasing the norm of the
        model's activations on hazardous data in earlier layers makes it
        difficult for later layers to process the activations".  Appendix
        B.7.4 tunes c to 6.5 / 300 / 300 across their three models -- a 46x
        range, treated as a primary hyperparameter alongside alpha
        (1200 / 350 / 1600).

        Every run in this project is c = 1.0.  Measured on the dSprites
        baseline over 8192 test images, mean ||mu_f|| = 2.55, so the target
        c*u has norm 1 and sits INSIDE the typical radius: this is norm
        DEFLATION, the opposite of the paper's stated mechanism.  It also
        entangles the alpha findings -- "alpha >= 100 means nothing is
        unlearned" is a statement about alpha *at c = 1*, and the forget
        gradient scales with c, so the usable alpha range moves with it.
        Sweeping c against alpha is the single most informative missing
        experiment.
      * TRAINING BUDGET.  The reference implementation runs `for epoch in
        range(1)` over at most `max_num_batches` = 80 (150 in some configs).
        One "epoch" here is 768 retain batches on dSprites and 1331 on CelebA,
        so even --epochs 1 is roughly an order of magnitude more optimisation
        than RMU is designed to receive.  This is the likely explanation for
        every sweep run peaking at epoch 1 and decaying thereafter -- it is
        overtraining, not a property of the method.  Use --max-batches to
        express the budget in the paper's units.
      * u's DISTRIBUTION.  The reference samples torch.rand -- Uniform[0,1),
        so u lies in the positive orthant -- then normalises.  This port used
        torch.randn (isotropic Gaussian).  --u-dist uniform matches the paper;
        gaussian remains the default so completed sweeps stay comparable.
      * PARAMETER SUBSET.  The paper updates only layers l-2..l; this updates
        the whole encoder.  Left as-is: the VAE encoder is four conv blocks,
        and restricting it to the final one would be an arbitrary analogy.
      * REDUCTION.  The reference uses mse_loss with mean reduction; latent_mse
        sums over latent dims.  Both terms are scaled identically, so the
        optimum is unchanged and only the effective learning rate differs.
    """

    name          = "rmu"
    description   = "RMU : steer forget latents toward a random direction"
    train_mode    = False        # frozen reference -- see module docstring
    forget_better = "lower"      # forget_loss = distance to the steering target

    def add_args(self, p):
        p.add_argument("--c", type=float, default=1.0,
                       help="Steering magnitude: forget latents are pulled toward c*u")
        p.add_argument("--alpha", type=float, default=50.0,
                       help="Weight on the retain anchor. THIS is the sensitive knob -- "
                            "see the calibration table in the class docstring")
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--u-dist", choices=("gaussian", "uniform"), default="gaussian",
                       help="Steering-vector distribution before normalisation. "
                            "uniform = Uniform[0,1), matching the reference "
                            "implementation; gaussian is this project's original")
        p.add_argument("--u-seed", type=int, default=0,
                       help="Seed for the random steering vector u. Every run in "
                            "this project used the default (0) until this flag "
                            "existed -- every reported collapse/self-correction "
                            "trajectory rests on a single fixed direction. Vary "
                            "this to check whether findings hold for other u.")

    def run_name(self, args):
        # alpha is the primary sweep axis, so it takes the {prefix}{value} slot that
        # nb_utils.discover_sweep_keys parses (prefix="a").  c is included so that
        # sweeping it produces distinct directories rather than silently colliding
        # with the skip-if-done marker.
        name = f"a{args.alpha}_c{args.c}_ep{args.epochs}"
        if getattr(args, "u_dist", "gaussian") == "uniform":
            name += "_u1"
        if getattr(args, "u_seed", 0) != 0:
            name += f"_s{args.u_seed}"
        return name

    def hparams(self, args):
        return {"c": args.c, "alpha": args.alpha, "epochs": args.epochs,
                "u_dist": getattr(args, "u_dist", "gaussian"),
                "u_seed": getattr(args, "u_seed", 0)}

    def prepare(self, model, frozen, forget_loader, retain_loader, device, args):
        # Fixed random unit vector, seeded so a rerun of the same config matches.
        g = torch.Generator(device="cpu").manual_seed(getattr(args, "u_seed", 0))
        d = model.cfg.latent_dim
        if getattr(args, "u_dist", "gaussian") == "uniform":
            u = torch.rand(d, generator=g).to(device)          # paper: U[0,1)
        else:
            u = torch.randn(d, generator=g).to(device)
        u = u / u.norm()
        frozen.eval()
        return {"u": u,
                "info": f"steering vector ||u||=1 "
                        f"({getattr(args, 'u_dist', 'gaussian')}, "
                        f"seed={getattr(args, 'u_seed', 0)}), c={args.c}"}

    def step(self, model, frozen, ctx, x_r, x_f, args):
        u = ctx["u"]

        mu_f, _ = model.encode(x_f)
        target  = (args.c * u).unsqueeze(0).expand_as(mu_f)
        loss_f  = latent_mse(mu_f, target)

        mu_r, _ = model.encode(x_r)
        with torch.no_grad():
            mu_r_frozen, _ = frozen.encode(x_r)
        loss_r = latent_mse(mu_r, mu_r_frozen)

        loss = loss_f + args.alpha * loss_r
        return loss, {"forget_loss": loss_f.item(), "retain_loss": loss_r.item()}


# ---------------------------------------------------------------------------
# Fisher scrubbing (Golatkar et al. 2020)
# ---------------------------------------------------------------------------

def _fmt_num(v: float) -> str:
    """Decimal string for a run-directory name, never scientific notation.

    f"{1e-05}" is '1e-05', and the 'e'/'-' break the `[0-9.]+` parameter slot
    that nb_utils.discover_sweep_keys parses.  format_float_positional always
    emits positional digits and trims trailing zeros.
    """
    import numpy as np
    return np.format_float_positional(float(v), trim="-")


class FisherScrub(UnlearnMethod):
    """Fisher scrubbing / "Fisher forgetting" (Golatkar et al. 2020, arXiv:1911.04933).

    A ONE-SHOT weight perturbation, not a training objective.  Gaussian noise
    is added to every weight, with per-weight magnitude scaled DOWN wherever
    the retain set says that weight matters:

        S(w) = w + (lambda * sigma_h^2)^(1/4) * F^(-1/4) * n,   n ~ N(0, I)

    F is the diagonal Fisher Information Matrix evaluated at w over the RETAIN
    set.  Weights with large Fisher are important for retain performance and
    receive little noise; weights with small Fisher are barely constrained by
    the retain data -- and are therefore free to have encoded forget-set
    specifics -- so they get scrambled.  The paper's Figure 3 shows exactly
    this: class-specific filters are destroyed, shared filters survive.

    Because there is no iterative objective, this class sets one_shot = True.
    loop.py then builds no optimiser and runs a single measuring pass; all the
    actual work happens in prepare().  `--epochs` is accepted but ignored.
    The tuning axis is --noise-scale, not epochs.

    Fidelity to the paper
    ---------------------
    Matching:
      * The NOISE-ONLY form is the paper's own deep-network recipe, not a
        simplification invented here.  Eq. (7) is a noisy Newton step,
        `w - B^-1 grad L_Dr(w) + (lambda sigma_h^2)^1/4 B^-1/4 eps`, but S4
        ("Fisher forgetting") drops the Newton term outright: "the diagonal is
        not a good enough approximation of B for a full Newton step ...
        However, the diagonal is still a good approximation for the purpose of
        adding noise.  Therefore we simplify the procedure and take h(w) = w".
      * F^(-1/4), not F^(-1/2).  Proposition 4 gives the noise COVARIANCE as
        Sigma = sqrt(lambda sigma_h^2) B^(-1/2); drawing n ~ N(0, Sigma) as
        Sigma^(1/2) eps gives the quarter power on the multiplier.
      * F is computed on the retain set at the current w, per S4.
      * (lambda * sigma_h^2)^(1/4) is a single scalar in the formula, so it is
        exposed as one knob, --noise-scale, rather than as two unidentifiable
        hyperparameters.

    Diverging:
      * EMPIRICAL rather than true Fisher.  Eq. (8) defines the FIM with the
        label sampled from the MODEL, `y ~ p_w(y|x)`.  Here the "label" is the
        image itself, and this project's decoder does not parameterise its
        output variance -- the CelebA/MSE path is a Gaussian likelihood whose
        variance is implicit -- so p_w(x|z) is not fully specified and
        sampling from it would mean inventing a variance.  The empirical
        Fisher, E_retain[(grad of the per-sample negative ELBO)^2], is
        well-defined on both the BCE and MSE paths and is what is used.
      * log p_w(y|x) -> the VAE has no conditional label likelihood, so the
        per-sample negative ELBO (a bound on -log p_w(x)) stands in for it.
      * DAMPING.  F entries reach ~0 for weights the retain set never
        exercises, and F^(-1/4) would then be inf.  --fisher-eps is added
        before inversion.  The paper does not discuss this; no implementation
        can skip it.
      * SAMPLE BUDGET.  The expectation in eq. (8) is over the whole retain
        set; --fisher-samples caps it (per-sample gradients need batch size 1,
        so the cost is linear in the count).

    Not scrubbed: BatchNorm running statistics.  They are buffers, not
    parameters, so they have no gradient and no Fisher entry -- there is no
    principled per-weight scale for them.

    Calibrating --noise-scale (measured: dSprites, FIM over 2000 retain
    samples, baseline val 35.79 / forget ELBO 38.63, budget 40.59)
    --------------------------------------------------------------------
        scale    retain val   forget ELBO   d_retain  d_forget  selectivity
        0.004        36.18         39.10       +0.39     +0.47        1.19
        0.006        36.69         39.65       +0.90     +1.02        1.13
        0.008        37.71         40.96       +1.93     +2.32        1.21
        0.010        39.43         42.66       +3.64     +4.03        1.11
        0.012        42.46         45.15       +6.67     +6.52        0.98  OVER
        0.015        50.63         51.51      +14.84    +12.87        0.87  OVER
        0.020        84.33         75.82      +48.55    +37.18        0.77  OVER

    Two things to know before using this.

    NARROW WINDOW.  Nothing measurable happens below ~4e-3, and the model is
    destroyed by 3e-2 (val 945).  The whole usable range is under one order of
    magnitude, so sweep it finely.

    SELECTIVITY IS ~1.  The last column is d_forget / d_retain -- how much
    more the forget set is damaged than the retain set.  It sits at 1.1-1.2
    across the entire in-budget range, i.e. Fisher scrubbing here is close to
    INDISCRIMINATE damage, and past the budget it inverts (<1: retain is hurt
    MORE than forget).  That is a property of this model, not a bug in the
    port: dSprites hearts share nearly all their low-level features with
    squares and ellipses, so few weights are heart-specific enough for the
    retain-set FIM to leave unprotected.  Do not read a rising forget ELBO as
    forgetting without checking it against the retain column.

    SEED SENSITIVITY.  The perturbation is a random draw, so one run is one
    sample of the method.  At scale=0.01 across --fisher-seed 0..4:
    val = 39.41 / 38.88 / 43.10 / 42.56 / 40.50, i.e. only 3 of 5 seeds stay
    inside the retain budget at all.  The same knife-edge behaviour found for
    RMU's steering vector.  Vary --fisher-seed before trusting any single
    result.

    WHY THIS METHOD DOES NOT APPROXIMATE THE ORACLE HERE -- read before
    reporting it as underperforming
    ----------------------------------------------------------------------
    Measured on real checkpoints, this method leaves latent geometry
    essentially untouched on dSprites and moves it AWAY from the oracle on
    CelebA. That is not a bug in this port and no noise_scale fixes it:

    1. h(w) = w means S(w) = w + n with E[n] = 0, so for ANY fixed target
       E||w_s - target||^2 = ||w - target||^2 + E||n||^2.  Scrubbing provably
       adds distance to the oracle.  Verified: the measured
       ||w_scrub - w_oracle|| matches that Pythagorean prediction to four
       significant figures, and cos(n, w_oracle - w_baseline) = -0.0014.
       The paper's objective (eq. 5, fig. 1) is to make P(S(w)|D) and
       P(S0(w)|Dr) indistinguishable by inflating the noise covariance --
       information destruction, not oracle approximation.  Different goal.
    2. There is no concept-specific subnetwork to target.  Diagonal FIMs over
       retain vs forget on the dSprites baseline: cosine 0.988, Spearman
       0.950, top-1% overlap 0.894.  The weights that receive the most noise
       (low F_retain) are the ones the forget set barely uses either.
    3. The paper assumes Df is "a small portion of the overall dataset ...
       lest one is better-off re-training than forgetting", and that training
       is stable under removing "a few samples".  dSprites hearts are 33% of
       the data; CelebA eyeglasses 6.5%.  Their experiments forget 100 images
       or one identity.

    Judge this method by its own metrics (the NATs information bound of their
    Proposition 2, re-learn time), or restore the dropped Newton term
    h(w) = w - B^-1 grad L_Dr(w), which is the only part that moves toward the
    retain optimum.  See docs/FISHER_DIAGNOSIS.md for the full diagnosis.

    All of the ELBO numbers above are OUTPUT-side quantities.  The project's
    actual question is about latent geometry, so evaluate real checkpoints
    with src/evaluation/separation.py and CKA-to-oracle too.
    """

    name          = "fisher"
    description   = "Fisher scrubbing (Golatkar et al. 2020), one-shot FIM-scaled noise"
    train_mode    = False    # BN running stats must not drift, and the FIM pass
                             # runs at batch size 1 -- train mode would be invalid
    one_shot      = True
    forget_better = "higher"  # forget_loss is a negative ELBO: higher = more forgotten

    DEFAULT_SAMPLES = 2000

    def add_args(self, p):
        p.add_argument("--noise-scale", type=float, default=1e-3,
                       help="The paper's (lambda*sigma_h^2)^(1/4) as a single scalar. "
                            "THE tuning knob: 0 = no-op, larger = more forgetting and "
                            "more collateral damage (their Figure 2 trade-off)")
        p.add_argument("--fisher-samples", type=int, default=self.DEFAULT_SAMPLES,
                       help="Retain samples used to estimate the diagonal FIM. "
                            "Per-sample gradients require batch size 1, so cost is "
                            "linear in this")
        p.add_argument("--fisher-eps", type=float, default=1e-8,
                       help="Added to F before inversion. Without it, weights the "
                            "retain set never exercises have F~0 and F^(-1/4)=inf")
        p.add_argument("--fisher-seed", type=int, default=0,
                       help="Seed for the noise draw n. Fisher scrubbing is a random "
                            "perturbation, so a single draw is one sample of the "
                            "method -- vary this to check a finding is not seed luck")
        p.add_argument("--epochs", type=int, default=1,
                       help="IGNORED (one-shot method); accepted so the shared loop "
                            "and CLI stay uniform")

    def run_name(self, args):
        name = f"sig{_fmt_num(args.noise_scale)}_ep1"
        if getattr(args, "fisher_samples", self.DEFAULT_SAMPLES) != self.DEFAULT_SAMPLES:
            name += f"_fs{args.fisher_samples}"
        if getattr(args, "fisher_seed", 0) != 0:
            name += f"_s{args.fisher_seed}"
        return name

    def hparams(self, args):
        return {"noise_scale":    args.noise_scale,
                "fisher_samples": getattr(args, "fisher_samples", self.DEFAULT_SAMPLES),
                "fisher_eps":     getattr(args, "fisher_eps", 1e-8),
                "fisher_seed":    getattr(args, "fisher_seed", 0)}

    # -- the two halves of the algorithm ------------------------------------

    @staticmethod
    def diagonal_fisher(model, retain_loader, device, n_samples: int) -> tuple[list, int]:
        """Diagonal empirical FIM over the retain set: mean of SQUARED PER-SAMPLE
        gradients of the negative ELBO.

        The per-sample part is load-bearing and is the classic way to get this
        wrong.  E[(grad L_i)^2] is not (grad E[L_i])^2 -- the latter lets
        opposing per-sample gradients cancel before squaring, which understates
        the importance of exactly the specialised weights this method needs to
        identify.  Hence batch size 1 rather than a batched shortcut.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        fisher = [torch.zeros_like(p) for p in params]
        seen = 0

        for batch in retain_loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            for i in range(x.size(0)):
                loss = model(x[i:i + 1], kl_weight=1.0).loss
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                with torch.no_grad():
                    for acc, g in zip(fisher, grads):
                        if g is not None:
                            acc += g.detach().pow(2)
                seen += 1
                if seen >= n_samples:
                    break
            if seen >= n_samples:
                break

        if seen == 0:
            raise RuntimeError("retain loader yielded no samples -- cannot estimate the FIM")
        with torch.no_grad():
            for acc in fisher:
                acc /= seen
        return fisher, seen

    @staticmethod
    @torch.no_grad()
    def apply_scrub(model, fisher: list, device, *, noise_scale: float,
                    eps: float, seed: int) -> dict:
        """In-place `w += noise_scale * (F + eps)^(-1/4) * n`, n ~ N(0, I)."""
        # CPU generator then move, matching RMU's steering vector -- keeps the
        # draw identical across CPU and CUDA runs.
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        params = [p for p in model.parameters() if p.requires_grad]

        sq_sum, n_el = 0.0, 0
        for p, f in zip(params, fisher):
            sigma = (f + eps).pow(-0.25)
            noise = torch.randn(p.shape, generator=g).to(device=p.device, dtype=p.dtype)
            delta = noise_scale * sigma * noise
            p.add_(delta)
            sq_sum += float(delta.pow(2).sum())
            n_el += delta.numel()

        return {"rms_delta": (sq_sum / max(n_el, 1)) ** 0.5}

    # -- UnlearnMethod interface --------------------------------------------

    def prepare(self, model, frozen, forget_loader, retain_loader, device, args):
        # `frozen` is unused: unlike Feature Unlearning and RMU, this method
        # compares against nothing -- the FIM is computed from the live model.
        n_samples = int(getattr(args, "fisher_samples", self.DEFAULT_SAMPLES))
        fisher, seen = self.diagonal_fisher(model, retain_loader, device, n_samples)

        flat = torch.cat([f.flatten() for f in fisher])
        stats = self.apply_scrub(
            model, fisher, device,
            noise_scale=float(args.noise_scale),
            eps=float(getattr(args, "fisher_eps", 1e-8)),
            seed=int(getattr(args, "fisher_seed", 0)),
        )

        info = (f"FIM over {seen} retain samples | "
                f"F: median={flat.median().item():.3e} max={flat.max().item():.3e} "
                f"zero-frac={(flat == 0).float().mean().item():.3f} | "
                f"scrub RMS delta={stats['rms_delta']:.3e}")
        return {"info": info, "fisher_samples_used": seen, **stats}

    def step(self, model, frozen, ctx, x_r, x_f, args):
        """Measurement only -- loop.py calls this under no_grad for one-shot
        methods, and the returned tensor is never backpropagated."""
        r_out = model(x_r, kl_weight=1.0)
        f_out = model(x_f, kl_weight=1.0)
        return r_out.loss, {"retain_loss": r_out.loss.item(),
                            "forget_loss": f_out.loss.item()}


# ---------------------------------------------------------------------------
# Linear concept erasure -- LEACE (Belrose et al. 2023) / INLP (Ravfogel 2020)
# ---------------------------------------------------------------------------

class _LinearConceptErasure(UnlearnMethod):
    """Shared plumbing for the two closed-form representation erasers.

    Both fit an affine map r(mu) = P mu + b on the TRAIN-split posterior means
    (encoded by the frozen baseline) and fold it exactly into fc_mu -- see
    src/unlearning/erasure.py for the math, the paper fidelity notes, and why
    the fold is exact rather than approximate.

    One-shot like Fisher scrubbing: all work happens in prepare(), the single
    loop pass only measures the already-erased model.  Unlike Fisher there is
    NO random draw -- both erasers are deterministic given the data, so there
    is no seed-variance study to run; that determinism is itself a property
    worth reporting next to Fisher's 2/3-in-budget seed lottery.

    Where these sit in the thesis's method taxonomy: they are the purpose-built
    "remove linearly decodable information" family.  If latent geometry moved
    only when linear decodability moved, these would be the strongest methods
    on the geometry metrics.  They are instead the natural probe of that link:
    LEACE guarantees linear probe AUROC ~ 0.5 while provably changing the
    representation as little as possible -- so every OTHER metric (MLP probe,
    kNN cohesion, CKA-to-oracle, probe-recovery curves) measures what linear
    guardedness does NOT buy you.
    """

    train_mode    = False     # nothing trains; eval-mode measuring pass only
    one_shot      = True
    forget_better = "higher"  # forget_loss is a negative ELBO, as for Fisher

    def add_args(self, p):
        p.add_argument("--fit-samples", type=int, default=0,
                       help="Cap on samples PER LOADER (forget and retain "
                            "separately) used to fit the eraser. 0 = use all. "
                            "Capping distorts the class prior inside Sigma_XX, "
                            "so leave at 0 unless fitting cost forces otherwise")
        p.add_argument("--epochs", type=int, default=1,
                       help="IGNORED (one-shot method); accepted so the shared "
                            "loop and CLI stay uniform")
        self._extra_args(p)

    def _extra_args(self, p) -> None:
        pass

    # -- fitting ------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def collect_mus(frozen, forget_loader, retain_loader, device, cap: int
                    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """(x, z, n_forget, n_retain): float64 posterior means + binary labels.

        Encoded with the FROZEN encoder so the fit sees exactly the baseline
        representation (model is still identical to it at prepare() time, but
        frozen.eval() makes the BatchNorm semantics explicit).  Concatenating
        both full loaders preserves the natural class prior, which Sigma_XX
        needs -- the two loaders partition the train split by concept.
        """
        frozen.eval()

        def _mus(loader):
            out, n = [], 0
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
                mu, _ = frozen.encode(x)
                out.append(mu.detach().double().cpu())
                n += x.size(0)
                if cap and n >= cap:
                    break
            m = torch.cat(out)
            return m[:cap] if cap else m

        mu_f, mu_r = _mus(forget_loader), _mus(retain_loader)
        x = torch.cat([mu_f, mu_r])
        z = torch.cat([torch.ones(len(mu_f)), torch.zeros(len(mu_r))]).double()
        return x, z, len(mu_f), len(mu_r)

    def _fit(self, x, z, args) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return (P, b, extra_info)."""
        raise NotImplementedError

    # -- UnlearnMethod interface --------------------------------------------

    def prepare(self, model, frozen, forget_loader, retain_loader, device, args):
        from src.unlearning.erasure import fold_affine_into_fc_mu

        cap = int(getattr(args, "fit_samples", 0) or 0)
        x, z, n_f, n_r = self.collect_mus(frozen, forget_loader, retain_loader,
                                          device, cap)
        p, b, extra = self._fit(x, z, args)
        fold_affine_into_fc_mu(model, p, b)

        # trace(P) = d - rank erased for a projection; report the erased rank.
        d = p.shape[0]
        rank_erased = d - int(round(float(torch.diagonal(p).sum())))
        info = (f"fitted on {n_f} forget + {n_r} retain mus (d={d}) | "
                f"rank erased = {rank_erased} | {extra}")
        return {"info": info, "rank_erased": rank_erased}

    def step(self, model, frozen, ctx, x_r, x_f, args):
        """Measurement only -- loop.py calls this under no_grad for one-shot
        methods, and the returned tensor is never backpropagated."""
        r_out = model(x_r, kl_weight=1.0)
        f_out = model(x_f, kl_weight=1.0)
        return r_out.loss, {"retain_loss": r_out.loss.item(),
                            "forget_loss": f_out.loss.item()}


class LeaceErase(_LinearConceptErasure):
    """LEACE: perfect linear concept erasure in closed form.

    Guarantees (see erasure.py's module docstring for the theorem numbers):
    after erasure NO linear classifier with a convex loss can detect the
    concept from mu -- equivalently Cov(mu', z) = 0 and the class-conditional
    means coincide exactly -- and among all affine maps with that property
    this one changes mu least, under every p.s.d. quadratic norm at once.

    The method has no tuning knobs, and at --erase-strength 1.0 (the default)
    this is exactly the theorem's map: the price floor of linear guardedness,
    not a hyperparameter artefact.

    --erase-strength adds a DOSE rather than a knob, interpolating between the
    identity and the eraser:

        r_a(x) = (1-a) x + a (P x + b) = ((1-a) I + a P) x + a b

    a=0 leaves the model untouched, a=1 is full LEACE.  Motivation: on a narrow
    bottleneck full erasure is unaffordable -- dSprites (d=8) lands at retain
    val 467 against a 40.59 budget, 12.7x baseline -- so erasure has no
    in-budget operating point and only two observable states, untouched and
    destroyed.  Sweeping `a` turns that into a dose-response curve and answers
    what the two endpoints cannot: whether linear guardedness trades smoothly
    against reconstruction or falls off a cliff.

    NOTE the guarantee holds at a=1 ONLY.  Below it the class-conditional means
    do not coincide and the residual cross-covariance is (1-a) of the original,
    so the representation is partially scrubbed and Thm 3.4 does not apply.
    Intermediate points are a dose, never a claim of linear guardedness.
    """

    name        = "leace"
    description = "LEACE (Belrose et al. 2023): closed-form linear erasure of mu"

    def _extra_args(self, p):
        p.add_argument("--erase-strength", type=float, default=1.0,
                       help="Interpolate identity->LEACE: r_a(x) = (1-a)x + a r(x). "
                            "1.0 (default) is the theorem's map and the only value "
                            "that is linearly guarded; lower values trace the "
                            "damage-vs-guardedness curve on bottlenecks too narrow "
                            "to afford full erasure")

    def run_name(self, args):
        name = "leace_ep1"
        a = float(getattr(args, "erase_strength", 1.0))
        if a != 1.0:
            # letters-then-digits, so nb_utils.discover_sweep_keys can parse it
            name += f"_a{a:g}"
        if getattr(args, "fit_samples", 0):
            name += f"_fs{args.fit_samples}"
        return name

    def hparams(self, args):
        return {"fit_samples": getattr(args, "fit_samples", 0),
                "erase_strength": float(getattr(args, "erase_strength", 1.0))}

    def _fit(self, x, z, args):
        from src.unlearning.erasure import fit_leace
        p, b = fit_leace(x, z)
        a = float(getattr(args, "erase_strength", 1.0))
        if a != 1.0:
            eye = torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
            p = (1.0 - a) * eye + a * p
            b = a * b
        # Residual class-mean gap: exactly 0 at a=1 (the theorem), and (1-a) of
        # the pre-erasure gap below it.  Printed either way so a partial run is
        # never mistaken for a guarded one.
        y = z.bool()
        gap = float(((x[y].mean(0) - x[~y].mean(0)) @ p.T).norm())
        note = f"post-erasure class-mean gap = {gap:.2e}"
        note += " (theory: 0)" if a == 1.0 else f" (a={a:g}, NOT guarded)"
        return p, b, note


class INLPErase(_LinearConceptErasure):
    """INLP: iterative nullspace projection of the concept classifier.

    The historical baseline LEACE supersedes: removes one direction per
    iteration (--n-iter is the sweep axis), guards only against the linear
    classifiers it has already trained, and being a pure linear projection it
    zeroes directions without equalising class means.  Kept as the comparison
    the erasure literature always reports, and because its n-iter axis gives
    the erasure notebook an actual damage-vs-erasure curve to plot where
    LEACE contributes a single point.
    """

    name        = "inlp"
    description = "INLP (Ravfogel et al. 2020): iterative nullspace projection of mu"

    def _extra_args(self, p):
        p.add_argument("--n-iter", type=int, default=8,
                       help="Nullspace-projection rounds; each removes one rank "
                            "from mu-space (the sweep axis). Bounded by latent_dim")
        p.add_argument("--inlp-seed", type=int, default=0,
                       help="random_state for the per-round logistic regression. "
                            "Fit is near-deterministic; exposed for completeness")

    def run_name(self, args):
        name = f"k{args.n_iter}_ep1"
        if getattr(args, "fit_samples", 0):
            name += f"_fs{args.fit_samples}"
        if getattr(args, "inlp_seed", 0):
            name += f"_s{args.inlp_seed}"
        return name

    def hparams(self, args):
        return {"n_iter": args.n_iter,
                "fit_samples": getattr(args, "fit_samples", 0),
                "inlp_seed": getattr(args, "inlp_seed", 0)}

    def _fit(self, x, z, args):
        from src.unlearning.erasure import fit_inlp
        p, b, aurocs = fit_inlp(x, z, n_iter=int(args.n_iter),
                                seed=int(getattr(args, "inlp_seed", 0)))
        path = " -> ".join(f"{a:.3f}" for a in aurocs)
        return p, b, f"classifier AUROC per round: {path}"


# ---------------------------------------------------------------------------

METHODS: dict[str, UnlearnMethod] = {
    m.name: m for m in (NegGrad(), Finetune(), FeatureUnlearn(), RMU(),
                        FisherScrub(), LeaceErase(), INLPErase())
}
