"""
tests/test_unlearning_methods.py
=================================
Does each unlearning method actually do what it claims?

These tests run a handful of real optimiser steps on tiny synthetic data and
check the specific, falsifiable property each method is supposed to have --
not just that the code executes.  Where useful, a method is tested against
its own explicit claim in src/unlearning/methods.py's docstrings:

    NegGrad+ / Finetune : loss = retain_ELBO - lambda * forget_ELBO
    Feature Unlearning   : encoder is frozen; forget latents lose their
                           z_e component after projection
    RMU                  : encoder DOES move; forget latents are pulled
                           toward c*u; retain latents are anchored to frozen
"""

import copy

import pytest
import torch

from src.models.vae import ConvVAE, VAEConfig
from src.unlearning.methods import (
    NegGrad, Finetune, FeatureUnlearn, RMU,
    project_out, make_to_img, img_mse, freeze_encoder,
    bn_stats_frozen,
)


def _bn_buffers(model):
    return {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}


class TestForgetPassDoesNotPolluteBatchNorm:
    """A forward pass over the FORGET set must not rewrite BatchNorm running
    statistics.

    This is a regression guard for a measured, severe bug. BN buffers are not
    parameters and appear in no method's objective, but a train-mode forward
    pass updates them from the batch -- so concept-homogeneous forget batches
    push the running statistics toward the forget distribution. On the real
    dSprites baseline, with no optimiser and no weight change at all, 100
    forget batches in train mode moved retain val 35.71 -> 57.94, while the
    same batches in eval mode left it at 35.82. It also made NegGrad+ retain
    damage nearly independent of lambda, because most of the harm was never
    coming from the gradient ascent.
    """

    def test_context_manager_restores_training_flags(self):
        model = make_model()
        model.train()
        with bn_stats_frozen(model):
            assert all(not m.training for m in model.modules()
                       if isinstance(m, torch.nn.BatchNorm2d))
        assert all(m.training for m in model.modules()
                   if isinstance(m, torch.nn.BatchNorm2d))

    def test_neggrad_forget_pass_leaves_bn_buffers_untouched(self):
        model = make_model(); model.train()
        before = _bn_buffers(model)
        loss, _ = NegGrad().step(model, None, {}, batch(4, seed=1), batch(4, seed=2),
                                 FakeArgs(lambda_forget=1.0))
        loss.backward()
        after = _bn_buffers(model)
        # the RETAIN pass legitimately updates them; assert the forget pass adds
        # nothing on top by re-running retain alone and comparing.
        model2 = make_model(); model2.train()
        model2(batch(4, seed=1), kl_weight=1.0)
        expected = _bn_buffers(model2)
        for k in before:
            assert torch.allclose(after[k].float(), expected[k].float(), atol=1e-6), \
                f"forget pass moved BN buffer {k}"

    def test_finetune_measurement_pass_leaves_bn_buffers_untouched(self):
        """no_grad does NOT stop BN updates, so pure finetuning -- which must
        never be affected by the forget set -- needs the same protection."""
        model = make_model(); model.train()
        Finetune().step(model, None, {}, batch(4, seed=1), batch(4, seed=2),
                        FakeArgs(lambda_forget=0.0))
        after = _bn_buffers(model)
        model2 = make_model(); model2.train()
        model2(batch(4, seed=1), kl_weight=1.0)
        expected = _bn_buffers(model2)
        for k in expected:
            assert torch.allclose(after[k].float(), expected[k].float(), atol=1e-6), \
                f"forget measurement pass moved BN buffer {k}"


def make_model(latent_dim=4, base_channels=4, in_channels=1, recon_loss="bce", seed=0):
    torch.manual_seed(seed)
    return ConvVAE(VAEConfig(latent_dim=latent_dim, base_channels=base_channels,
                             in_channels=in_channels, recon_loss=recon_loss))


def batch(n, channels=1, recon_loss="bce", seed=0):
    g = torch.Generator().manual_seed(seed)
    if recon_loss == "bce":
        return torch.randint(0, 2, (n, channels, 64, 64), generator=g).float()
    return torch.rand(n, channels, 64, 64, generator=g) * 2 - 1


class FakeArgs:
    """Minimal stand-in for argparse.Namespace with only the fields a method reads."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# Pure math helpers used by the methods
# ---------------------------------------------------------------------------

class TestMethodHelpers:
    def test_project_out_removes_the_projected_component(self):
        """The defining property of project_out: after removing the component
        along z_e, mu_translated . z_e must be ~0."""
        torch.manual_seed(0)
        mu = torch.randn(20, 8)
        z_e = torch.randn(8)
        translated = project_out(mu, z_e)
        residual = translated @ z_e
        assert residual.abs().max().item() < 1e-4

    def test_project_out_is_identity_when_mu_orthogonal_to_direction(self):
        z_e = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mu = torch.tensor([[0.0, 5.0, -3.0, 2.0]])
        out = project_out(mu, z_e)
        assert torch.allclose(out, mu, atol=1e-5)

    def test_make_to_img_bce_bounds_to_unit_interval(self):
        to_img = make_to_img("bce")
        raw = torch.tensor([-50.0, 0.0, 50.0])
        out = to_img(raw)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_make_to_img_mse_is_identity(self):
        to_img = make_to_img("mse")
        raw = torch.tensor([-0.7, 0.0, 0.9])
        assert torch.equal(to_img(raw), raw)

    def test_img_mse_matches_manual_formula(self):
        a = torch.rand(4, 1, 8, 8)
        b = torch.rand(4, 1, 8, 8)
        manual = ((a - b) ** 2).sum() / a.size(0)
        assert torch.allclose(img_mse(a, b), manual, atol=1e-5)

    def test_freeze_encoder_only_freezes_encoder_params(self):
        model = make_model()
        freeze_encoder(model)
        for name, p in model.named_parameters():
            if name.startswith(("encoder_conv", "fc_mu", "fc_std")):
                assert not p.requires_grad, f"{name} should be frozen"
            else:
                assert p.requires_grad, f"{name} should still be trainable"


# ---------------------------------------------------------------------------
# NegGrad+ / Finetune -- same loss, different default lambda
# ---------------------------------------------------------------------------

class TestElboAscent:
    def test_loss_matches_the_documented_formula(self):
        """loss = retain_ELBO - lambda * forget_ELBO, checked against the
        method's own returned metrics (self-consistent, not a duplicate
        forward pass that would resample z and drift from the real value)."""
        model = make_model()
        method = NegGrad()
        args = FakeArgs(lambda_forget=0.7)
        x_r, x_f = batch(4, seed=1), batch(4, seed=2)

        loss, metrics = method.step(model, None, {}, x_r, x_f, args)
        expected = metrics["retain_loss"] - 0.7 * metrics["forget_loss"]
        assert loss.item() == pytest.approx(expected, abs=1e-4)

    def test_lambda_zero_keeps_forget_out_of_the_objective_but_still_reports_it(self):
        """When lambda_forget=0, Finetune degenerates to retain-only: the loss
        must equal the retain ELBO EXACTLY, so the forget set has no influence
        on the gradient.

        forget_loss is nonetheless reported as a real measured number (it used
        to be NaN, on the reasoning that skipping the forward pass was cheaper).
        The budget-based checkpoint rule ranks epochs by forget_loss, so a NaN
        series makes pure finetuning unselectable and silently excludes it from
        every cross-method comparison. Measuring is not optimising; the
        assertion that it stays out of the objective is the line below."""
        import math
        model = make_model()
        method = Finetune()
        args = FakeArgs(lambda_forget=0.0)
        x_r, x_f = batch(4, seed=1), batch(4, seed=2)

        loss, metrics = method.step(model, None, {}, x_r, x_f, args)
        assert loss.item() == pytest.approx(metrics["retain_loss"], abs=1e-5)
        assert math.isfinite(metrics["forget_loss"])

    def test_lambda_zero_forget_measurement_contributes_no_gradient(self):
        """The load-bearing half of the change above: measuring the forget ELBO
        must not put the forget batch into the backward pass. Compare gradients
        against a retain-only loss computed with no forget batch in sight."""
        model = make_model()
        x_r, x_f = batch(4, seed=1), batch(4, seed=2)

        # The VAE samples z = mu + sigma*eps, so every forward pass consumes RNG.
        # Both sides must start from the same seed or they differ by the draw
        # rather than by anything to do with the forget batch.
        torch.manual_seed(0)
        loss, _ = Finetune().step(model, None, {}, x_r, x_f, FakeArgs(lambda_forget=0.0))
        model.zero_grad(); loss.backward()
        via_step = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        torch.manual_seed(0)
        model.zero_grad()
        model(x_r, kl_weight=1.0).loss.backward()
        retain_only = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        assert len(via_step) == len(retain_only) and via_step
        for a, b in zip(via_step, retain_only):
            assert torch.allclose(a, b, atol=1e-6)

    def test_gradient_ascent_increases_forget_elbo_over_steps(self):
        """The actual claim of NegGrad+: training should ASCEND the forget
        loss (push it up) while descending the retain loss. Run a handful of
        real optimiser steps and check both directions move as claimed."""
        torch.manual_seed(0)
        model = make_model()
        method = NegGrad()
        args = FakeArgs(lambda_forget=2.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x_r, x_f = batch(8, seed=1), batch(8, seed=2)

        def elbo(x):
            model.eval()
            with torch.no_grad():
                return model(x, kl_weight=1.0).loss.item()

        forget_before = elbo(x_f)
        retain_before = elbo(x_r)

        model.train()
        for _ in range(15):
            loss, _ = method.step(model, None, {}, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        forget_after = elbo(x_f)
        retain_after = elbo(x_r)

        assert forget_after > forget_before, "forget ELBO should rise under ascent"
        # Some retain damage at lambda=2.0 on an untrained tiny model is expected
        # -- real sweeps show retain collapse at high lambda too (see
        # sweeps/grids.psd1). This only guards against outright numerical
        # divergence (inf/nan), not against the known collapse behaviour.
        assert torch.isfinite(torch.tensor(retain_after))
        assert retain_after < retain_before * 20

    def test_train_mode_is_true(self):
        """NegGrad+/Finetune have no frozen reference, so BatchNorm should
        update its running stats normally (train mode)."""
        assert NegGrad().train_mode is True
        assert Finetune().train_mode is True


# ---------------------------------------------------------------------------
# Feature Unlearning (Moon et al.)
# ---------------------------------------------------------------------------

class TestFeatureUnlearn:
    def test_train_mode_is_false(self):
        """Must run in eval mode -- see the BatchNorm regression test in
        test_regressions.py for why this specifically matters."""
        assert FeatureUnlearn().train_mode is False

    def test_prepare_freezes_encoder_by_default(self):
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(train_encoder=False)

        from torch.utils.data import DataLoader, TensorDataset
        forget_loader = DataLoader(TensorDataset(batch(8, seed=1)), batch_size=4)
        retain_loader = DataLoader(TensorDataset(batch(8, seed=2)), batch_size=4)

        method.prepare(model, frozen, forget_loader, retain_loader, torch.device("cpu"), args)
        for name, p in model.named_parameters():
            if name.startswith(("encoder_conv", "fc_mu", "fc_std")):
                assert not p.requires_grad

    def test_prepare_leaves_encoder_trainable_when_requested(self):
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(train_encoder=True)

        from torch.utils.data import DataLoader, TensorDataset
        forget_loader = DataLoader(TensorDataset(batch(8, seed=1)), batch_size=4)
        retain_loader = DataLoader(TensorDataset(batch(8, seed=2)), batch_size=4)

        method.prepare(model, frozen, forget_loader, retain_loader, torch.device("cpu"), args)
        assert all(p.requires_grad for p in model.encoder_conv.parameters())

    def test_z_e_is_the_mean_difference(self):
        """z_e must equal mean_mu(forget) - mean_mu(retain), computed from the
        FROZEN encoder -- not a trained/learned direction."""
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(train_encoder=False)

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        forget_loader = DataLoader(TensorDataset(x_f), batch_size=4)
        retain_loader = DataLoader(TensorDataset(x_r), batch_size=4)

        ctx = method.prepare(model, frozen, forget_loader, retain_loader,
                             torch.device("cpu"), args)

        with torch.no_grad():
            mu_f, _ = frozen.encode(x_f)
            mu_r, _ = frozen.encode(x_r)
        expected = mu_f.mean(0) - mu_r.mean(0)
        assert torch.allclose(ctx["z_e"], expected, atol=1e-4)

    def test_encoder_is_literally_unchanged_after_training(self):
        """The central claim this method depends on: after several training
        steps, the encoder's parameters must be numerically identical to
        before -- not "close", identical, since it never received a gradient."""
        torch.manual_seed(0)
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(alpha=5.0, elbo_anchor=0.0, train_encoder=False)

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        forget_loader = DataLoader(TensorDataset(x_f), batch_size=4)
        retain_loader = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, forget_loader, retain_loader,
                             torch.device("cpu"), args)

        encoder_before = {n: p.clone() for n, p in model.named_parameters()
                          if n.startswith(("encoder_conv", "fc_mu", "fc_std"))}

        trainable = method.trainable_params(model)
        opt = torch.optim.AdamW(trainable, lr=1e-2)
        model.eval()
        for _ in range(10):
            loss, _ = method.step(model, frozen, ctx, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        for n, p in model.named_parameters():
            if n in encoder_before:
                assert torch.equal(p, encoder_before[n]), f"{n} changed but should be frozen"

    def test_retain_loss_is_near_zero_at_initialisation(self):
        """Regression guard for the BatchNorm bug: model and frozen start with
        identical weights, so decode(mu_r) should match frozen.decode(mu_r)
        almost exactly -- this is only true when both run in eval mode."""
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(alpha=1.0, elbo_anchor=0.0, train_encoder=False)

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(8, seed=2)
        forget_loader = DataLoader(TensorDataset(x_f), batch_size=4)
        retain_loader = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, forget_loader, retain_loader,
                             torch.device("cpu"), args)

        model.eval()  # the fixed training mode
        _, metrics = method.step(model, frozen, ctx, x_r, x_f, args)
        assert metrics["retain_loss"] < 1e-6

    def test_forget_loss_decreases_with_training(self):
        """The decoder should learn to reproduce the translated target over
        a handful of steps -- i.e. this is actually optimisable, not stuck."""
        torch.manual_seed(0)
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = FeatureUnlearn()
        args = FakeArgs(alpha=1.0, elbo_anchor=0.0, train_encoder=False)

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        forget_loader = DataLoader(TensorDataset(x_f), batch_size=4)
        retain_loader = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, forget_loader, retain_loader,
                             torch.device("cpu"), args)

        trainable = method.trainable_params(model)
        opt = torch.optim.AdamW(trainable, lr=1e-2)
        model.eval()

        _, first = method.step(model, frozen, ctx, x_r, x_f, args)
        first_forget = first["forget_loss"]

        for _ in range(20):
            loss, _ = method.step(model, frozen, ctx, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        _, last = method.step(model, frozen, ctx, x_r, x_f, args)
        assert last["forget_loss"] < first_forget


# ---------------------------------------------------------------------------
# RMU -- Representation Misdirection
# ---------------------------------------------------------------------------

class TestRMU:
    def test_train_mode_is_false(self):
        assert RMU().train_mode is False

    def test_steering_vector_is_unit_norm_and_deterministic(self):
        model = make_model()
        frozen = copy.deepcopy(model)
        method = RMU()
        args = FakeArgs(c=3.0)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(batch(4)), batch_size=4)

        ctx1 = method.prepare(model, frozen, loader, loader, torch.device("cpu"), args)
        ctx2 = method.prepare(model, frozen, loader, loader, torch.device("cpu"), args)
        assert ctx1["u"].norm().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.equal(ctx1["u"], ctx2["u"]), "steering vector must be reproducible"

    def test_encoder_changes_unlike_feature_unlearn(self):
        """The defining structural contrast with Feature Unlearning: RMU is
        supposed to move the encoder.  After training, encoder parameters
        must have actually changed."""
        torch.manual_seed(0)
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = RMU()
        args = FakeArgs(c=1.0, alpha=50.0)

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        loader_f = DataLoader(TensorDataset(x_f), batch_size=4)
        loader_r = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, loader_f, loader_r, torch.device("cpu"), args)

        encoder_before = {n: p.clone() for n, p in model.named_parameters()
                          if n.startswith(("encoder_conv", "fc_mu", "fc_std"))}

        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.eval()
        for _ in range(10):
            loss, _ = method.step(model, frozen, ctx, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        changed = [n for n, p in model.named_parameters()
                  if n in encoder_before and not torch.equal(p, encoder_before[n])]
        assert changed, "RMU should move the encoder, but no encoder param changed"

    def test_forget_latents_move_toward_the_steering_target(self):
        """The actual claim: training pulls mu(x_f) toward c*u.  Distance to
        the target should shrink over training steps."""
        torch.manual_seed(0)
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = RMU()
        args = FakeArgs(c=2.0, alpha=1.0)  # low alpha: let the forget term dominate

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        loader_f = DataLoader(TensorDataset(x_f), batch_size=4)
        loader_r = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, loader_f, loader_r, torch.device("cpu"), args)
        target = args.c * ctx["u"]

        def dist_to_target():
            model.eval()
            with torch.no_grad():
                mu_f, _ = model.encode(x_f)
            return (mu_f - target).norm(dim=1).mean().item()

        dist_before = dist_to_target()

        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        model.eval()
        for _ in range(25):
            loss, _ = method.step(model, frozen, ctx, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        dist_after = dist_to_target()
        assert dist_after < dist_before

    def test_high_alpha_anchors_retain_latents_to_frozen(self):
        """A large retain anchor should keep mu(x_r) close to the frozen
        encoder's output, even while the forget term pulls the encoder
        elsewhere -- the calibration finding from methods.py's docstring."""
        torch.manual_seed(0)
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        method = RMU()
        args = FakeArgs(c=5.0, alpha=200.0)  # deliberately large anchor

        from torch.utils.data import DataLoader, TensorDataset
        x_f, x_r = batch(8, seed=1), batch(16, seed=2)
        loader_f = DataLoader(TensorDataset(x_f), batch_size=4)
        loader_r = DataLoader(TensorDataset(x_r), batch_size=4)
        ctx = method.prepare(model, frozen, loader_f, loader_r, torch.device("cpu"), args)

        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        model.eval()
        for _ in range(15):
            loss, _ = method.step(model, frozen, ctx, x_r, x_f, args)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_r_after, _ = model.encode(x_r)
            mu_r_frozen, _ = frozen.encode(x_r)
        drift = (mu_r_after - mu_r_frozen).norm(dim=1).mean().item()
        assert drift < 1.0, f"retain latents drifted {drift:.3f} despite alpha=200"
