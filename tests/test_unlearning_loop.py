"""
tests/test_unlearning_loop.py
==============================
End-to-end tests for src/unlearning/loop.py using the `tiny_spec` fixture
(in-memory data, no real dataset needed) -- checkpointing, skip-if-done,
history.json, and that run_unlearning actually wires methods.py into the
shared loop correctly for every registered method.
"""

import json

import torch

from src.unlearning.loop import run_unlearning, load_baseline
from src.unlearning.methods import METHODS


def make_args(method_name, **overrides):
    from types import SimpleNamespace
    base = dict(lr=1e-3, grad_clip=1.0, save_every=0, num_workers=0, out_dir=None,
               epochs=2, max_batches=0, val_tolerance=0.10)
    if method_name in ("neg_grad", "finetune"):
        base["lambda_forget"] = 1.0
    elif method_name == "feature_unlearn":
        base.update(alpha=1.0, elbo_anchor=0.0, train_encoder=False,
                    recon_norm="l2", ms_ssim=0.0, project_to="zero")
    elif method_name == "rmu":
        base.update(c=1.0, alpha=50.0, u_dist="gaussian")
    elif method_name == "fisher":
        # fisher_samples kept tiny: the FIM pass runs at batch size 1, so the
        # default 2000 would dominate the whole test suite's runtime.
        base.update(noise_scale=1e-3, fisher_samples=8, fisher_eps=1e-8,
                    fisher_seed=0, epochs=1)
    elif method_name == "leace":
        base.update(fit_samples=0, epochs=1)
    elif method_name == "inlp":
        # n_iter must stay below the tiny VAE's latent_dim (4), or the
        # projection legitimately zeroes the whole latent space.
        base.update(fit_samples=0, n_iter=2, inlp_seed=0, epochs=1)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_load_baseline_returns_independent_frozen_copy(tiny_spec):
    model, frozen, ckpt = load_baseline(tiny_spec, torch.device("cpu"))
    assert ckpt["epoch"] == 1
    # Editing `model` must not affect `frozen` -- they must not share storage.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for pm, pf in zip(model.parameters(), frozen.parameters()):
        assert not torch.equal(pm, pf)
    assert all(not p.requires_grad for p in frozen.parameters())


def test_every_registered_method_runs_end_to_end(tiny_spec):
    """Smoke test across the full method registry: each one should train for
    a couple of epochs and produce a valid output directory."""
    for name, method in METHODS.items():
        args = make_args(name)
        out_dir = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
        assert out_dir is not None, f"{name} run returned None unexpectedly"
        assert (out_dir / "final.pt").exists(), f"{name} did not save final.pt"
        assert (out_dir / "history.json").exists(), f"{name} did not save history.json"


def test_skip_if_done_avoids_recomputation(tiny_spec, monkeypatch):
    """Calling run_unlearning twice with the same config should train once,
    then return None (skip) on the second call without re-entering the loop."""
    from src.unlearning.methods import NegGrad
    method = NegGrad()
    args = make_args("neg_grad")

    out_dir_1 = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    assert out_dir_1 is not None
    mtime_1 = (out_dir_1 / "final.pt").stat().st_mtime

    out_dir_2 = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    assert out_dir_2 is None, "second call should skip, not retrain"
    mtime_2 = (out_dir_1 / "final.pt").stat().st_mtime
    assert mtime_1 == mtime_2, "final.pt should not have been rewritten"


def test_history_json_has_one_entry_per_epoch(tiny_spec):
    from src.unlearning.methods import Finetune
    method = Finetune()
    args = make_args("finetune", epochs=3)

    out_dir = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    history = json.loads((out_dir / "history.json").read_text())
    for key in ("retain_loss", "forget_loss", "total_loss", "val_loss"):
        assert len(history[key]) == 3, f"{key} should have 3 entries, got {len(history[key])}"


def test_best_checkpoint_has_lowest_val_loss(tiny_spec):
    """best.pt is only written when save_every triggers it or a new best is
    found -- verify the metrics inside best.pt actually are the minimum val
    loss seen during training."""
    from src.unlearning.methods import NegGrad
    method = NegGrad()
    args = make_args("neg_grad", epochs=4, lambda_forget=0.0)  # lambda=0 -> stable descent

    out_dir = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    history = json.loads((out_dir / "history.json").read_text())
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)

    assert best_ckpt["metrics"]["loss"] == min(history["val_loss"])


def test_run_name_encodes_hyperparameters_in_output_path(tiny_spec):
    """Output directory naming must match each method's run_name() -- this is
    what notebooks/nb_utils.discover_sweep_keys parses."""
    from src.unlearning.methods import FeatureUnlearn
    method = FeatureUnlearn()
    args = make_args("feature_unlearn", alpha=3.5, epochs=1)

    out_dir = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    assert out_dir.name == "alpha3.5_ep1"


def test_out_dir_override_is_respected(tiny_spec, tmp_path):
    from src.unlearning.methods import NegGrad
    method = NegGrad()
    custom = tmp_path / "custom_location"
    args = make_args("neg_grad", epochs=1, out_dir=str(custom))

    out_dir = run_unlearning(tiny_spec, method, args, torch.device("cpu"))
    assert out_dir == custom
    assert (custom / "final.pt").exists()


# ---------------------------------------------------------------------------
# Paper-fidelity additions (see docs/ARCHITECTURE.md "Fidelity to the source papers")
# ---------------------------------------------------------------------------

class TestBatchBudget:
    """--max-batches expresses the training budget in the unit RMU's reference
    implementation actually uses (80-150 batches TOTAL, not epochs)."""

    def test_max_batches_caps_optimiser_steps(self, tiny_spec, monkeypatch):
        from src.unlearning.methods import RMU
        method = RMU()
        seen = []
        orig = method.step
        monkeypatch.setattr(method, "step",
                            lambda *a, **k: (seen.append(1), orig(*a, **k))[1])
        run_unlearning(tiny_spec, method,
                       make_args("rmu", epochs=2, max_batches=3),
                       torch.device("cpu"))
        assert len(seen) == 6, f"expected 2 epochs x 3 batches, got {len(seen)}"

    def test_max_batches_gets_its_own_run_directory(self, tiny_spec):
        """Otherwise a budgeted run would collide with the full-pass run's
        skip-if-done marker and silently be skipped."""
        from src.unlearning.methods import RMU
        full = run_unlearning(tiny_spec, RMU(), make_args("rmu"),
                              torch.device("cpu"))
        cap  = run_unlearning(tiny_spec, RMU(), make_args("rmu", max_batches=2),
                              torch.device("cpu"))
        assert cap is not None and cap != full
        assert cap.name.endswith("_b2")

    def test_zero_max_batches_is_a_full_pass(self, tiny_spec):
        from src.unlearning.methods import RMU
        out = run_unlearning(tiny_spec, RMU(), make_args("rmu", max_batches=0),
                             torch.device("cpu"))
        assert not out.name.endswith("_b0"), "default must not rename the directory"


class TestTradeoffSelection:
    """best.pt selects on retain val alone, which is monotonically optimised by
    not unlearning -- so it returns the least-unlearned epoch. best_tradeoff.pt
    selects the most forgetting inside a retain-val budget instead."""

    def test_history_records_the_selection_rule(self, tiny_spec):
        from src.unlearning.methods import NegGrad
        out = run_unlearning(tiny_spec, NegGrad(), make_args("neg_grad", epochs=3),
                             torch.device("cpu"))
        sel = json.loads((out / "history.json").read_text())["selection"]
        assert sel["forget_better"] == "higher"
        assert sel["val_budget"] == sel["baseline_val"] * (1 + sel["val_tolerance"])

    def test_tradeoff_picks_most_forgetting_within_budget(self, tiny_spec):
        from src.unlearning.methods import NegGrad
        out = run_unlearning(tiny_spec, NegGrad(),
                             make_args("neg_grad", epochs=3, val_tolerance=1e6),
                             torch.device("cpu"))
        hist = json.loads((out / "history.json").read_text())
        ep = hist["selection"]["tradeoff_epoch"]
        assert ep is not None, "a huge tolerance must admit at least one epoch"
        # forget_better == "higher", so it must be the argmax of forget_loss.
        assert ep == hist["forget_loss"].index(max(hist["forget_loss"])) + 1
        assert (out / "best_tradeoff.pt").exists()

    def test_no_tradeoff_checkpoint_when_budget_unmeetable(self, tiny_spec):
        """An impossible budget must leave best_tradeoff.pt absent rather than
        silently falling back to best.pt -- the absence is the finding."""
        from src.unlearning.methods import NegGrad
        out = run_unlearning(tiny_spec, NegGrad(),
                             make_args("neg_grad", epochs=2, val_tolerance=-1.0),
                             torch.device("cpu"))
        assert not (out / "best_tradeoff.pt").exists()
        assert json.loads((out / "history.json").read_text())["selection"]["tradeoff_epoch"] is None

    def test_finetune_with_no_forget_term_is_still_selectable(self, tiny_spec):
        """lambda=0 keeps the forget set out of the objective but still MEASURES
        the forget ELBO, so pure finetuning is rankable under the same budget
        rule as every other method and does get a best_tradeoff.pt.

        This inverts an earlier assertion. The old behaviour recorded NaN and
        wrote no tradeoff checkpoint, which meant pure finetuning silently
        dropped out of any cross-method comparison built on the shared
        selection rule -- a bookkeeping artefact being mistaken for "this
        method has no trade-off to make"."""
        import math
        from src.unlearning.methods import Finetune
        out = run_unlearning(tiny_spec, Finetune(),
                             make_args("finetune", lambda_forget=0.0,
                                       val_tolerance=1e6),
                             torch.device("cpu"))
        assert (out / "best_tradeoff.pt").exists()
        hist = json.loads((out / "history.json").read_text())
        assert hist["selection"]["tradeoff_epoch"] is not None
        assert all(math.isfinite(f) for f in hist["forget_loss"])


class TestFeatureUnlearnVariants:
    """The L1 / MS-SSIM / threshold flags from Moon et al. must run end to end
    and must not collide with the default run's directory."""

    def test_paper_faithful_variant_runs_and_is_kept_separate(self, tiny_spec):
        from src.unlearning.methods import FeatureUnlearn
        base = run_unlearning(tiny_spec, FeatureUnlearn(),
                              make_args("feature_unlearn"), torch.device("cpu"))
        var  = run_unlearning(tiny_spec, FeatureUnlearn(),
                              make_args("feature_unlearn", recon_norm="l1",
                                        ms_ssim=0.5, project_to="threshold"),
                              torch.device("cpu"))
        assert var is not None and var != base
        assert var.name.endswith("_n1_ms0.5_pt1")
        hist = json.loads((var / "history.json").read_text())
        assert all(v == v for v in hist["forget_loss"]), "NaN in forget loss"

    def test_rmu_uniform_steering_vector_runs_and_is_kept_separate(self, tiny_spec):
        from src.unlearning.methods import RMU
        base = run_unlearning(tiny_spec, RMU(), make_args("rmu"),
                              torch.device("cpu"))
        var  = run_unlearning(tiny_spec, RMU(), make_args("rmu", u_dist="uniform"),
                              torch.device("cpu"))
        assert var is not None and var != base and var.name.endswith("_u1")


class TestRMUSteeringSeed:
    """Every RMU run before --u-seed existed used a hardcoded seed=0 steering
    vector. Reproducibility across u requires being able to vary it."""

    def test_default_seed_reproduces_original_run_name(self, tiny_spec):
        from src.unlearning.methods import RMU
        out = run_unlearning(tiny_spec, RMU(), make_args("rmu"), torch.device("cpu"))
        assert not out.name.endswith(tuple(f"_s{i}" for i in range(1, 10))), \
            "default u_seed=0 must not rename the directory"

    def test_different_seeds_produce_different_steering_vectors(self, tiny_spec):
        from src.unlearning.methods import RMU
        method = RMU()
        model, frozen, _ = load_baseline(tiny_spec, torch.device("cpu"))
        fl, rl = tiny_spec.concept_loaders(0)
        ctx0 = method.prepare(model, frozen, fl, rl, torch.device("cpu"), make_args("rmu", u_seed=0))
        ctx1 = method.prepare(model, frozen, fl, rl, torch.device("cpu"), make_args("rmu", u_seed=1))
        assert not torch.allclose(ctx0["u"], ctx1["u"])

    def test_same_seed_is_deterministic(self, tiny_spec):
        from src.unlearning.methods import RMU
        method = RMU()
        model, frozen, _ = load_baseline(tiny_spec, torch.device("cpu"))
        fl, rl = tiny_spec.concept_loaders(0)
        ctx_a = method.prepare(model, frozen, fl, rl, torch.device("cpu"), make_args("rmu", u_seed=7))
        ctx_b = method.prepare(model, frozen, fl, rl, torch.device("cpu"), make_args("rmu", u_seed=7))
        assert torch.equal(ctx_a["u"], ctx_b["u"])

    def test_nonzero_seed_gets_its_own_run_directory(self, tiny_spec):
        from src.unlearning.methods import RMU
        default = run_unlearning(tiny_spec, RMU(), make_args("rmu"), torch.device("cpu"))
        seeded  = run_unlearning(tiny_spec, RMU(), make_args("rmu", u_seed=3), torch.device("cpu"))
        assert seeded is not None and seeded != default and seeded.name.endswith("_s3")


ENCODER_KEYS = ("encoder_conv", "fc_mu", "fc_std")


def _encoder_delta(run_dir, tiny_spec):
    """Largest absolute change in any encoder parameter OR buffer, run vs baseline."""
    base = torch.load(tiny_spec.baseline_ckpt, map_location="cpu",
                      weights_only=True)["model_state_dict"]
    got  = torch.load(run_dir / "final.pt", map_location="cpu",
                      weights_only=True)["model_state_dict"]
    return max((base[k].float() - got[k].float()).abs().max().item()
               for k in base if k.startswith(ENCODER_KEYS))


class TestFeatureUnlearnEncoderFlags:
    """--train-encoder and --elbo-anchor used to appear in NEITHER run_name nor
    hparams.  A --train-encoder run therefore resolved to the same directory as
    its frozen-encoder twin, hit loop.py's final.pt skip marker, and silently
    handed back the wrong model -- with nothing in metrics.json to reveal it."""

    def test_train_encoder_does_not_collide_with_the_frozen_run(self, tiny_spec):
        from src.unlearning.methods import FeatureUnlearn
        frozen = run_unlearning(tiny_spec, FeatureUnlearn(),
                                make_args("feature_unlearn"), torch.device("cpu"))
        trained = run_unlearning(tiny_spec, FeatureUnlearn(),
                                 make_args("feature_unlearn", train_encoder=True),
                                 torch.device("cpu"))
        assert trained is not None, "collided with the frozen run and was skipped"
        assert trained != frozen and trained.name.endswith("_te1")

    def test_elbo_anchor_does_not_collide_with_the_unanchored_run(self, tiny_spec):
        from src.unlearning.methods import FeatureUnlearn
        plain = run_unlearning(tiny_spec, FeatureUnlearn(),
                               make_args("feature_unlearn"), torch.device("cpu"))
        anchored = run_unlearning(tiny_spec, FeatureUnlearn(),
                                  make_args("feature_unlearn", elbo_anchor=1.0),
                                  torch.device("cpu"))
        assert anchored is not None and anchored != plain
        assert anchored.name.endswith("_ea1.0")

    def test_both_flags_are_recorded_in_hparams(self, tiny_spec):
        """Directory naming alone is not enough -- metrics.json must say which
        variant produced the weights."""
        from src.unlearning.methods import FeatureUnlearn
        hp = FeatureUnlearn().hparams(
            make_args("feature_unlearn", train_encoder=True, elbo_anchor=0.5))
        assert hp["train_encoder"] is True
        assert hp["elbo_anchor"] == 0.5

    def test_frozen_encoder_is_bit_identical_but_trained_one_is_not(self, tiny_spec):
        """The claim the thesis rests on: with the encoder frozen, mu(x) cannot
        move at all, so every mu-based metric is an identity rather than a
        measurement.  Checked over buffers too -- BatchNorm running stats would
        drift if train_mode were ever flipped back to True."""
        from src.unlearning.methods import FeatureUnlearn
        frozen = run_unlearning(tiny_spec, FeatureUnlearn(),
                                make_args("feature_unlearn"), torch.device("cpu"))
        trained = run_unlearning(tiny_spec, FeatureUnlearn(),
                                 make_args("feature_unlearn", train_encoder=True,
                                           alpha=20.0),
                                 torch.device("cpu"))
        assert _encoder_delta(frozen, tiny_spec) == 0.0
        assert _encoder_delta(trained, tiny_spec) > 0.0
