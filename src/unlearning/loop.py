"""
src/unlearning/loop.py
======================
The single unlearning training loop, shared by every method and dataset.

Replaces the loop that used to be copy-pasted into each of the eight
``experiments/unlearn/{dataset}/{method}.py`` scripts.

Output layout is deliberately UNCHANGED from those scripts, so existing results,
notebooks and ``nb_utils.discover_sweep_keys`` keep working:

    results/<dataset>/<dataset>_<method>/<run_name>/
        best.pt            lowest retain val loss
        best_tradeoff.pt   most forgetting within the retain-val budget
        epoch_0001.pt ...  every --save-every epochs
        final.pt           last epoch; also the skip-if-done marker
        history.json       per-epoch retain / forget / total / val

Why two "best" checkpoints
--------------------------
best.pt selects on retain val loss alone.  That criterion is monotonically
optimised by not unlearning at all, so it systematically returns the LEAST
unlearned epoch of a run -- for RMU, always epoch 1.  It is kept because
existing results and notebooks refer to it.

best_tradeoff.pt is the checkpoint that forgets most among the epochs whose
retain val loss stays within --val-tolerance of the baseline, which is the
selection rule the unlearning literature actually uses (Kurmanji et al. tune
"to get a high forget-error while not destroying retain-error").  Each method
declares via UnlearnMethod.forget_better which direction of its reported
forget_loss counts as more forgetting.  Runs where no epoch meets the budget
get no best_tradeoff.pt, which is itself the finding.
"""

from __future__ import annotations

import json
import logging
import math
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn

from src.models.vae import ConvVAE, VAEConfig
from src.models.flow import RealNVP, FlowConfig
from src.unlearning.methods import UnlearnMethod
from src.unlearning.spec import DatasetSpec, SEED
from src.utils.seed import seed_all
from src.unlearning.provenance import write_run_meta

torch.serialization.add_safe_globals([VAEConfig])

log = logging.getLogger(__name__)


def _batch_x(batch, device):
    return (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)


def save_checkpoint(model, optimizer, epoch, metrics, path: Path) -> None:
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        # None for one-shot methods (Fisher scrubbing), which never build an
        # optimiser -- there is no iterative state to resume from.
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics":              metrics,
        "model_config":         model.cfg,
    }, path)


@torch.no_grad()
def run_val(model, val_loader, device) -> dict[str, float]:
    """Mean ELBO over the retain validation split (beta = 1)."""
    model.eval()
    totals = {"loss": 0.0, "loss_recon": 0.0, "loss_kl": 0.0}
    n = 0
    for batch in val_loader:
        out = model(_batch_x(batch, device), kl_weight=1.0)
        totals["loss"]       += out.loss.item()
        totals["loss_recon"] += out.loss_recon.item()
        totals["loss_kl"]    += out.loss_kl.item()
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def load_baseline(spec: DatasetSpec, device, override: str | None = None):
    """Return (model, frozen_reference, checkpoint_dict).

    `override` re-points the starting checkpoint away from spec.baseline_ckpt.
    It exists for the post-hoc erasure methods (LEACE, INLP): those are
    closed-form transforms of a trained encoder rather than training runs, so
    they have no --train-seed axis and were the only rows in the master table
    stuck at n=1 -- awkward, since they sit at the extreme of the headline
    attack claim. Refitting them on each replicate baseline from queue28 is the
    only way they can carry an error bar at all.

    NOTE the retain budget still comes from whichever checkpoint is loaded
    here, so a replicate baseline brings its own budget with it. That is
    correct: the budget is defined relative to the model being unlearned.
    """
    ckpt_path = Path(override) if override else spec.baseline_ckpt
    log.info("Loading baseline: %s", ckpt_path)
    torch.serialization.add_safe_globals([VAEConfig, FlowConfig])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    _cfg = ckpt["model_config"]
    _Model = RealNVP if type(_cfg).__name__ == "FlowConfig" else ConvVAE
    model = _Model(_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    frozen = _Model(_cfg).to(device)
    frozen.load_state_dict(ckpt["model_state_dict"])
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()

    log.info("  epoch %d | val loss %.4f", ckpt["epoch"], ckpt["metrics"]["loss"])
    return model, frozen, ckpt


def run_unlearning(spec: DatasetSpec, method: UnlearnMethod, args, device) -> Path | None:
    """Train one unlearning run.  Returns the output directory, or None if skipped."""
    # --train-seed varies TRAINING stochasticity (shuffle order, reparameterisation
    # noise) while spec.SEED continues to fix the data SPLIT -- vary the split and
    # runs stop being comparable at all.  Defaulting to SEED means every existing
    # run reproduces bit-identically.  Distinct from --u-seed/--fisher-seed/
    # --inlp-seed, which reseed only a method's own random vector: those studies
    # measured method-intrinsic variance and left training identical, so the two
    # sources of run-to-run spread are separable.
    train_seed = getattr(args, "train_seed", None)
    train_seed = SEED if train_seed is None else int(train_seed)
    seed_all(train_seed)

    max_batches = getattr(args, "max_batches", 0) or 0

    run_name = method.run_name(args)
    if max_batches:
        # Must stay parseable by nb_utils.discover_sweep_keys, hence the
        # trailing letters-then-digits form.
        run_name += f"_b{max_batches}"
    if train_seed != SEED:
        run_name += f"_ts{train_seed}"

    # A refit against a replicate baseline is a different run, so it needs its
    # own directory -- without this every refit would collide on the canonical
    # run name and the last one would win, silently.
    _bl = getattr(args, "baseline_ckpt", None)
    if _bl:
        _tag = Path(_bl).parent.name.split("_")[-1]      # e.g. "ts1"
        run_name += f"_bl{_tag}"

    out_dir = (Path(args.out_dir) if getattr(args, "out_dir", None)
               else spec.method_dir(method.name) / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "final.pt").exists():
        log.info("Already done -- skipping: %s", out_dir)
        return None

    # Provenance BEFORE training, so an interrupted run still records what it
    # was attempting.  The directory name is lossy -- it omits every flag left
    # at its default -- so it cannot serve as the record of a configuration.
    meta_path = write_run_meta(
        out_dir, args=args, method_name=method.name, dataset=spec.name,
        run_name=run_name,
        seeds={"split": SEED, "train": train_seed,
               "method": {k: v for k, v in vars(args).items()
                          if k.endswith("_seed") and k != "train_seed"}},
    )
    log.info("Provenance -> %s", meta_path.name)

    model, frozen, baseline_ckpt = load_baseline(
        spec, device, getattr(args, "baseline_ckpt", None))

    forget_loader, retain_loader = spec.concept_loaders(args.num_workers)
    val_loader = spec.val_loader(args.num_workers)

    # Budget reference. For a beta=1 model the recorded checkpoint loss IS the beta=1
    # val, so use it unchanged (keeps existing champions bit-identical). For a beta!=1
    # variant the recorded loss is at the training beta (recon + beta*KL), which is NOT
    # comparable to run_val's beta=1 val below -- the budget would be vacuous and every
    # epoch would pass. Measure the baseline's beta=1 retain val the same way run_val
    # does, so the budget and the per-epoch val sit on one yardstick.
    if getattr(spec, "train_beta", 1.0) != 1.0:
        # The stored, seeded reference (spec._beta1_retain_val; ledger S12) when the spec's own
        # baseline is loaded; an override checkpoint still gets a fresh measurement.
        from src.unlearning.spec import _beta1_retain_val
        baseline_val = (_beta1_retain_val(spec) if getattr(args, "baseline_ckpt", None) is None
                        else run_val(model, val_loader, device)["loss"])
        log.info("  beta=%.1f variant: retain budget from beta=1 retain-val %.2f "
                 "(recorded train-beta loss %.2f ignored)",
                 spec.train_beta, baseline_val, float(baseline_ckpt["metrics"]["loss"]))
    else:
        baseline_val = float(baseline_ckpt["metrics"]["loss"])
    val_tolerance = getattr(args, "val_tolerance", 0.10)
    val_budget    = baseline_val * (1.0 + val_tolerance)
    log.info("Forget batches: %d | Retain batches: %d",
             len(forget_loader), len(retain_loader))

    # One-shot methods do all their work in prepare() -- Fisher scrubbing
    # computes the FIM and applies its closed-form weight perturbation there.
    # They get no optimiser and exactly one pass, which only MEASURES the
    # already-scrubbed model.  See UnlearnMethod.one_shot.
    one_shot = getattr(method, "one_shot", False)

    ctx = method.prepare(model, frozen, forget_loader, retain_loader, device, args)

    trainable = method.trainable_params(model)
    optimizer = (None if one_shot
                 else torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4))
    n_epochs = 1 if one_shot else args.epochs

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())

    log.info("=" * 66)
    log.info("%s | %s | %s", spec.name, method.name, method.description)
    log.info("  concept=%s | lr=%.1e | %s",
             spec.concept_desc, args.lr,
             "  ".join(f"{k}={v}" for k, v in method.hparams(args).items()))
    log.info("  trainable params %d / %d | BatchNorm %s",
             n_train, n_total,
             "updating (train mode)" if method.train_mode else "frozen (eval mode)")
    if one_shot:
        log.info("  ONE-SHOT: perturbation applied in prepare(); the single pass "
                 "below measures it, no optimiser runs (--epochs ignored)")
    if max_batches:
        log.info("  budget %d batches/epoch (of %d)", max_batches, len(retain_loader))
    if ctx.get("info"):
        log.info("  %s", ctx["info"])
    log.info("  retain val budget %.2f (baseline %.2f + %.0f%%) | forget is better %s",
             val_budget, baseline_val, 100 * val_tolerance, method.forget_better)
    log.info("=" * 66)

    history = {"retain_loss": [], "forget_loss": [], "total_loss": [], "val_loss": []}
    best_val = math.inf
    best_trade: float | None = None
    trade_epoch: int | None = None
    metrics: dict = {}

    for epoch in range(1, n_epochs + 1):
        model.train(method.train_mode)
        t0 = time.time()
        sums = {"retain_loss": 0.0, "forget_loss": 0.0, "total_loss": 0.0}
        n = 0

        for retain_batch, forget_batch in zip(retain_loader, cycle(forget_loader)):
            if max_batches and n >= max_batches:
                break
            x_r = _batch_x(retain_batch, device)
            x_f = _batch_x(forget_batch, device)

            if one_shot:
                # Measure only.  Taking a gradient step here would undo the
                # closed-form perturbation prepare() just applied.
                with torch.no_grad():
                    loss, step_metrics = method.step(model, frozen, ctx, x_r, x_f, args)
            else:
                loss, step_metrics = method.step(model, frozen, ctx, x_r, x_f, args)

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()

            for k, v in step_metrics.items():
                # setdefault, not sums[k] += : a method is free to report extra
                # diagnostic metrics beyond retain/forget, and an unknown key
                # used to raise KeyError here.
                sums[k] = sums.get(k, 0.0) + (
                    0.0 if (isinstance(v, float) and math.isnan(v)) else v)
            sums["total_loss"] += loss.item()
            n += 1

        val   = run_val(model, val_loader, device)
        avgs  = {k: v / n for k, v in sums.items()}

        # Preserve the old convention: forget_loss is NaN when never computed.
        if math.isnan(step_metrics.get("forget_loss", 0.0)):
            avgs["forget_loss"] = float("nan")

        for k in avgs:
            history.setdefault(k, []).append(avgs[k])
        history["val_loss"].append(val["loss"])

        log.info("Epoch %2d/%d | retain=%.4f  forget=%.4f  total=%.4f | val=%.2f | %.1fs",
                 epoch, n_epochs, avgs["retain_loss"], avgs["forget_loss"],
                 avgs["total_loss"], val["loss"], time.time() - t0)

        metrics = {"loss": val["loss"],
                   "retain": avgs["retain_loss"],
                   "forget": avgs["forget_loss"]}

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(model, optimizer, epoch, metrics, out_dir / "best.pt")
            log.info("  -> best val %.4f", best_val)

        # Most forgetting among epochs that stay inside the retain-val budget.
        f_val = avgs["forget_loss"]
        if not math.isnan(f_val) and val["loss"] <= val_budget:
            improved = best_trade is None or (
                f_val > best_trade if method.forget_better == "higher"
                else f_val < best_trade)
            if improved:
                best_trade, trade_epoch = f_val, epoch
                save_checkpoint(model, optimizer, epoch, metrics,
                                out_dir / "best_tradeoff.pt")
                log.info("  -> best tradeoff: forget %.4f at val %.2f (budget %.2f)",
                         f_val, val["loss"], val_budget)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, metrics,
                            out_dir / f"epoch_{epoch:04d}.pt")

    if trade_epoch is None:
        log.warning("No epoch met the retain-val budget %.2f -- no best_tradeoff.pt. "
                    "Every setting here damages retain beyond tolerance.", val_budget)

    history["selection"] = {
        "baseline_val":   baseline_val,
        "val_tolerance":  val_tolerance,
        "val_budget":     val_budget,
        "forget_better":  method.forget_better,
        "best_val_epoch": int(history["val_loss"].index(min(history["val_loss"])) + 1),
        "tradeoff_epoch": trade_epoch,
        "max_batches":    max_batches,
        # One-shot runs have a single entry; "epoch" is not a tuning axis for
        # them (the noise scale is), so epoch-wise selection is degenerate.
        "one_shot":       one_shot,
    }

    save_checkpoint(model, optimizer, n_epochs, metrics, out_dir / "final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("Done -> %s", out_dir)
    return out_dir
