"""
Optuna hyperparameter search for CelebA VAE training.

Fixed:  kl_max_weight = 1.0
Tuned:  latent_dim, lr, kl_warmup_epochs, weight_decay, base_channels, batch_size

Objective: mean val_loss (true ELBO at β=1) over the last 5 epochs.
Pruning:   MedianPruner — trials worse than the median at the same epoch are cut early.
Storage:   SQLite (resume-safe — re-running the same command continues where it left off).

Usage:
    python experiments/optune.py --experiment celeba_baseline --n-trials 50
    python experiments/optune.py --experiment celeba_baseline --n-trials 50 --tune-epochs 60
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.misc import fmt_time, fmt_12h
from src.utils.config import load_experiment_config, _deep_merge
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train
from src.data.celeba import get_dataloaders as celeba_loaders

optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-epoch spam


# ─────────────────────────────────────────────────────────────────────────────
# ETA tracker
# ─────────────────────────────────────────────────────────────────────────────

class TimeEstimator:
    """EMA-based per-trial time estimator."""
    def __init__(self, total: int, alpha: float = 0.3):
        self.total = total
        self.done  = 0
        self.alpha = alpha
        self.ema   = None
        self.start = time.time()

    def update(self) -> dict:
        self.done += 1
        elapsed  = time.time() - self.start
        avg      = elapsed / self.done
        self.ema = avg if self.ema is None else self.alpha * avg + (1 - self.alpha) * self.ema
        remaining_s = (self.total - self.done) * self.ema
        return {
            "elapsed":   elapsed,
            "avg":       self.ema,
            "remaining": remaining_s,
            "eta":       datetime.now() + timedelta(seconds=remaining_s),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

def suggest_hparams(trial: optuna.Trial, tune_epochs: int) -> dict:
    """
    CelebA search space — kl_max_weight is fixed at 1.0 throughout.

    latent_dim:        64–256  — CelebA needs far more capacity than dSprites (8).
                                 Sampled from discrete set to avoid tiny differences.
    lr:                1e-4–5e-3 log-uniform — wide enough to catch Adam's sweet spot
                                 on MSE-loss face images.
    kl_warmup_epochs:  5–50    — how many epochs before β reaches 1.0.  Capped well
                                 below tune_epochs so the model always reaches full KL.
    weight_decay:      1e-5–1e-2 log-uniform — regularises the large latent space.
    base_channels:     32 or 64 — doubles encoder/decoder capacity. 64 is ~4× params.
    batch_size:        64/128/256 — trades step noise for GPU utilisation.
    """
    return {
        "latent_dim":        trial.suggest_categorical("latent_dim",        [64, 96, 128, 160, 192, 256]),
        "lr":                trial.suggest_float(      "lr",                 1e-4, 5e-3, log=True),
        "kl_warmup_epochs":  trial.suggest_int(        "kl_warmup_epochs",  5, min(50, tune_epochs - 5)),
        "weight_decay":      trial.suggest_float(      "weight_decay",       1e-5, 1e-2, log=True),
        "base_channels":     trial.suggest_categorical("base_channels",     [32, 64]),
        "batch_size":        trial.suggest_categorical("batch_size",        [64, 128, 256]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(experiment_name: str, device: torch.device, tune_epochs: int, subset: int | None = None):

    def objective(trial: optuna.Trial) -> float:
        hp  = suggest_hparams(trial, tune_epochs)
        cfg = load_experiment_config(experiment_name)

        cfg = _deep_merge(cfg, {
            "model": {
                "latent_dim":   hp["latent_dim"],
                "base_channels": hp["base_channels"],
            },
            "training": {
                "lr":               hp["lr"],
                "kl_max_weight":    1.0,                    # fixed — never varied
                "kl_warmup_epochs": hp["kl_warmup_epochs"],
                "weight_decay":     hp["weight_decay"],
                "epochs":           tune_epochs,
            },
            "dataloader": {
                "batch_size": hp["batch_size"],
            },
        })

        ds = cfg["dataset"]
        dl = cfg["dataloader"]
        t  = cfg["training"]

        # ── Data ──────────────────────────────────────────────────────────────
        train_loader, val_loader, _ = celeba_loaders(
            data_root=ds["data_root"],
            image_size=ds["image_size"],
            batch_size=dl["batch_size"],
            num_workers=dl["num_workers"],
            val_fraction=ds.get("val_fraction", 0.1),
            seed=t["seed"],
            subset=subset,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        m = cfg["model"]
        model = ConvVAE(VAEConfig(
            latent_dim=m["latent_dim"],
            base_channels=m["base_channels"],
            in_channels=ds["channels"],
            recon_loss=ds.get("recon_loss", "mse"),
            eps=m.get("eps", 1e-6),
        ))

        # ── Train ─────────────────────────────────────────────────────────────
        trial_dir = Path(f"results/tuning/{experiment_name}/trial_{trial.number:03d}")
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "config.json", "w") as f:
            json.dump({**cfg, "hparams": hp}, f, indent=2, default=str)

        try:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=t["lr"],
                weight_decay=t["weight_decay"],
                epochs=t["epochs"],
                grad_clip=t["grad_clip"],
                kl_warmup_epochs=t["kl_warmup_epochs"],
                kl_max_weight=1.0,
                use_cosine_lr=t["use_cosine_lr"],
                checkpoint_dir=None,      # no per-trial checkpoints (save space)
                save_every=None,
                device=device,
                seed=t["seed"],
                trial=trial,              # enables epoch-level pruning in trainer.py
            )
        except optuna.exceptions.TrialPruned:
            raise

        # ── Score: mean ELBO over the last 5 epochs ───────────────────────────
        # val_loss = recon + kl  at β=1.0  (true ELBO, lower is better)
        tail = 5
        score = float(np.mean(history["val_loss"][-tail:]))

        result = {
            "score":          score,
            "val_loss":       float(np.mean(history["val_loss"][-tail:])),
            "val_recon":      float(np.mean(history["val_recon"][-tail:])),
            "val_kl":         float(np.mean(history["val_kl"][-tail:])),
            "epochs_run":     len(history["val_loss"]),
            "hparams":        hp,
        }

        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return score

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optuna VAE hyperparameter search")
    parser.add_argument("--experiment",   required=True,      help="Experiment config name, e.g. celeba_baseline")
    parser.add_argument("--n-trials",     type=int, default=50)
    parser.add_argument("--tune-epochs",  type=int, default=50, help="Epochs per trial (shorter than full training)")
    parser.add_argument("--subset",       type=int, default=75_000,
                                          help="Training images per trial (default 75k; pass 0 for full dataset)")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()
    args.subset = args.subset if args.subset > 0 else None   # 0 means full dataset

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
    )

    db_path = Path(f"results/tuning/{args.experiment}.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    estimator = TimeEstimator(args.n_trials)

    # ── Study ─────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.experiment,
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,             # safe to re-run — continues from last trial
        sampler=optuna.samplers.TPESampler(seed=48025845),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8,          # don't prune until 8 complete trials exist
            n_warmup_steps=15,           # don't prune within the first 15 epochs of a trial
            interval_steps=2,            # check every 2 epochs after warmup
        ),
    )

    # ── Callback: print after every trial ─────────────────────────────────────
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            status = f"PRUNED at epoch {trial.last_step}" if trial.state == optuna.trial.TrialState.PRUNED else "FAILED"
            print(f"\n[Trial {trial.number:3d}] {status}")
            return

        stats = estimator.update()
        best  = study.best_trial

        print(
            f"\n{'─'*60}\n"
            f"[Trial {trial.number:3d}]  score={trial.value:.4f}\n"
            f"  Params : {trial.params}\n"
            f"  Best   : trial {best.number} → {best.value:.4f}  {best.params}\n"
            f"  Done   : {estimator.done}/{estimator.total}\n"
            f"  Elapsed: {fmt_time(stats['elapsed'])}   "
            f"Avg/trial: {fmt_time(stats['avg'])}   "
            f"ETA: {fmt_12h(stats['eta'])}\n"
            f"{'─'*60}"
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    subset_str = f"{args.subset:,}" if args.subset else "full dataset"
    print(f"Study      : {args.experiment}")
    print(f"Device     : {device}")
    print(f"Trials     : {args.n_trials}  ({args.tune_epochs} epochs each)")
    print(f"Subset     : {subset_str} images per trial")
    print(f"DB         : {db_path}")
    print(f"Completed  : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} existing\n")

    study.optimize(
        make_objective(args.experiment, device, args.tune_epochs, subset=args.subset),
        n_trials=args.n_trials,
        callbacks=[callback],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL:  {best.number}  →  score = {best.value:.4f}")
    print(f"Params:")
    for k, v in best.params.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*60}\n")

    # Save best-trial summary
    out_path = Path(f"results/tuning/{args.experiment}/best_trial.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trial": best.number, "score": best.value, "params": best.params}, f, indent=2)
    print(f"Best trial saved → {out_path}")


if __name__ == "__main__":
    main()
