"""
experiments/optune_beta.py
==========================
Dataset-agnostic Optuna search for the beta-VAE variant runs. Unlike
experiments/optune.py (CelebA-only, tunes latent_dim, fixes beta=1), this driver:

  * reads the dataset AND the two HELD constants -- latent_dim and kl_max_weight --
    from the experiment config, and NEVER tunes them (the user's policy: hold
    latent_dim + beta fixed, retune everything else, so beta is the only
    manipulated axis vs the beta=1 models);
  * tunes lr, base_channels, kl_warmup_epochs, weight_decay, batch_size;
  * scores each trial on the model's OWN-beta validation objective
    (mean val_loss over the last 5 epochs) -- exactly the quantity trainer.py
    selects best.pt by, so the tuner and the final training agree. At beta=4 this
    is recon + 4*KL, matching the beta=1 methodology at its own beta (recon + 1*KL).

Usage:
    python experiments/optune_beta.py --experiment dsprites_b4_baseline --n-trials 40 --tune-epochs 40

Writes results/tuning/<experiment>.db (resume-safe) and
results/tuning/<experiment>/best_trial.json (the winning hyperparameters, ready to
paste into the experiment config).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_experiment_config, _deep_merge
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _subset_loader(loader, subset, batch_size, num_workers):
    """HPO speed: restrict a train loader to `subset` samples (the final baseline
    is later trained on the FULL set via experiments/train.py). The underlying
    dataset is already a random_split Subset, so the first `subset` indices are a
    representative sample. Returns the loader unchanged if subset is None/too big."""
    if not subset:
        return loader
    from torch.utils.data import Subset, DataLoader
    base = loader.dataset
    if subset >= len(base):
        return loader
    sub = Subset(base, list(range(subset)))
    return DataLoader(sub, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def make_loaders(cfg, batch_size, subset):
    """(train_loader, val_loader) for the config's dataset. Mirrors experiments/
    train.py's dispatch; only the baseline (non-retain) datasets are tuned."""
    dtype = cfg["_dataset_type"]
    ds, t = cfg["dataset"], cfg["training"]
    nw = cfg["dataloader"]["num_workers"]
    if dtype == "dsprites":
        from src.data.dsprites import get_dataloaders
        tr, va, _ = get_dataloaders(
            npz_path=ds["npz_path"], batch_size=batch_size,
            val_fraction=ds["val_fraction"], test_fraction=ds["test_fraction"],
            num_workers=nw, seed=t["seed"], include_labels=False)
        return _subset_loader(tr, subset, batch_size, nw), va
    if dtype == "mpi3d":
        from src.data.mpi3d import get_dataloaders
        tr, va, _ = get_dataloaders(
            npz_path=ds["npz_path"], batch_size=batch_size,
            val_fraction=ds["val_fraction"], test_fraction=ds["test_fraction"],
            num_workers=nw, seed=t["seed"], include_labels=False)
        return _subset_loader(tr, subset, batch_size, nw), va
    if dtype == "celeba":
        from src.data.celeba import get_dataloaders
        tr, va, _ = get_dataloaders(
            data_root=ds["data_root"], image_size=ds.get("image_size", 64),
            batch_size=batch_size, num_workers=nw,
            val_fraction=ds.get("val_fraction", 0.1), seed=t["seed"], subset=subset)
        return tr, va
    raise ValueError(f"optune_beta does not handle dataset type {dtype!r}")


def suggest_hparams(trial, tune_epochs):
    """Search space -- latent_dim and kl_max_weight are NOT here; they are held
    fixed at the config's values."""
    return {
        "lr":               trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "kl_warmup_epochs": trial.suggest_int("kl_warmup_epochs", 5, min(50, tune_epochs - 5)),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "base_channels":    trial.suggest_categorical("base_channels", [32, 64]),
        "batch_size":       trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def make_objective(experiment, device, tune_epochs, subset):
    def objective(trial):
        hp = suggest_hparams(trial, tune_epochs)
        cfg = load_experiment_config(experiment)
        beta = float(cfg["training"]["kl_max_weight"])   # HELD (e.g. 4.0)
        latent = int(cfg["model"]["latent_dim"])         # HELD (e.g. 8)
        objective = cfg["training"].get("objective", "elbo")  # HELD
        tc_beta = float(cfg["training"].get("tc_beta", 1.0))    # HELD (beta-TCVAE knob)
        cfg = _deep_merge(cfg, {
            "model": {"base_channels": hp["base_channels"]},
            "training": {"lr": hp["lr"], "kl_warmup_epochs": hp["kl_warmup_epochs"],
                         "weight_decay": hp["weight_decay"], "epochs": tune_epochs},
            "dataloader": {"batch_size": hp["batch_size"]},
        })
        ds, t = cfg["dataset"], cfg["training"]
        tr, va = make_loaders(cfg, hp["batch_size"], subset)
        model = ConvVAE(VAEConfig(
            latent_dim=latent, base_channels=hp["base_channels"],
            in_channels=ds["channels"], recon_loss=ds.get("recon_loss", "mse"),
            eps=cfg["model"].get("eps", 1e-6)))
        history = train(
            model=model, train_loader=tr, val_loader=va,
            lr=t["lr"], weight_decay=t["weight_decay"], epochs=tune_epochs,
            grad_clip=t["grad_clip"], kl_warmup_epochs=t["kl_warmup_epochs"],
            kl_max_weight=beta, use_cosine_lr=t["use_cosine_lr"],
            objective=objective, tc_beta=tc_beta, dataset_size=len(tr.dataset),
            checkpoint_dir=None, save_every=None, device=device,
            seed=t["seed"], trial=trial)
        tail = 5
        score = float(np.mean(history["val_loss"][-tail:]))
        rec = {"score": score,
               "val_loss": score,
               "val_recon": float(np.mean(history["val_recon"][-tail:])),
               "val_kl": float(np.mean(history["val_kl"][-tail:])),
               "beta": beta, "latent_dim": latent,
               "epochs_run": len(history["val_loss"]), "hparams": hp}
        d = Path(f"results/tuning/{experiment}/trial_{trial.number:03d}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(rec, indent=2))
        return score
    return objective


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--tune-epochs", type=int, default=40)
    ap.add_argument("--subset", type=int, default=0, help="images/trial for celeba; 0=full (dsprites/mpi3d ignore this)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    subset = args.subset if args.subset > 0 else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cfg0 = load_experiment_config(args.experiment)
    beta = cfg0["training"]["kl_max_weight"]; latent = cfg0["model"]["latent_dim"]
    print(f"Study {args.experiment}: HELD latent_dim={latent}, kl_max_weight(beta)={beta}; "
          f"tuning lr/base_channels/kl_warmup/weight_decay/batch_size")
    print(f"Device {device} | {args.n_trials} trials x {args.tune_epochs} epochs | subset={subset}")

    db = Path(f"results/tuning/{args.experiment}.db"); db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.experiment, direction="minimize",
        storage=f"sqlite:///{db}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=48025845),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15, interval_steps=2))

    start = time.time()
    def cb(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            b = study.best_trial
            print(f"[trial {trial.number:3d}] score={trial.value:.3f} {trial.params} | "
                  f"best={b.value:.3f} (trial {b.number}) | {(time.time()-start)/60:.1f} min")
        else:
            print(f"[trial {trial.number:3d}] {trial.state.name} at step {trial.last_step}")

    study.optimize(make_objective(args.experiment, device, args.tune_epochs, subset),
                   n_trials=args.n_trials, callbacks=[cb])

    best = study.best_trial
    out = {"experiment": args.experiment, "beta": beta, "latent_dim": latent,
           "best_value_ownbeta_valloss": best.value, "best_params": best.params,
           "n_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
    p = Path(f"results/tuning/{args.experiment}/best_trial.json")
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=2))
    print("\nBEST:", json.dumps(out, indent=2))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
