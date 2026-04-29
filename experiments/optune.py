"""
Optuna hyperparameter tuning for beta-VAE (dsprites / celeba).

- Saves full per-trial configs + metrics
- Uses MedianPruner for early stopping
- Tracks runtime + ETA (EMA-based estimator)
- Persists study in SQLite for resume support

Usage:
python experiments/optune.py --experiment dsprites_vae --n-trials 30
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_experiment_config, _deep_merge
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train
from src.data.dsprites import get_dataloaders as dsprites_loaders
from src.data.celeba import get_dataloaders as celeba_loaders


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

DATASET_LOADERS = {
    "dsprites": dsprites_loaders,
    "celeba": celeba_loaders,
}


# ---------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------

class TimeEstimator:
    def __init__(self, total_trials: int):
        self.total_trials = total_trials
        self.start = time.time()
        self.done = 0
        self.ema = None
        self.alpha = 0.3

    def update(self):
        now = time.time()
        elapsed = now - self.start

        self.done += 1
        avg = elapsed / self.done

        self.ema = avg if self.ema is None else (self.alpha * avg + (1 - self.alpha) * self.ema)

        remaining_trials = self.total_trials - self.done
        eta_seconds = remaining_trials * self.ema
        eta = datetime.now() + timedelta(seconds=eta_seconds)

        return {
            "elapsed": elapsed,
            "avg": self.ema,
            "remaining": eta_seconds,
            "eta": eta,
        }


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_12h(dt: datetime) -> str:
    return dt.strftime("%I:%M:%S %p")


# ---------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------

def make_objective(experiment_name: str, device: torch.device):

    def objective(trial: optuna.Trial) -> float:
        cfg = load_experiment_config(experiment_name)
        dataset_type = cfg["_dataset_type"]

        # dsprites200k baseline
        cfg = _deep_merge(cfg, {
            "model": {
                "latent_dim": trial.suggest_int("latent_dim", 5,20),
            },
            "training": {
                "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
                "kl_max_weight": trial.suggest_float("kl_max_weight", 0.1, 8),
                "kl_warmup_epochs": trial.suggest_int("kl_warmup_epochs", 5, 30),
                "epochs": 50,
            },
        })

        # # celeba vae baseline
        # cfg = _deep_merge(cfg, {
        #     "model": {
        #         "latent_dim": trial.suggest_int("latent_dim", 32,256),
        #     },
        #     "training": {
        #         "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        #         "kl_max_weight": trial.suggest_float("kl_max_weight", 0.02, 0.3),
        #         "kl_warmup_epochs": trial.suggest_int("kl_warmup_epochs", 10, 60),
        #         "epochs": 120,
        #     },
        # })

        trial_dir = Path(f"results/tuning/{experiment_name}/trial_{trial.number}")
        trial_dir.mkdir(parents=True, exist_ok=True)

        with open(trial_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # ---------------- data ----------------
        ds, dl, t = cfg["dataset"], cfg["dataloader"], cfg["training"]

        if dataset_type == "dsprites":
            train_loader, val_loader, _ = DATASET_LOADERS["dsprites"](
                npz_path=ds["npz_path"],
                batch_size=dl["batch_size"],
                val_fraction=ds["val_fraction"],
                test_fraction=ds["test_fraction"],
                num_workers=dl["num_workers"],
                seed=t["seed"],
            )
        else:
            train_loader, val_loader, _ = DATASET_LOADERS["celeba"](
                data_root=ds["data_root"],
                image_size=ds["image_size"],
                batch_size=dl["batch_size"],
                num_workers=dl["num_workers"],
                download=ds.get("download", True),
            )

        # ---------------- model ----------------
        m = cfg["model"]

        model = ConvVAE(VAEConfig(
            latent_dim=m["latent_dim"],
            base_channels=m.get("base_channels", 32),
            in_channels=ds["channels"],
            recon_loss=ds.get("recon_loss", "bce"),
            eps=m.get("eps", 1e-6),
        ))

        # ---------------- train ----------------
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=t["lr"],
            weight_decay=t["weight_decay"],
            epochs=t["epochs"],
            grad_clip=t["grad_clip"],
            kl_warmup_epochs=t["kl_warmup_epochs"],
            kl_max_weight=t["kl_max_weight"],
            use_cosine_lr=t["use_cosine_lr"],
            checkpoint_dir=str(trial_dir),
            save_every=0,
            device=device,
            seed=t["seed"],
            trial=trial,
        )

        torch.save(model.state_dict(), trial_dir / "model.pt")

        best_val = float(np.mean(history["val_loss"][-3:]))
        final_val = float(history["val_loss"][-1])

        result = {
            "best_val_loss": best_val,
            "final_val_loss": final_val,
            "epochs": len(history["val_loss"]),
        }

        with open(trial_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        return best_val

    return objective


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu")
    )

    Path("results/tuning").mkdir(parents=True, exist_ok=True)

    estimator = TimeEstimator(args.n_trials)

    def callback(study, trial):
        stats = estimator.update()

        print(
            f"\n[Trial {trial.number}] value={trial.value:.4f}\n"
            f"Elapsed:     {fmt_time(stats['elapsed'])}\n"
            f"Avg/trial:   {stats['avg']:.2f} sec\n"
            f"Remaining:   {fmt_time(stats['remaining'])}\n"
            f"ETA:         {fmt_12h(stats['eta'])}\n"
            f"Best so far: {study.best_value:.4f}"
        )

    study = optuna.create_study(
        study_name=args.experiment,
        direction="minimize",
        storage=f"sqlite:///results/tuning/{args.experiment}.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=48025845),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    study.optimize(
        make_objective(args.experiment, device),
        n_trials=args.n_trials,
        callbacks=[callback],
    )

    print("\n=== BEST TRIAL ===")
    print(f"Loss: {study.best_value:.4f}")
    print(f"Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()