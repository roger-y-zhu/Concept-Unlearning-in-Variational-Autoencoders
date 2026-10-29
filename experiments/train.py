# experiments/train.py
#  python experiments/train.py --experiment dsprites_baseline

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.misc import fmt_time, fmt_12h, speak
from src.utils.config import load_experiment_config
from src.models.vae import ConvVAE, VAEConfig
from src.models.flow import RealNVP, FlowConfig
from src.training.trainer import train

# dataset registry
from src.data.dsprites import get_dataloaders as dsprites_loaders, get_retain_dataloaders as dsprites_retain_loaders
from src.data.celeba import (
    get_dataloaders        as celeba_loaders,
    get_retain_dataloaders as celeba_retain_loaders,
)
from src.data.mpi3d import (
    get_dataloaders        as mpi3d_loaders,
    get_retain_dataloaders as mpi3d_retain_loaders,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%m-%d %I:%M:%S %p",
)

DATASET_LOADERS = {
    "dsprites":        dsprites_loaders,
    "dsprites_retain": dsprites_retain_loaders,
    "celeba":          celeba_loaders,
    "celeba_retain":   celeba_retain_loaders,
    "mpi3d":           mpi3d_loaders,
    "mpi3d_retain":    mpi3d_retain_loaders,
}


def compute_eta(start_time: float, current_epoch: int, total_epochs: int) -> datetime:
    elapsed = time.time() - start_time
    avg_epoch = elapsed / max(current_epoch, 1)
    remaining_epochs = total_epochs - current_epoch
    remaining_seconds = remaining_epochs * avg_epoch
    return datetime.now() + timedelta(seconds=remaining_seconds)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiment",
        required=True,
        help="Name of experiment config, e.g. dsprites_baseline",
    )

    p.add_argument("--device", default=None)

    # ad-hoc overrides
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--kl-max-weight", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train-seed", type=int, default=None,
                   help="Seed TRAINING stochasticity (weight init, shuffle order, "
                        "reparameterisation noise) while leaving the data SPLIT at "
                        "training.seed. Mirrors experiments/unlearn/run.py. Appends "
                        "_ts<N> to the experiment name so replicates land in their "
                        "own results directory. Omitting it reproduces existing "
                        "runs exactly. "
                        "NOTE the config's `training.seed` drives BOTH the split and "
                        "the trainer, so overriding that instead would change which "
                        "images each replicate saw -- making the replicates "
                        "incomparable rather than repeated.")

    return p.parse_args()


def apply_cli_overrides(cfg, args):
    if args.latent_dim is not None:
        cfg["model"]["latent_dim"] = args.latent_dim

    if args.kl_max_weight is not None:
        cfg["training"]["kl_max_weight"] = args.kl_max_weight

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    if args.train_seed is not None:
        # Separate key from training.seed: the latter still seeds the data
        # split, which must NOT vary between replicates of the same model.
        cfg["training"]["train_seed"] = args.train_seed
        cfg["name"] = f"{cfg['name']}_ts{args.train_seed}"
        results_group = cfg["results_dir"].rsplit("/", 2)[1]
        cfg["results_dir"] = f"results/{results_group}/{cfg['name']}"


def main():
    overall_start = time.time()

    args = parse_args()
    cfg = load_experiment_config(args.experiment)
    apply_cli_overrides(cfg, args)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device_str = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    device = torch.device(device_str)

    # --------------------------------------------------
    # Config summary
    # --------------------------------------------------

    dataset_type = cfg["_dataset_type"]
    ds = cfg["dataset"]
    dl = cfg["dataloader"]
    m = cfg["model"]
    t = cfg["training"]

    log.info("=" * 70)
    log.info("Experiment Summary")
    log.info("=" * 70)

    log.info("Experiment        : %s", cfg["name"])
    log.info("Dataset           : %s", dataset_type)
    log.info("Device            : %s", device)

    log.info("")

    log.info("Model Parameters")
    if cfg["_model_type"] == "flow":
        log.info("  model           : RealNVP flow")
        log.info("  n_coupling      : %d", m.get("n_coupling", 8))
        log.info("  hidden_channels : %d", m.get("hidden_channels", 64))
        log.info("  in_channels     : %d", ds["channels"])
    else:
        log.info("  latent_dim      : %d", m["latent_dim"])
        log.info("  base_channels   : %d", m.get("base_channels", 32))
        log.info("  in_channels     : %d", ds["channels"])
        log.info("  recon_loss      : %s", ds.get("recon_loss", "bce"))

    log.info("")

    log.info("Training Parameters")
    log.info("  epochs          : %d", t["epochs"])
    log.info("  lr              : %.6f", t["lr"])
    log.info("  weight_decay    : %.6f", t["weight_decay"])
    log.info("  grad_clip       : %.2f", t["grad_clip"])
    log.info("  kl_max_weight   : %.4f", t["kl_max_weight"])
    log.info("  kl_warmup_ep    : %d", t["kl_warmup_epochs"])
    log.info("  cosine_lr       : %s", t["use_cosine_lr"])
    log.info("  seed            : %d", t["seed"])

    log.info("")

    log.info("Data Parameters")
    log.info("  source          : %s",
             ds.get("npz_path", ds.get("data_root", "N/A")))
    log.info("  batch_size      : %d", dl["batch_size"])
    log.info("  num_workers     : %d", dl["num_workers"])

    if dataset_type in ("dsprites", "dsprites_retain"):
        log.info("  val_fraction    : %.2f", ds["val_fraction"])
        log.info("  test_fraction   : %.2f", ds["test_fraction"])
        if dataset_type == "dsprites_retain":
            log.info("  concept_col     : %d", ds.get("concept_col", 1))
            log.info("  concept_val     : %d (excluded)", ds.get("concept_val", 2))

    log.info("")
    log.info("Results Directory : %s", cfg["results_dir"])
    log.info("=" * 70)

    # --------------------------------------------------
    # Data loading
    # --------------------------------------------------

    data_start = time.time()

    if dataset_type in ("dsprites", "dsprites_retain"):
        loader_kwargs = dict(
            npz_path=ds["npz_path"],
            batch_size=dl["batch_size"],
            val_fraction=ds["val_fraction"],
            test_fraction=ds["test_fraction"],
            num_workers=dl["num_workers"],
            seed=t["seed"],
            include_labels=dl.get("include_labels", False),
        )
        if dataset_type == "dsprites_retain":
            loader_kwargs["concept_col"] = ds.get("concept_col", 1)
            loader_kwargs["concept_val"] = ds.get("concept_val", 2)
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](**loader_kwargs)
    elif dataset_type in ("mpi3d", "mpi3d_retain"):
        loader_kwargs = dict(
            npz_path=ds["npz_path"],
            batch_size=dl["batch_size"],
            val_fraction=ds["val_fraction"],
            test_fraction=ds["test_fraction"],
            num_workers=dl["num_workers"],
            seed=t["seed"],
            include_labels=dl.get("include_labels", False),
        )
        if dataset_type == "mpi3d_retain":
            loader_kwargs["forget_factor"] = ds.get("forget_factor", 0)
            loader_kwargs["forget_val"] = ds.get("forget_val", 2)
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](**loader_kwargs)
    elif dataset_type == "celeba_retain":
        train_loader, val_loader, _ = celeba_retain_loaders(
            data_root    = ds["data_root"],
            forget_attr  = ds.get("forget_attr", "Eyeglasses"),
            image_size   = ds.get("image_size", 64),
            batch_size   = dl["batch_size"],
            num_workers  = dl["num_workers"],
            val_fraction = ds.get("val_fraction", 0.1),
            test_fraction= ds.get("test_fraction", 0.0),
            seed         = t["seed"],
        )
    else:
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](
            data_root=ds["data_root"],
            image_size=ds["image_size"],
            batch_size=dl["batch_size"],
            num_workers=dl["num_workers"],
            download=ds.get("download", True),
            include_labels=dl.get("include_labels", False),
        )

    data_elapsed = time.time() - data_start
    log.info("Data loading complete in %s", fmt_time(data_elapsed))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    if cfg["_model_type"] == "flow":
        model = RealNVP(FlowConfig(
            in_channels=ds["channels"],
            image_size=ds.get("image_size", 64),
            n_coupling=m.get("n_coupling", 8),
            hidden_channels=m.get("hidden_channels", 64),
            levels=m.get("levels", 256),
            alpha=m.get("alpha", 0.05),
        ))
    else:
        model = ConvVAE(VAEConfig(
            latent_dim=m["latent_dim"],
            base_channels=m.get("base_channels", 32),
            in_channels=ds["channels"],
            recon_loss=ds.get("recon_loss", "bce"),
            eps=m.get("eps", 1e-6),
        ))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    log.info("Total parameters     : %s", f"{total_params:,}")
    log.info("Trainable parameters : %s", f"{trainable_params:,}")

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    train_start_time = time.time()
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
        objective=t.get("objective", "elbo"),
        tc_beta=float(t.get("tc_beta", 1.0)),
        dataset_size=len(train_loader.dataset),
        use_cosine_lr=t["use_cosine_lr"],
        checkpoint_dir=cfg["results_dir"],
        save_every=cfg["checkpointing"]["save_every"],
        device=device,
        # train_seed (if given) seeds ONLY the trainer -- weight init, shuffle
        # order, reparameterisation noise.  The loaders above keep t["seed"], so
        # every replicate sees exactly the same data split and the runs stay
        # comparable.  Overriding t["seed"] itself would vary the split too,
        # which would make replicates different experiments rather than repeats.
        seed=t.get("train_seed", t["seed"]),
    )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    total_elapsed = time.time() - overall_start

    log.info("")
    log.info("=" * 70)
    log.info("Training complete")
    log.info("Total runtime: %s", fmt_time(total_elapsed))
    log.info("Finished at : %s", fmt_12h(datetime.now()))
    log.info("=" * 70)

    speak(f"Training complete")

if __name__ == "__main__":
    main()
