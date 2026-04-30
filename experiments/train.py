# experiments/train.py
#  python experiments/train.py --experiment dsprites_vae_baseline

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

from src.utils.misc import fmt_time, fmt_12h

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_experiment_config
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train

# dataset registry
from src.data.dsprites import get_dataloaders as dsprites_loaders
from src.data.celeba import get_dataloaders as celeba_loaders

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%m-%d %I:%M:%S %p",
)

DATASET_LOADERS = {
    "dsprites": dsprites_loaders,
    "celeba": celeba_loaders,
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
        help="Name of experiment config, e.g. dsprites_beta_vae",
    )

    p.add_argument("--device", default=None)

    # ad-hoc overrides
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--kl-max-weight", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

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
    log.info("  latent_dim      : %d", m["latent_dim"])
    log.info("  base_channels   : %d", m.get("base_channels", 32))
    log.info("  in_channels     : %d", ds["channels"])
    log.info("  recon_loss      : %s", ds.get("recon_loss", "bce"))

    total_params_est = (
            m["latent_dim"]
            * m.get("base_channels", 32)
    )
    log.info("  quick size hint : latent=%d × base=%d",
             m["latent_dim"],
             m.get("base_channels", 32))

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

    if dataset_type == "dsprites":
        log.info("  val_fraction    : %.2f", ds["val_fraction"])
        log.info("  test_fraction   : %.2f", ds["test_fraction"])

    log.info("")
    log.info("Results Directory : %s", cfg["results_dir"])
    log.info("=" * 70)

    # --------------------------------------------------
    # Data loading
    # --------------------------------------------------

    data_start = time.time()

    if dataset_type == "dsprites":
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](
            npz_path=ds["npz_path"],
            batch_size=dl["batch_size"],
            val_fraction=ds["val_fraction"],
            test_fraction=ds["test_fraction"],
            num_workers=dl["num_workers"],
            seed=t["seed"],
            include_labels=dl.get("include_labels", False),
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
        use_cosine_lr=t["use_cosine_lr"],
        checkpoint_dir=cfg["results_dir"],
        save_every=cfg["checkpointing"]["save_every"],
        device=device,
        seed=t["seed"],
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


if __name__ == "__main__":
    main()
