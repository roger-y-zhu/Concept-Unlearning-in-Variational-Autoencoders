# experiments/train.py

import argparse, json, logging, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_experiment_config
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train

# dataset registry — keeps train.py clean
from src.data.dsprites import get_dataloaders as dsprites_loaders
from src.data.celeba   import get_dataloaders as celeba_loaders

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

DATASET_LOADERS = {
    "dsprites": dsprites_loaders,
    "celeba":   celeba_loaders,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="Name of experiment config, e.g. dsprites_beta_vae")
    p.add_argument("--device", default=None)
    # allow ad-hoc overrides without editing yaml
    p.add_argument("--latent-dim",      type=int,   default=None)
    p.add_argument("--kl-max-weight",   type=float, default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    return p.parse_args()

def apply_cli_overrides(cfg, args):
    if args.latent_dim:     cfg["model"]["latent_dim"]       = args.latent_dim
    if args.kl_max_weight:  cfg["training"]["kl_max_weight"] = args.kl_max_weight
    if args.epochs:         cfg["training"]["epochs"]        = args.epochs
    if args.lr:             cfg["training"]["lr"]            = args.lr

def main():
    args = parse_args()
    cfg  = load_experiment_config(args.experiment)
    apply_cli_overrides(cfg, args)

    # Device
    device_str = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(device_str)
    log.info("Experiment : %s", cfg["name"])
    log.info("Device     : %s", device)

    # Data — dispatch to the right loader
    dataset_type = cfg["_dataset_type"]
    ds, dl = cfg["dataset"], cfg["dataloader"]

    if dataset_type == "dsprites":
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](
            npz_path       = ds["npz_path"],
            batch_size     = dl["batch_size"],
            val_fraction   = ds["val_fraction"],
            test_fraction  = ds["test_fraction"],
            num_workers    = dl["num_workers"],
            seed           = cfg["training"]["seed"],
            include_labels = dl.get("include_labels", False),
        )
    else:  # celeba
        train_loader, val_loader, _ = DATASET_LOADERS[dataset_type](
            data_root      = ds["data_root"],
            image_size     = ds["image_size"],
            batch_size     = dl["batch_size"],
            num_workers    = dl["num_workers"],
            download       = ds.get("download", True),
            include_labels = dl.get("include_labels", False),
        )

    # Model
    m = cfg["model"]
    vae_cfg = VAEConfig(
        latent_dim    = m["latent_dim"],
        base_channels = m["base_channels"],
        in_channels   = ds["channels"],
        recon_loss    = ds.get("recon_loss", "bce"),
        eps           = m["eps"],
    )
    model = ConvVAE(VAEConfig(
        latent_dim=m["latent_dim"],
        base_channels=m.get("base_channels", 32),
        in_channels=ds["channels"],
        recon_loss=ds.get("recon_loss", "bce"),
        eps=m.get("eps", 1e-6),
    ))

    # Train
    t = cfg["training"]
    history = train(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        lr               = t["lr"],
        weight_decay     = t["weight_decay"],
        epochs           = t["epochs"],
        grad_clip        = t["grad_clip"],
        kl_warmup_epochs = t["kl_warmup_epochs"],
        kl_max_weight    = t["kl_max_weight"],
        use_cosine_lr    = t["use_cosine_lr"],
        checkpoint_dir   = cfg["results_dir"],
        save_every       = cfg["checkpointing"]["save_every"],
        device           = device,
        seed             = t["seed"],
    )

    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    # also save the resolved config for reproducibility
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

if __name__ == "__main__":
    main()