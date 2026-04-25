"""experiments/train_celeba.py

Entry point for training the baseline convolutional VAE on CelebA.

Usage
-----
# Standard run (reads configs/base_vae.yaml + configs/datasets/celeba.yaml):
    python experiments/train_celeba.py

# Override anything from the CLI:
    python experiments/train_celeba.py --latent-dim 32 --epochs 50 --device cuda

Notes on CelebA download
------------------------
torchvision will attempt to download CelebA automatically from Google Drive.
Google Drive throttles large downloads, so this often fails. If it does:

  1. Download the files manually from:
         https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
     You need: img_align_celeba.zip, list_attr_celeba.txt,
               list_eval_partition.txt, list_landmarks_align_celeba.txt

  2. Unzip img_align_celeba.zip so your folder looks like:
         data/celeba/img_align_celeba/000001.jpg ...
         data/celeba/list_attr_celeba.txt
         data/celeba/list_eval_partition.txt
         data/celeba/list_landmarks_align_celeba.txt

  3. Re-run with --no-download:
         python experiments/train_celeba.py --no-download
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.celeba import get_dataloaders
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline VAE on CelebA.")

    p.add_argument("--model-config", default="configs/base_vae.yaml")
    p.add_argument("--data-config",  default="configs/datasets/celeba.yaml")

    # Model
    p.add_argument("--latent-dim",    type=int,   default=None)
    p.add_argument("--base-channels", type=int,   default=None)

    # Training
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--kl-warmup-epochs", type=int,   default=None)
    p.add_argument("--kl-max-weight",    type=float, default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--seed",             type=int,   default=None)
    p.add_argument("--device",           type=str,   default=None)

    # CelebA-specific
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip torchvision download attempt (use if files are already on disk).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = load_yaml(args.model_config)
    data_cfg  = load_yaml(args.data_config)

    # Apply CLI overrides
    t = model_cfg["training"]
    m = model_cfg["model"]
    dl = data_cfg["dataloader"]
    ds = data_cfg["dataset"]

    if args.latent_dim:        m["latent_dim"]          = args.latent_dim
    if args.base_channels:     m["base_channels"]        = args.base_channels
    if args.epochs:            t["epochs"]               = args.epochs
    if args.lr:                t["lr"]                   = args.lr
    if args.kl_warmup_epochs:  t["kl_warmup_epochs"]     = args.kl_warmup_epochs
    if args.kl_max_weight:     t["kl_max_weight"]        = args.kl_max_weight
    if args.batch_size:        dl["batch_size"]          = args.batch_size
    if args.seed:              t["seed"]                 = args.seed
    if args.no_download:       ds["download"]            = False

    # Device
    device_str = args.device
    if device_str is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    log.info("Using device: %s", device)

    # Data — CelebA is always 3-channel RGB
    train_loader, val_loader, _ = get_dataloaders(
        data_root      = ds["data_root"],
        image_size     = ds["image_size"],
        batch_size     = dl["batch_size"],
        num_workers    = dl["num_workers"],
        download       = ds.get("download", True),
        include_labels = dl.get("include_labels", False),
    )

    # Model — 3-channel input, MSE loss for continuous pixels
    vae_cfg = VAEConfig(
        latent_dim    = m["latent_dim"],
        base_channels = m["base_channels"],
        in_channels   = ds["channels"],   # 3
        recon_loss    = "mse",
        eps           = m["eps"],
    )
    model = ConvVAE(vae_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    results_dir = data_cfg["results_dir"]

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
        checkpoint_dir   = results_dir,
        save_every       = model_cfg["checkpointing"]["save_every"],
        device           = device,
        seed             = t["seed"],
    )

    history_path = Path(results_dir) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved to %s", history_path)


if __name__ == "__main__":
    main()
