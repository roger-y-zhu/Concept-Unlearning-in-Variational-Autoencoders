"""
Load random images, reconstruct them, and also generate fresh samples
from z ~ N(0, I) in latent space.
Works with dsprites and celeba experiment configs.
Usage:
    python scripts/reconstruct_three.py \
        --experiment dsprites_vae_baseline \
        --checkpoint results/dsprites/dsprites_vae_baseline/epoch_0150.pt
    python scripts/reconstruct_three.py \
        --experiment celeba75k_vae \
        --checkpoint results/celeba/celeba75k_vae/best.pt
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils.config import load_experiment_config
from src.models.vae import ConvVAE
from src.data.dsprites import get_dataloaders as dsprites_loaders
from src.data.celeba import get_dataloaders as celeba_loaders

DATASET_LOADERS = {
    "dsprites": dsprites_loaders,
    "celeba": celeba_loaders,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def unpack_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def to_display_image(x: torch.Tensor, recon_loss: str) -> np.ndarray:
    """Convert a raw model tensor to a display-ready numpy array.

    Inputs:
      - original images (CelebA): normalised to [-1, 1]  (recon_loss="mse")
      - original images (dSprites): binary {0, 1}         (recon_loss="bce")
      - reconstructions / generated:
          MSE path → decoder Tanh output in [-1, 1]
          BCE path → already passed through sigmoid, in [0, 1]

    All cases end up in [0, 1] after this function.
    """
    x = x.detach().cpu().float()
    if recon_loss == "mse":
        # [-1, 1] → [0, 1]  (covers both real images and MSE decoder output)
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.ndim == 3:
        return x.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    return x.squeeze(0).numpy()             # (1, H, W) → (H, W)


def postprocess_decode(x: torch.Tensor, recon_loss: str) -> torch.Tensor:
    """Apply the appropriate output activation to raw decoder logits.

    BCE: decoder outputs raw logits → apply sigmoid to get [0, 1]
    MSE: decoder already has Tanh applied → leave in [-1, 1];
         to_display_image handles the final denormalisation.
    """
    if recon_loss == "bce":
        return torch.sigmoid(x)
    return x  # MSE: stay in [-1, 1]


def build_loaders(cfg: dict):
    dataset_type = cfg["_dataset_type"]
    ds = cfg["dataset"]
    dl = cfg["dataloader"]
    t = cfg["training"]
    if dataset_type == "dsprites":
        return DATASET_LOADERS[dataset_type](
            npz_path=ds["npz_path"],
            batch_size=dl["batch_size"],
            val_fraction=ds["val_fraction"],
            test_fraction=ds["test_fraction"],
            num_workers=dl["num_workers"],
            seed=t["seed"],
            include_labels=False,
        )
    return DATASET_LOADERS[dataset_type](
        data_root=ds["data_root"],
        image_size=ds["image_size"],
        batch_size=dl["batch_size"],
        num_workers=dl["num_workers"],
        download=ds.get("download", True),
        include_labels=False,
    )


def choose_loader(split: str, loaders):
    train_loader, val_loader, test_loader = loaders
    candidates = {"train": train_loader, "val": val_loader, "test": test_loader}
    preferred = candidates.get(split)
    if preferred is not None and len(preferred.dataset) > 0:
        return preferred
    for name in ["test", "val", "train"]:
        loader = candidates.get(name)
        if loader is None:
            continue
        if len(loader.dataset) == 0:
            continue
        print(f"Requested split '{split}' unavailable, using '{name}' instead")
        return loader
    raise ValueError("No non-empty dataset split found")


def sample_images(loader: DataLoader, n: int, device: torch.device) -> torch.Tensor:
    sample_loader = DataLoader(
        loader.dataset,
        batch_size=max(64, n),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    batch = next(iter(sample_loader))
    imgs = unpack_images(batch)
    if imgs.size(0) < n:
        raise ValueError(f"Batch has only {imgs.size(0)} images, need at least {n}")
    idx = torch.randperm(imgs.size(0))[:n]
    return imgs[idx]


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    cfg = load_experiment_config(args.experiment)
    loaders = build_loaders(cfg)
    loader = choose_loader(args.split, loaders)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    model = ConvVAE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    imgs = sample_images(loader, args.n, device).to(device)

    with torch.no_grad():
        mu, std = model.encode(imgs)
        z_recon = mu
        recon = postprocess_decode(model.decode(z_recon), model.cfg.recon_loss)

        z_prior = torch.randn(args.n, model.cfg.latent_dim, device=device)
        generated = postprocess_decode(model.decode(z_prior), model.cfg.recon_loss)

    print(f"Loaded checkpoint : {args.checkpoint}")
    print(f"Device            : {device}")
    print(f"Input batch shape : {tuple(imgs.shape)}")
    print(f"Latent mu shape   : {tuple(mu.shape)}")
    print(f"Latent std shape  : {tuple(std.shape)}")
    print(f"Generated z shape : {tuple(z_prior.shape)}")

    n = imgs.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    recon_loss = model.cfg.recon_loss
    is_gray = imgs.shape[1] == 1  # single-channel (dSprites)

    for i in range(n):
        x     = to_display_image(imgs[i],      recon_loss)
        x_hat = to_display_image(recon[i],     recon_loss)
        x_gen = to_display_image(generated[i], recon_loss)

        imshow_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1} if is_gray else {}

        axes[i, 0].imshow(x,     **imshow_kwargs)
        axes[i, 1].imshow(x_hat, **imshow_kwargs)
        axes[i, 2].imshow(x_gen, **imshow_kwargs)

        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 1].set_title(f"Reconstruction {i + 1}")
        axes[i, 2].set_title(f"Generated {i + 1}")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle("VAE: original → reconstruction + prior samples", fontsize=14)
    plt.tight_layout()

    out_path = (
        Path(args.out)
        if args.out is not None
        else ROOT / "results" / "figures" / f"{args.experiment}_recon_and_generation.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()