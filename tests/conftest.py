"""
tests/conftest.py
==================
Shared fixtures. Everything runs on CPU with tiny models/datasets so the whole
suite finishes in well under a minute -- these are correctness tests, not
performance benchmarks.

Synthetic data (not the real dSprites/CelebA files) is used throughout so the
suite is hermetic: it runs on a machine that has never downloaded either
dataset, and never touches anything under results/ or data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vae import ConvVAE, VAEConfig

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny VAEs -- small enough that a handful of forward/backward passes on CPU
# is effectively instant, but with the real architecture (4 conv layers,
# 64x64 spatial path), so these exercise actual code, not a stand-in.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_vae_bce():
    """dSprites-shaped: 1 channel, BCE reconstruction, latent_dim=4."""
    torch.manual_seed(0)
    cfg = VAEConfig(latent_dim=4, base_channels=4, in_channels=1, recon_loss="bce")
    return ConvVAE(cfg).to(DEVICE)


@pytest.fixture
def tiny_vae_mse():
    """CelebA-shaped: 3 channel, MSE + Tanh reconstruction, latent_dim=4."""
    torch.manual_seed(0)
    cfg = VAEConfig(latent_dim=4, base_channels=4, in_channels=3, recon_loss="mse")
    return ConvVAE(cfg).to(DEVICE)


def make_batch(n, channels, recon_loss, seed=0):
    """A batch of fake images in the right domain for the given recon mode."""
    g = torch.Generator().manual_seed(seed)
    if recon_loss == "bce":
        return torch.randint(0, 2, (n, channels, 64, 64), generator=g).float()
    return torch.rand(n, channels, 64, 64, generator=g) * 2 - 1  # [-1, 1]


@pytest.fixture
def bce_batch():
    return make_batch(8, 1, "bce")


@pytest.fixture
def mse_batch():
    return make_batch(8, 3, "mse")


def save_checkpoint(model, path: Path, epoch=1, val_loss=0.0):
    """Minimal checkpoint in the same shape src/unlearning/loop.py expects."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "metrics": {"loss": val_loss},
        "model_config": model.cfg,
    }, path)


# ---------------------------------------------------------------------------
# Synthetic dSprites .npz -- same keys/shapes as the real file, tiny N.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dsprites_npz(tmp_path):
    """
    A tiny .npz matching DSpritesDataset's expected keys.

    N=900, shape column (index 1 of latents_classes) split evenly across
    {square=0, ellipse=1, heart=2} so forget (heart) and retain are both
    non-trivially sized after the train/val/test split.
    """
    rng = np.random.RandomState(0)
    n = 900
    imgs = rng.randint(0, 2, size=(n, 64, 64)).astype(np.uint8)

    latents_classes = np.zeros((n, 6), dtype=np.int64)
    latents_classes[:, 1] = np.tile([0, 1, 2], n // 3)          # shape, balanced
    latents_classes[:, 2] = rng.randint(0, 6, n)                 # scale
    latents_classes[:, 3] = rng.randint(0, 40, n)                # orientation
    latents_classes[:, 4] = rng.randint(0, 32, n)                # posX
    latents_classes[:, 5] = rng.randint(0, 32, n)                # posY

    latents_values = latents_classes.astype(np.float64)

    path = tmp_path / "synthetic_dsprites.npz"
    np.savez(path, imgs=imgs, latents_values=latents_values,
             latents_classes=latents_classes)
    return path


# ---------------------------------------------------------------------------
# Synthetic CelebA folder + attribute file -- same on-disk layout as the real
# dataset, tiny N.  Images must be >= 178x178 because get_transform always
# applies CenterCrop(178) before resizing.
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_celeba_dir(tmp_path):
    """
    tmp_path/
        img_align_celeba/   000001.jpg ... 000040.jpg  (200x200 RGB noise)
        list_attr_celeba.txt

    Eyeglasses is +1 for indices divisible by 5 (8 of 40 -> 20% prevalence,
    close enough to the real ~6.5% to exercise imbalance-sensitive code paths
    without needing a large N).
    """
    rng = np.random.RandomState(0)
    n = 40
    img_dir = tmp_path / "img_align_celeba"
    img_dir.mkdir()

    filenames = [f"{i:06d}.jpg" for i in range(1, n + 1)]
    for fn in filenames:
        arr = rng.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / fn)

    attr_names = ["Eyeglasses", "Smiling"]
    lines = [str(n), " ".join(attr_names)]
    for i, fn in enumerate(filenames, start=1):
        eyeglasses = 1 if i % 5 == 0 else -1
        smiling = 1 if i % 2 == 0 else -1
        lines.append(f"{fn} {eyeglasses} {smiling}")
    (tmp_path / "list_attr_celeba.txt").write_text("\n".join(lines) + "\n")

    return img_dir


# ---------------------------------------------------------------------------
# Tiny DatasetSpec -- for testing src/unlearning/loop.py without touching
# any real data file or the real dSprites/CelebA loaders.
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_spec(tmp_path, tiny_vae_bce):
    """A DatasetSpec whose loaders serve fixed, tiny, in-memory tensors."""
    from src.unlearning.spec import DatasetSpec
    from torch.utils.data import DataLoader, TensorDataset

    ckpt_path = tmp_path / "baseline.pt"
    save_checkpoint(tiny_vae_bce, ckpt_path, epoch=1, val_loss=1.23)

    forget_x = make_batch(16, 1, "bce", seed=1)
    retain_x = make_batch(32, 1, "bce", seed=2)
    val_x    = make_batch(16, 1, "bce", seed=3)
    eval_forget_x = make_batch(16, 1, "bce", seed=4)
    eval_retain_x = make_batch(16, 1, "bce", seed=5)

    class TinySpec(DatasetSpec):
        name          = "tiny"
        baseline_ckpt = ckpt_path
        results_dir   = tmp_path / "results"
        dir_prefix    = "tiny"
        batch_size    = 4
        num_workers   = 0
        concept_desc  = "toy concept"

        def concept_loaders(self, num_workers=None):
            return (DataLoader(TensorDataset(forget_x), batch_size=4, shuffle=True, drop_last=True),
                    DataLoader(TensorDataset(retain_x), batch_size=4, shuffle=True, drop_last=True))

        def val_loader(self, num_workers=None):
            return DataLoader(TensorDataset(val_x), batch_size=4)

        def eval_loaders(self, num_workers=None):
            return (DataLoader(TensorDataset(eval_forget_x), batch_size=4),
                    DataLoader(TensorDataset(eval_retain_x), batch_size=4))

    return TinySpec()
