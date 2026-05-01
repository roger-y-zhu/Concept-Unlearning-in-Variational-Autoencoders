"""VAE training logic.

Responsibilities
----------------
- One epoch of training / validation
- KL annealing (linear warmup from 0 → 1 over `kl_warmup_epochs`)
- Gradient clipping
- Checkpoint saving (best val loss + periodic)
- Metric logging via Python logging

No model architecture or data loading lives here.
"""

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils.misc import fmt_12h
from src.utils.seed import seed_all
from datetime import datetime, timedelta
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KL annealing
# ---------------------------------------------------------------------------

def kl_weight(epoch: int, warmup_epochs: int, max_weight: float = 1.0) -> float:
    """Linear KL warmup: 0 at epoch 0, max_weight at warmup_epochs.

    Prevents posterior collapse early in training by letting the model
    focus on reconstruction before regularisation pressure kicks in.
    """
    if warmup_epochs <= 0:
        return max_weight
    return min(max_weight, max_weight * epoch / warmup_epochs)


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    kl_w: float,
    grad_clip: float,
) -> dict[str, float]:
    """Run one train or eval epoch.

    Args:
        model:     The VAE.
        loader:    DataLoader (train or val).
        optimizer: If None, run in eval mode (no grad updates).
        device:    Compute device.
        kl_w:      Current KL weight (β).
        grad_clip: Max gradient norm (only used during training).

    Returns:
        Dict with averaged 'loss', 'loss_recon', 'loss_kl'.
    """
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = {"loss": 0.0, "loss_recon": 0.0, "loss_kl": 0.0}
    n_batches = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            # Support dataloaders that return (img,) or (img, labels, …)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            out = model(x, kl_weight=kl_w)

            if training:
                optimizer.zero_grad()
                out.loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            totals["loss"]       += out.loss.item()
            totals["loss_recon"] += out.loss_recon.item()
            totals["loss_kl"]    += out.loss_kl.item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    # Optimisation
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    grad_clip: float = 1.0,
    # KL schedule
    kl_warmup_epochs: int = 10,
    kl_max_weight: float = 1.0,
    # LR schedule
    use_cosine_lr: bool = True,
    # Checkpointing
    checkpoint_dir: str | Path = "results/vae",
    save_every: int = 10,
    # Misc
    device: torch.device | str = "cpu",
    seed: int = 48025845,
    trial=None
) -> dict[str, Any]:
    """Train the VAE and return history.

    Args:
        model:              ConvVAE instance.
        train_loader:       Training DataLoader.
        val_loader:         Validation DataLoader.
        lr:                 Peak learning rate for AdamW.
        weight_decay:       L2 regularisation coefficient.
        epochs:             Number of training epochs.
        grad_clip:          Gradient clipping max norm (0 = disabled).
        kl_warmup_epochs:   Epochs over which KL weight linearly ramps from 0→1.
        kl_max_weight:      Maximum KL weight (set <1 for β-VAE).
        use_cosine_lr:      Whether to use cosine LR annealing.
        checkpoint_dir:     Where to save model checkpoints.
        save_every:         Save a periodic checkpoint every N epochs.
        device:             Torch device.
        seed:               Seed for reproducibility (sets torch manual seed).

    Returns:
        history dict with lists of per-epoch metrics.
    """
    seed_all(seed)
    device = torch.device(device)
    model = model.to(device)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
        if use_cosine_lr
        else None
    )

    history: dict[str, list] = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss":   [], "val_recon":   [], "val_kl":   [],
        "kl_weight":  [], "lr":          [],
    }

    best_val_loss = math.inf
    best_epoch = -1

    log.info(
        "Starting training — device: %s | epochs: %d | latent_dim: %d | lr: %.1e",
        device, epochs, model.cfg.latent_dim, lr,
    )

    epoch_times: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        kl_w = kl_weight(epoch, kl_warmup_epochs, kl_max_weight)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Train --------------------------------------------------------
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, kl_w, grad_clip
        )

        # ---- Validate -----------------------------------------------------
        val_metrics = _run_epoch(
            model, val_loader, None, device, kl_w, grad_clip
        )

        if scheduler is not None:
            scheduler.step()

        # ---- timing -------------------------------------------------------
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        remaining_seconds = remaining_epochs * avg_epoch_time

        eta_dt = datetime.now() + timedelta(seconds=remaining_seconds)

        # ---- Record -------------------------------------------------------
        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["loss_recon"])
        history["train_kl"].append(train_metrics["loss_kl"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["loss_recon"])
        history["val_kl"].append(val_metrics["loss_kl"])
        history["kl_weight"].append(kl_w)
        history["lr"].append(current_lr)

        log.info(
            "Epoch %3d/%d | β=%.3f | "
            "train loss=%.2f (recon=%.2f kl=%.2f) | "
            "val loss=%.2f (recon=%.2f kl=%.2f) | "
            "epoch_time=%.1fs | eta=%s",
            epoch, epochs, kl_w,
            train_metrics["loss"], train_metrics["loss_recon"], train_metrics["loss_kl"],
            val_metrics["loss"], val_metrics["loss_recon"], val_metrics["loss_kl"],
            epoch_time,
            fmt_12h(eta_dt),
        )

        # ---- Checkpointing ------------------------------------------------
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            _save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_dir / "best.pt")
            log.info("  ↳ New best val loss %.4f at epoch %d", best_val_loss, epoch)

        if save_every > 0 and epoch % save_every == 0:
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    log.info(
        "Training complete. Best val loss %.4f at epoch %d.", best_val_loss, best_epoch
    )
    return history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model.cfg,
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict:
    """Load a checkpoint into model (in-place) and return the saved dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded checkpoint from '%s' (epoch %d)", path, ckpt["epoch"])
    return ckpt
