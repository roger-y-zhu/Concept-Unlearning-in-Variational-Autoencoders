"""experiments/train_dsprites.py

Entry point for training the baseline convolutional VAE on dSprites.

Usage
-----
# Standard run (uses configs/base_vae.yaml + configs/datasets/dsprites.yaml):
    python experiments/train_baseline.py

# Override any config value via CLI:
    python experiments/train_baseline.py \
        --model-config configs/base_vae.yaml \
        --data-config configs/datasets/dsprites.yaml \
        --latent-dim 16 \
        --epochs 50 \
        --device cuda

# Download dSprites first if you haven't already:
    python -m src.data.dsprites --download --data-dir data/dsprites
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# Make sure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dsprites import get_dataloaders
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import train

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(model_cfg: dict, data_cfg: dict, args: argparse.Namespace) -> dict:
    """Merge YAML configs with any CLI overrides."""
    cfg = {**model_cfg, **data_cfg}

    # CLI overrides — only apply if the user explicitly set them
    overrides = {
        "latent_dim":         args.latent_dim,
        "base_channels":      args.base_channels,
        "epochs":             args.epochs,
        "lr":                 args.lr,
        "kl_warmup_epochs":   args.kl_warmup_epochs,
        "kl_max_weight":      args.kl_max_weight,
        "batch_size":         args.batch_size,
        "device":             args.device,
        "seed":               args.seed,
    }
    for key, val in overrides.items():
        if val is not None:
            # Patch into the appropriate nested dict
            for section in ("model", "training", "dataloader", "dataset"):
                if section in cfg and key in cfg[section]:
                    cfg[section][key] = val
                    break
            else:
                cfg[key] = val

    return cfg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline VAE on dSprites.")

    # Config files
    p.add_argument("--model-config", default="configs/base_vae.yaml",
                   help="Path to model YAML config.")
    p.add_argument("--data-config",  default="configs/datasets/dsprites.yaml",
                   help="Path to dataset YAML config.")

    # Model overrides
    p.add_argument("--latent-dim",    type=int,   default=None)
    p.add_argument("--base-channels", type=int,   default=None)

    # Training overrides
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--kl-warmup-epochs", type=int,   default=None)
    p.add_argument("--kl-max-weight",    type=float, default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--seed",             type=int,   default=None)

    # Device
    p.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:1', 'mps', …"
             " Defaults to cuda if available, else cpu.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load and merge configs
    model_cfg = load_yaml(args.model_config)
    data_cfg  = load_yaml(args.data_config)
    cfg = merge_configs(model_cfg, data_cfg, args)

    # 2. Resolve device
    if cfg.get("device") is None:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            cfg["device"] = "mps"
    device = torch.device(cfg["device"])
    log.info("Using device: %s", device)

    # 3. Print resolved config
    log.info("Resolved config:\n%s", json.dumps(cfg, indent=2, default=str))

    # 4. Data
    ds_cfg     = cfg["dataset"]
    dl_cfg     = cfg["dataloader"]
    train_loader, val_loader, _ = get_dataloaders(
        npz_path        = ds_cfg["npz_path"],
        batch_size      = dl_cfg["batch_size"],
        val_fraction    = ds_cfg["val_fraction"],
        test_fraction   = ds_cfg["test_fraction"],
        num_workers     = dl_cfg["num_workers"],
        seed            = cfg["training"]["seed"],
        include_labels  = dl_cfg.get("include_labels", False),
    )

    # 5. Model
    m_cfg = cfg["model"]
    vae_config = VAEConfig(
        latent_dim    = m_cfg["latent_dim"],
        base_channels = m_cfg["base_channels"],
        eps           = m_cfg["eps"],
    )
    model = ConvVAE(vae_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # 6. Train
    t_cfg = cfg["training"]
    results_dir = cfg["results_dir"]

    history = train(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        lr               = t_cfg["lr"],
        weight_decay     = t_cfg["weight_decay"],
        epochs           = t_cfg["epochs"],
        grad_clip        = t_cfg["grad_clip"],
        kl_warmup_epochs = t_cfg["kl_warmup_epochs"],
        kl_max_weight    = t_cfg["kl_max_weight"],
        use_cosine_lr    = t_cfg["use_cosine_lr"],
        checkpoint_dir   = results_dir,
        save_every       = cfg["checkpointing"]["save_every"],
        device           = device,
        seed             = t_cfg["seed"],
    )

    # 7. Save history
    history_path = Path(results_dir) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved to %s", history_path)


if __name__ == "__main__":
    main()
