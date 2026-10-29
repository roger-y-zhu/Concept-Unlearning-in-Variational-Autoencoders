"""MO4 figure: geometry-vs-recoverability dissociation on MPI3D.

The oracle never saw red objects, so it cannot RECONSTRUCT them (it renders red as
brown/olive), yet is-red stays perfectly linearly DECODABLE from its latents
(AUROC 1.0, ledger M5). This figure shows the reconstruction half of that split:
red test images decoded by the baseline (faithful red) vs the oracle (colour
shifted away from red), with a retain (non-red) row as a control both reproduce.

Plain matplotlib, no styling frills (per user preference). ->
results/figures/mpi3d/oracle_geometry_vs_recoverability.png
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402
from src.data.mpi3d import get_concept_loaders  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ = str(ROOT / "data/mpi3d/mpi3d_real.npz")
N = 8  # images per class

baseline, _ = load_vae_checkpoint(ROOT / "results/mpi3d/mpi3d_baseline/epoch_0120.pt", DEVICE)
oracle,   _ = load_vae_checkpoint(ROOT / "results/mpi3d/mpi3d_oracle/epoch_0120.pt", DEVICE)


def to_img(t):  # (C,H,W) in [-1,1] tanh space -> (H,W,C) in [0,1]
    return np.clip((t.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)


@torch.no_grad()
def recon(model, x):
    mu, _ = model.encode(x)
    return model.decode(mu)


forget_loader, retain_loader = get_concept_loaders(
    npz_path=NPZ, batch_size=64, num_workers=0, split="test")
red = next(iter(forget_loader))[:N].to(DEVICE)       # red objects (forget)
ret = next(iter(retain_loader))[:N].to(DEVICE)       # non-red (retain control)

red_b, red_o = recon(baseline, red), recon(oracle, red)
ret_b, ret_o = recon(baseline, ret), recon(oracle, ret)

rows = [("red: original", red), ("red: baseline recon", red_b), ("red: ORACLE recon", red_o),
        ("retain: original", ret), ("retain: baseline recon", ret_b), ("retain: ORACLE recon", ret_o)]

fig, axes = plt.subplots(len(rows), N, figsize=(N * 1.15, len(rows) * 1.25))
for r, (label, batch) in enumerate(rows):
    for c in range(N):
        ax = axes[r, c]
        ax.imshow(to_img(batch[c])); ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
fig.suptitle("MPI3D geometry-vs-recoverability: the oracle cannot reconstruct red "
             "(renders it brown/olive)\nyet is-red is perfectly linearly decodable from "
             "its latents (AUROC 1.0). Retain objects both reproduce.", fontsize=9)
fig.tight_layout(rect=[0.06, 0, 1, 0.95])
out = ROOT / "results/figures/mpi3d/oracle_geometry_vs_recoverability.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"-> {out.relative_to(ROOT)}")
