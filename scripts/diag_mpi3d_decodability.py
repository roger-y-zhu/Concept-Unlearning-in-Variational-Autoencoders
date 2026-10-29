"""Diagnostic: which MPI3D concepts does the BASELINE VAE make linearly decodable?

M5 showed is-red has linear-probe AUROC 1.0 (trivially separable global colour),
but M7's per-factor MIG showed object_shape is barely encoded (~0.003). Decodability
(can a linear probe read it?) and disentanglement (does one latent axis own it?) are
different questions. This measures one-vs-rest linear-probe AUROC for every value of
every factor from the frozen baseline latents, so a candidate SECOND forget concept
can be chosen on evidence: a concept that is NOT trivially decodable would test
whether the 'concept never leaves the latent space' story is colour-specific.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402
from src.data.mpi3d import (MPI3DDataset, _splits, load_factor_matrix,  # noqa: E402
                            FACTOR_NAMES, FACTOR_SIZES, COLOR_NAMES)
from torch.utils.data import DataLoader, Subset  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ = str(ROOT / "data/mpi3d/mpi3d_real.npz")
SEED = 48025845

model, _ = load_vae_checkpoint(ROOT / "results/mpi3d/mpi3d_baseline/epoch_0120.pt", DEVICE)
ds = MPI3DDataset(NPZ, include_labels=False)
_, _, test_sub = _splits(len(ds), 0.1, 0.1, SEED)
idx = np.array(list(test_sub))
# subsample for a fast probe (20k is ample for AUROC)
rng = np.random.default_rng(SEED)
sel = rng.choice(len(idx), size=20000, replace=False)
idx_s = idx[sel]
loader = DataLoader(Subset(ds, list(idx_s)), batch_size=256, shuffle=False, num_workers=0)


@torch.no_grad()
def encode(loader):
    mus = []
    for b in loader:
        x = b[0] if isinstance(b, (tuple, list)) else b
        mu, _ = model.encode(x.to(DEVICE))
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus)


mu = encode(loader)
factors = load_factor_matrix(NPZ, idx_s)
n = len(mu); ntr = int(0.7 * n)
tr, te = slice(0, ntr), slice(ntr, n)

print(f"baseline latents {mu.shape} | 20k test-split subsample")
print(f"{'factor':16} {'value':10} {'one-vs-rest AUROC':>18}  {'positive frac':>13}")
print("-" * 62)
for fi, fname in enumerate(FACTOR_NAMES):
    vals = factors[fname]
    # continuous-ish factors (40 values) -> skip per-value; only the small ones
    if FACTOR_SIZES[fi] > 6:
        continue
    for v in range(FACTOR_SIZES[fi]):
        y = (vals == v).astype(int)
        if y[tr].sum() < 10 or y[te].sum() < 10:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(mu[tr], y[tr])
        auroc = roc_auc_score(y[te], clf.decision_function(mu[te]))
        label = COLOR_NAMES[v] if fname == "object_color" and v < len(COLOR_NAMES) else str(v)
        print(f"{fname:16} {label:10} {auroc:>18.4f}  {y.mean():>13.3f}")
