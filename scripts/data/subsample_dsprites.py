# scripts/subsample_dsprites.py

import numpy as np
from pathlib import Path

SRC  = Path("data/dsprites/dsprites.npz")
DST  = Path("data/dsprites/dsprites_200k.npz")
N    = 200_000
SEED = 48025845

print(f"Loading {SRC}...")
data = np.load(SRC, allow_pickle=True, encoding="latin1")

imgs    = data["imgs"]            # (737280, 64, 64)
values  = data["latents_values"]  # (737280, 6) float — continuous factor values
classes = data["latents_classes"] # (737280, 6) int   — discrete class indices

total = len(imgs)
print(f"Total images: {total:,} — sampling {N:,}...")

rng = np.random.default_rng(SEED)
idx = rng.choice(total, size=N, replace=False)
idx.sort()

np.savez_compressed(
    DST,
    imgs            = imgs[idx],
    latents_values  = values[idx],
    latents_classes = classes[idx],   # was missing
)
print(f"Saved to {DST}")