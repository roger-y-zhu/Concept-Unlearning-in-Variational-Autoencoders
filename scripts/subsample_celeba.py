# scripts/subsample_celeba.py

from pathlib import Path
import shutil
import numpy as np


SRC_DIR = Path("data/celeba/img_align_celeba")
DST_DIR = Path("data/celeba/img_align_celeba_75k")

N = 75_000
SEED = 48025845


print(f"Loading image list from {SRC_DIR}...")

files = sorted([
    p for p in SRC_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

total = len(files)
print(f"Total images: {total:,} — sampling {N:,}...")

if N > total:
    raise ValueError(
        f"Requested {N:,} images but only found {total:,}"
    )

rng = np.random.default_rng(SEED)
idx = rng.choice(total, size=N, replace=False)
idx.sort()

selected = [files[i] for i in idx]

DST_DIR.mkdir(parents=True, exist_ok=True)

for i, src_path in enumerate(selected, 1):
    dst_path = DST_DIR / src_path.name
    shutil.copy2(src_path, dst_path)

    if i % 10_000 == 0:
        print(f"Copied {i:,}/{N:,} images...")

print(f"Saved subset to {DST_DIR}")
print(f"Final image count: {len(selected):,}")