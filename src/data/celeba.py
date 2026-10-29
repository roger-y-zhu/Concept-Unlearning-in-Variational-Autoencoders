"""CelebA dataset loading and preprocessing.

Supports:
- folder-based CelebA (e.g. img_align_celeba_75k)
- optional deterministic subset sampling

Images:
- RGB
- centre crop 178 -> resize -> normalize to [-1, 1]

Reference:
Liu et al. (2015)
"Deep Learning Face Attributes in the Wild"
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms

log = logging.getLogger(__name__)

VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def get_transform(image_size: int = 64) -> transforms.Compose:
    """Standard CelebA preprocessing."""
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


class ImageFolderDataset(Dataset):
    """Fast folder-based image dataset."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 64,
        subset: int | None = None,
        seed: int = 48025845,
    ) -> None:
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(f"CelebA folder not found: {self.root}")

        files = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
        )

        if not files:
            raise ValueError(f"No images found in {self.root}")

        # -------------------------
        # optional subset sampling
        # -------------------------
        if subset is not None:
            if subset <= 0 or subset > len(files):
                raise ValueError(
                    f"subset must be in [1, {len(files)}], got {subset}"
                )

            rng = np.random.RandomState(seed)
            idx = rng.choice(len(files), subset, replace=False)
            files = [files[i] for i in idx]

        self.files = files
        self.transform = get_transform(image_size)

        log.info(
            "CelebA dataset loaded | root=%s | images=%d | subset=%s",
            self.root,
            len(self.files),
            subset,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


def get_dataloaders(
    data_root: str,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,          # API compat
    include_labels: bool = False,    # API compat
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    subset: int | None = None,
    seed: int = 48025845,
):
    """Build train / val / test loaders."""

    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    dataset = ImageFolderDataset(
        root=data_root,
        image_size=image_size,
        subset=subset,
        seed=seed,
    )

    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_test = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    log.info(
        "CelebA split sizes | train=%d val=%d test=%d",
        n_train, n_val, n_test
    )

    return train_loader, val_loader, test_loader

CELEBA_ATTR_FILE = "list_attr_celeba.txt"
EYEGLASSES_COL = "Eyeglasses"

class LabeledCelebADataset(Dataset):
    """CelebA images with a single binary attribute label."""

    def __init__(self, root: str | Path, attr_name: str = EYEGLASSES_COL, image_size: int = 64) -> None:
        self.root = Path(root)
        self.transform = get_transform(image_size)

        attr_path = self.root.parent / CELEBA_ATTR_FILE
        if not attr_path.exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_path}")

        # parse header
        with open(attr_path) as f:
            n_images = int(f.readline().strip())
            attr_names = f.readline().strip().split()
            if attr_name not in attr_names:
                raise ValueError(f"Attribute '{attr_name}' not in {attr_names}")
            col = attr_names.index(attr_name)

            rows = []
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                label = 1 if int(parts[1 + col]) == 1 else 0
                rows.append((filename, label))

        # only keep images that exist in root
        existing = {p.name for p in self.root.iterdir() if p.suffix.lower() in VALID_SUFFIXES}
        self.samples = [(fn, lbl) for fn, lbl in rows if fn in existing]

        log.info("LabeledCelebA | images=%d | attr=%s | positives=%d",
                 len(self.samples), attr_name,
                 sum(lbl for _, lbl in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filename, label = self.samples[idx]
        with Image.open(self.root / filename) as img:
            img = img.convert("RGB")
            return self.transform(img), torch.tensor(label, dtype=torch.long)


def get_labeled_dataloader(
    data_root: str,
    attr_name: str = EYEGLASSES_COL,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """Single dataloader returning (image, label) for a CelebA attribute."""
    dataset = LabeledCelebADataset(data_root, attr_name, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Retain split — filters out images where a concept attribute is present
# ---------------------------------------------------------------------------

class _AttrFilteredDataset(Dataset):
    """Wraps a folder of CelebA images, filtering by a binary attribute.

    Args:
        root:       Path to the image folder (img_align_celeba).
        attr_name:  Attribute to filter on (default: "Eyeglasses").
        keep_val:   Keep images where attribute == keep_val (+1 present, -1 absent).
                    Default -1 keeps the *retain* set (Eyeglasses absent).
        image_size: Spatial resolution after centre-crop + resize.
    """

    def __init__(
        self,
        root: str | Path,
        attr_name: str = EYEGLASSES_COL,
        keep_val: int = -1,
        image_size: int = 64,
    ) -> None:
        self.root = Path(root)
        self.transform = get_transform(image_size)

        attr_path = self.root.parent / CELEBA_ATTR_FILE
        if not attr_path.exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_path}")

        with open(attr_path) as f:
            _ = f.readline()                    # n_images line
            attr_names = f.readline().strip().split()
            if attr_name not in attr_names:
                raise ValueError(f"Attribute '{attr_name}' not in list: {attr_names}")
            col = attr_names.index(attr_name)

            self.files: list[Path] = []
            for line in f:
                parts = line.strip().split()
                filename = parts[0]
                val = int(parts[1 + col])
                if val == keep_val:
                    p = self.root / filename
                    if p.exists():
                        self.files.append(p)

        log.info(
            "_AttrFilteredDataset | attr=%s | keep_val=%+d | images=%d",
            attr_name, keep_val, len(self.files),
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.files[idx]) as img:
            return self.transform(img.convert("RGB"))


def get_concept_loaders(
    data_root: str,
    forget_attr: str = EYEGLASSES_COL,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 48025845,
) -> tuple[DataLoader, DataLoader]:
    """Return (forget_loader, retain_train_loader) for unlearning.

    forget_loader : images where ``forget_attr`` == +1  (concept present)
    retain_loader : training portion of images where ``forget_attr`` == -1

    The val split of the retain set is not returned here; obtain it separately
    via ``get_retain_dataloaders`` for use as the validation loader.
    Mirrors the dSprites ``get_concept_loaders`` interface.
    """
    forget_ds = _AttrFilteredDataset(
        root=data_root, attr_name=forget_attr, keep_val=+1, image_size=image_size,
    )
    retain_full = _AttrFilteredDataset(
        root=data_root, attr_name=forget_attr, keep_val=-1, image_size=image_size,
    )

    n_val   = int(len(retain_full) * val_fraction)
    n_train = len(retain_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    retain_train_ds, _ = random_split(retain_full, [n_train, n_val], generator=generator)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    forget_loader = DataLoader(forget_ds,       shuffle=True, drop_last=True, **common)
    retain_loader = DataLoader(retain_train_ds, shuffle=True, drop_last=True, **common)

    log.info(
        "CelebA concept loaders | attr=%s | forget=%d  retain_train=%d",
        forget_attr, len(forget_ds), n_train,
    )
    return forget_loader, retain_loader


def get_concept_eval_loaders(
    data_root: str,
    forget_attr: str = EYEGLASSES_COL,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 48025845,
) -> tuple[DataLoader, DataLoader]:
    """Return (forget_loader, retain_val_loader) for SCORING an unlearned model.

    Differs from ``get_concept_loaders`` in two ways:

    * The retain loader is the **held-out val split**, not the train split, so
      retain ELBO is not measured on the data the model just fitted.
    * Both loaders are deterministic (shuffle=False, drop_last=False) so every
      sample is counted exactly once and the score is reproducible.

    The forget loader still covers the full forget set: "did the model forget
    the images it was asked to forget" is a question about the training forget
    set, so there is nothing to hold out there.

    The split matches ``get_concept_loaders`` exactly (same seed, same two-way
    split of the retain pool), so train and val never overlap.
    """
    forget_ds = _AttrFilteredDataset(
        root=data_root, attr_name=forget_attr, keep_val=+1, image_size=image_size,
    )
    retain_full = _AttrFilteredDataset(
        root=data_root, attr_name=forget_attr, keep_val=-1, image_size=image_size,
    )

    n_val   = int(len(retain_full) * val_fraction)
    n_train = len(retain_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    _, retain_val_ds = random_split(retain_full, [n_train, n_val], generator=generator)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    forget_loader = DataLoader(forget_ds,     shuffle=False, drop_last=False, **common)
    retain_loader = DataLoader(retain_val_ds, shuffle=False, drop_last=False, **common)

    log.info(
        "CelebA concept EVAL loaders | attr=%s | forget=%d  retain_val=%d",
        forget_attr, len(forget_ds), n_val,
    )
    return forget_loader, retain_loader


def get_retain_dataloaders(
    data_root: str,
    forget_attr: str = EYEGLASSES_COL,
    image_size: int = 64,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 48025845,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build retain train / val / (optionally empty) test loaders for CelebA.

    The retain set consists of images where ``forget_attr`` is *absent*
    (attribute value == -1 in CelebA's +1/-1 encoding).  This mirrors the
    dSprites ``get_retain_dataloaders`` interface used in ``experiments/unlearn.py``.

    Returns:
        (retain_train_loader, retain_val_loader, retain_test_loader)
        retain_test_loader will have 0 samples when test_fraction == 0.
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    dataset = _AttrFilteredDataset(
        root=data_root,
        attr_name=forget_attr,
        keep_val=-1,          # keep images WITHOUT the forget concept
        image_size=image_size,
    )

    n_total = len(dataset)
    n_val   = int(n_total * val_fraction)
    n_test  = int(n_total * test_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common)

    log.info(
        "CelebA retain split | attr=%s | train=%d val=%d test=%d",
        forget_attr, n_train, n_val, n_test,
    )

    return train_loader, val_loader, test_loader