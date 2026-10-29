"""dSprites dataset loading and preprocessing.

dSprites is a dataset of 2D shapes with known generative factors.

Images:
- shape: square / ellipse / heart
- size: 64x64
- values: binary {0, 1}
- total: ~737k

Reference:
https://github.com/google-deepmind/dsprites-dataset
"""

import logging
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


log = logging.getLogger(__name__)

DSPRITES_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)

FACTOR_NAMES = ["color", "shape", "scale", "orientation", "posX", "posY"]
FACTOR_SIZES = [1, 3, 6, 40, 32, 32]


class DSpritesDataset(Dataset):
    """Memory-efficient dSprites dataset.

    Optimisations:
    - use mmap_mode="r" to avoid full eager RAM load
    - avoid float conversion at init time
    - avoid tensor materialisation for labels unless requested
    - convert only per-sample inside __getitem__
    """

    def __init__(
        self,
        npz_path: str | Path,
        include_labels: bool = False,
    ) -> None:
        self.path = Path(npz_path)
        self.include_labels = include_labels

        if not self.path.exists():
            raise FileNotFoundError(
                f"dSprites file not found at '{self.path}'.\n"
                f"Download with:\n"
                f"python -m src.data.dsprites --download"
            )

        log.info("Loading dSprites from %s", self.path)

        data = np.load(
            self.path,
            mmap_mode="r",
            allow_pickle=False,
        )

        # keep numpy memmap, do not eagerly convert
        self.images = data["imgs"]  # (N, 64, 64), uint8

        if include_labels:
            self.labels = data["latents_values"]   # float
            self.classes = data["latents_classes"] # int

        self.n = len(self.images)

        log.info(
            "Loaded %d images | shape=%s",
            self.n,
            self.images.shape[1:],
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # convert only requested sample
        img = torch.from_numpy(
            self.images[idx]
        ).unsqueeze(0).float()

        if not self.include_labels:
            return img

        label = torch.from_numpy(
            self.labels[idx]
        ).float()

        cls = torch.from_numpy(
            self.classes[idx]
        ).long()

        return img, label, cls


def get_dataloaders(
    npz_path: str | Path,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers: int = 4,
    seed: int = 48025845,
    include_labels: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test loaders."""

    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be < 1.0"
        )

    dataset = DSpritesDataset(
        npz_path=npz_path,
        include_labels=include_labels,
    )

    n_total = len(dataset)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
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

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **common,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **common,
    )

    log.info(
        "Split sizes | train=%d val=%d test=%d",
        n_train,
        n_val,
        n_test,
    )

    return train_loader, val_loader, test_loader


def download_dataset(data_dir: str | Path) -> None:
    """Download dSprites if missing."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    out = data_dir / "dsprites.npz"

    if out.exists():
        print(f"Already exists: {out}")
        return

    print(f"Downloading dSprites to {out} ...")
    urllib.request.urlretrieve(DSPRITES_URL, out)
    print("Done.")

def get_concept_loaders(
    npz_path: str | Path,
    concept_col: int = 1,
    concept_val: int = 2,
    batch_size: int = 256,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 48025845,
    split: str = "train",
) -> tuple[DataLoader, DataLoader]:
    """
    Return (forget_loader, retain_loader) drawn from one split.

    Reproduces the same train/val/test split as get_dataloaders, so the three
    splits never overlap regardless of which one is requested.

    concept_col: column index in latents_classes (1 = shape)
    concept_val: class value to forget (2 = heart)
    split:       "train" (default, used for unlearning) | "val" | "test".
                 Use "test" to score an unlearned model on HELD-OUT data --
                 scoring on "train" measures fit to the data just trained on.

    For split != "train" the loaders are deterministic (shuffle=False,
    drop_last=False) so every sample is counted exactly once.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    dataset = DSpritesDataset(npz_path, include_labels=True)

    n_total = len(dataset)
    n_test  = int(n_total * test_fraction)
    n_val   = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    chosen = {"train": train_ds, "val": val_ds, "test": test_ds}[split]

    split_idx    = np.array(chosen.indices)
    shape_labels = dataset.classes[split_idx, concept_col]

    forget_idx = split_idx[shape_labels == concept_val].tolist()
    retain_idx = split_idx[shape_labels != concept_val].tolist()

    is_train = split == "train"
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    forget_loader = DataLoader(Subset(dataset, forget_idx),
                               shuffle=is_train, drop_last=is_train, **common)
    retain_loader = DataLoader(Subset(dataset, retain_idx),
                               shuffle=is_train, drop_last=is_train, **common)

    log.info("Concept loaders [%s] | forget=%d  retain=%d",
             split, len(forget_idx), len(retain_idx))
    return forget_loader, retain_loader


def get_retain_dataloaders(
    npz_path: str | Path,
    concept_col: int = 1,
    concept_val: int = 2,
    batch_size: int = 256,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 48025845,
    include_labels: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test loaders where train and val are filtered to
    the retain set (all samples except concept_val).

    Uses the exact same random split as get_dataloaders so retain indices
    never overlap with the test set used for evaluation.
    """
    dataset = DSpritesDataset(npz_path, include_labels=include_labels)

    n_total = len(dataset)
    n_test  = int(n_total * test_fraction)
    n_val   = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Load class labels from npz directly so filtering works regardless of include_labels
    classes = np.load(npz_path, mmap_mode="r", allow_pickle=False)["latents_classes"]

    train_idx = np.array(train_ds.indices)
    val_idx   = np.array(val_ds.indices)
    test_idx  = np.array(test_ds.indices)

    retain_train_idx = train_idx[classes[train_idx, concept_col] != concept_val].tolist()
    retain_val_idx   = val_idx[classes[val_idx,   concept_col] != concept_val].tolist()

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(Subset(dataset, retain_train_idx), shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(Subset(dataset, retain_val_idx),   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(Subset(dataset, test_idx),         shuffle=False, drop_last=False, **common)

    log.info(
        "Retain split | train=%d  val=%d  test=%d",
        len(retain_train_idx), len(retain_val_idx), len(test_idx),
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download dSprites dataset"
    )
    parser.add_argument(
        "--download",
        action="store_true",
    )
    parser.add_argument(
        "--data-dir",
        default="data/dsprites",
        type=str,
    )

    args = parser.parse_args()

    if args.download:
        download_dataset(args.data_dir)