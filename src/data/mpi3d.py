"""MPI3D-real dataset loading and preprocessing.

MPI3D-real is a dataset of REAL photographs of a physical robot platform holding
objects, recorded across seven fully-known, independent generative factors. It
is the "real world, disentangled" corner of the design space that dSprites
(synthetic, disentangled) and CelebA (real, entangled) leave empty: it lets a
finding be attributed to disentanglement or to realism, not to both at once.

Images:
- RGB, 64x64
- 1,036,800 images = product of the factor sizes below
- stored uncompressed in the .npz under key "images", uint8

Factor layout (Gondal et al. 2019). The flat image index unravels, in C order,
against FACTOR_SIZES; i.e. array.reshape([6,6,2,3,3,40,40,64,64,3]) recovers the
factors on the first seven axes.

    factor 0  object_color       6  white green red blue brown olive
    factor 1  object_shape       6  cone cube cylinder hexagonal pyramid sphere
    factor 2  object_size        2  small large
    factor 3  camera_height      3  top center bottom
    factor 4  background_color   3  purple sea-green salmon
    factor 5  horizontal_axis   40
    factor 6  vertical_axis     40

FORGET CONCEPT for this project: object_color (factor 0), value "red" (index 2).
This is a different concept TYPE from the other two datasets -- a low-level,
spatially-global attribute rather than a shape class (dSprites) or a localized
object (CelebA eyeglasses) -- which is deliberate: it broadens the range of
concept kinds the evaluation is tested against.

Preprocessing mirrors CelebA, not dSprites: these are natural RGB images, so the
VAE uses an MSE reconstruction loss with a Tanh decoder, and images are
normalised to [-1, 1]. (dSprites is binary and uses BCE.)

CONCEPT LABEL CONVENTION. The labeled path yields (image, binary) where
``binary = 1`` iff the forget factor equals the forget value. This matches
CelebA's (image, 0/1) loader, so ``encode_dataset`` reads the label straight from
``batch[1]`` and every downstream metric is called with ``forget_val=1`` -- as
opposed to dSprites, whose 3-tuple batch is read at the hardcoded shape column.

Reference:
Gondal et al. (2019), "On the Transfer of Inductive Bias from Simulation to the
Real World: A New Disentanglement Dataset", NeurIPS 2019.
https://github.com/rr-learning/disentanglement_dataset
"""

import logging
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

log = logging.getLogger(__name__)

# Official mirror (the original Google Cloud bucket now 403s; the rr-learning
# repo README points here).
MPI3D_REAL_URL = (
    "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_real.npz"
)

FACTOR_NAMES = [
    "object_color", "object_shape", "object_size",
    "camera_height", "background_color", "horizontal_axis", "vertical_axis",
]
FACTOR_SIZES = [6, 6, 2, 3, 3, 40, 40]
COLOR_NAMES = ["white", "green", "red", "blue", "brown", "olive"]
SHAPE_NAMES = ["cone", "cube", "cylinder", "hexagonal", "pyramid", "sphere"]

DEFAULT_FORGET_FACTOR = 0   # object_color
DEFAULT_FORGET_VALUE = 2    # red

SEED = 48025845


def memmap_path(npz_path: str | Path) -> Path:
    """Sibling flat .npy path used for true memory-mapping (see MPI3DDataset)."""
    return Path(npz_path).with_name("mpi3d_real_images.npy")


def ensure_memmap(npz_path: str | Path) -> Path:
    """Create the memory-mappable .npy once from the .npz, if missing.

    Reads the 12.7 GB image array (loads it into RAM briefly) and writes it back
    as a flat .npy. That .npy -- unlike the .npz -- is genuinely memory-mapped by
    np.load(mmap_mode="r"), so every later process opens it instantly with no
    resident-memory cost. Idempotent.
    """
    npy = memmap_path(npz_path)
    if npy.exists():
        log.info("Memmap already present: %s", npy)
        return npy
    log.info("Converting %s -> %s (one-time, ~12.7 GB) ...", npz_path, npy)
    data = np.load(npz_path, allow_pickle=False)
    key = "images" if "images" in data.files else data.files[0]
    arr = data[key]
    # np.save writes a .npy header + the raw uint8 block, which is mmap-able.
    np.save(npy, arr)
    log.info("Wrote memmap %s (%.1f GB)", npy, npy.stat().st_size / 1e9)
    return npy


def factor_values(n: int, forget_factor: int) -> np.ndarray:
    """The value of ``forget_factor`` for every flat index in ``[0, n)``.

    The images are stored in C order against FACTOR_SIZES, so unravelling the
    index recovers each factor. Vectorised: one int64 array, ~8 MB for the full
    set, computed once.
    """
    return np.unravel_index(np.arange(n), FACTOR_SIZES)[forget_factor].astype(np.int64)


class MPI3DDataset(Dataset):
    """Memory-efficient MPI3D-real dataset.

    Mirrors the dSprites loader's memmap strategy: the 12.7 GB image array is
    never eagerly materialised, only the requested sample is converted per
    ``__getitem__``. Factor values are derived from the index, so no label array
    is stored on disk.

    When ``include_labels`` is set, ``__getitem__`` returns ``(image, binary)``
    where ``binary = 1`` iff the forget factor equals ``forget_val`` -- the
    CelebA-style 2-tuple that ``encode_dataset`` reads directly.
    """

    def __init__(
        self,
        npz_path: str | Path,
        include_labels: bool = False,
        forget_factor: int = DEFAULT_FORGET_FACTOR,
        forget_val: int = DEFAULT_FORGET_VALUE,
    ) -> None:
        self.path = Path(npz_path)
        self.include_labels = include_labels
        self.forget_factor = forget_factor
        self.forget_val = forget_val

        if not self.path.exists():
            raise FileNotFoundError(
                f"MPI3D-real file not found at '{self.path}'.\n"
                f"Download with:\n"
                f"python -m src.data.mpi3d --download"
            )

        # We prefer a sibling flat .npy (created once via ensure_memmap), which
        # np.load(mmap_mode="r") genuinely memory-maps -- unlike the .npz, which
        # loads all 12.7 GB into RAM. Create it with
        # `python -m src.data.mpi3d --convert`.
        self._npy = memmap_path(self.path)
        self._use_npy = self._npy.exists()
        if not self._use_npy:
            log.warning("Memmap %s not found -- will load the full .npz into RAM "
                        "(12.7 GB). Run `python -m src.data.mpi3d --convert`.", self._npy)

        # LAZY per-process open. self._images is NOT set here and is dropped from
        # the pickle (see __getstate__): on Windows a DataLoader with
        # num_workers>0 SPAWNS workers by pickling the whole Dataset, and pickling
        # a 12.7 GB np.memmap raises OSError [Errno 22]. Storing only the path and
        # opening the memmap on first access *inside each process* means the
        # pickle carries a few strings, and every worker mmaps the shared file
        # itself -- near-zero resident memory, no serialization of the array.
        self._images = None
        self.n = len(self._open_images())   # one cheap open to learn the length

        if include_labels:
            self.concept = (
                factor_values(self.n, forget_factor) == forget_val
            ).astype(np.int64)

        log.info("Loaded %d images | shape=%s | memmap=%s",
                 self.n, self._open_images().shape[1:], self._use_npy)

    def _open_images(self):
        """Open (once per process) and return the image array.

        Cached in self._images, which __getstate__ excludes so each spawned
        worker reopens its own handle to the shared memmap file.
        """
        if self._images is None:
            if self._use_npy:
                self._images = np.load(self._npy, mmap_mode="r", allow_pickle=False)
            else:
                data = np.load(self.path, mmap_mode="r", allow_pickle=False)
                key = "images" if "images" in data.files else data.files[0]
                self._images = data[key]
        return self._images

    def __getstate__(self):
        # Do not pickle the open memmap handle (unpicklable at 12.7 GB); workers
        # reopen it lazily via _open_images().
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # copy() because the memmap slice is read-only and torch would warn;
        # HWC uint8 -> CHW float in [-1, 1] (MSE/Tanh, matching CelebA).
        arr = np.asarray(self._open_images()[idx], dtype=np.float32).copy()
        img = torch.from_numpy(arr).permute(2, 0, 1).div_(255.0).mul_(2.0).sub_(1.0)
        if not self.include_labels:
            return img
        return img, torch.tensor(int(self.concept[idx]), dtype=torch.long)


def _common(batch_size, num_workers):
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _splits(n_total, val_fraction, test_fraction, seed):
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    return random_split(range(n_total), [n_train, n_val, n_test], generator=gen)


def get_dataloaders(
    npz_path: str | Path,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers: int = 4,
    seed: int = SEED,
    include_labels: bool = False,
    forget_factor: int = DEFAULT_FORGET_FACTOR,
    forget_val: int = DEFAULT_FORGET_VALUE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test loaders.

    ``include_labels`` attaches the binary concept label to every batch; the
    test loader with labels is the encode-and-score loader (mirrors
    ``dsprites.get_dataloaders(include_labels=True)`` used by the warmers).
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    dataset = MPI3DDataset(npz_path, include_labels=include_labels,
                           forget_factor=forget_factor, forget_val=forget_val)
    train_idx, val_idx, test_idx = _splits(len(dataset), val_fraction, test_fraction, seed)

    common = _common(batch_size, num_workers)
    train_loader = DataLoader(Subset(dataset, list(train_idx)), shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(Subset(dataset, list(val_idx)),   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(Subset(dataset, list(test_idx)),  shuffle=False, drop_last=False, **common)

    log.info("MPI3D split sizes | train=%d val=%d test=%d",
             len(train_idx), len(val_idx), len(test_idx))
    return train_loader, val_loader, test_loader


def get_concept_loaders(
    npz_path: str | Path,
    forget_factor: int = DEFAULT_FORGET_FACTOR,
    forget_val: int = DEFAULT_FORGET_VALUE,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = SEED,
    split: str = "train",
) -> tuple[DataLoader, DataLoader]:
    """Return (forget_loader, retain_loader) drawn from one split.

    Reproduces the same train/val/test split as ``get_dataloaders`` so the three
    never overlap regardless of which is requested. Mirrors the dSprites
    interface exactly, including the ``split`` argument:

    * ``split="train"`` -- what unlearning trains on (shuffled, drop_last).
    * ``split="test"``  -- held-out scoring (deterministic, every sample once).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    dataset = MPI3DDataset(npz_path, include_labels=False)
    train_idx, val_idx, test_idx = _splits(len(dataset), val_fraction, test_fraction, seed)
    chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]

    split_idx = np.array(list(chosen))
    color = factor_values(len(dataset), forget_factor)[split_idx]

    forget_idx = split_idx[color == forget_val].tolist()
    retain_idx = split_idx[color != forget_val].tolist()

    is_train = split == "train"
    common = _common(batch_size, num_workers)
    forget_loader = DataLoader(Subset(dataset, forget_idx),
                               shuffle=is_train, drop_last=is_train, **common)
    retain_loader = DataLoader(Subset(dataset, retain_idx),
                               shuffle=is_train, drop_last=is_train, **common)

    log.info("MPI3D concept loaders [%s] | forget=%d retain=%d",
             split, len(forget_idx), len(retain_idx))
    return forget_loader, retain_loader


def get_retain_dataloaders(
    npz_path: str | Path,
    forget_factor: int = DEFAULT_FORGET_FACTOR,
    forget_val: int = DEFAULT_FORGET_VALUE,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = SEED,
    include_labels: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build retain train / retain val / (full) test loaders.

    train and val are filtered to the retain set (forget factor != forget_val);
    test is the full test split, used for held-out evaluation. Uses the same
    random split as ``get_dataloaders`` so retain indices never leak into test.
    Mirrors ``dsprites.get_retain_dataloaders``.
    """
    dataset = MPI3DDataset(npz_path, include_labels=include_labels,
                           forget_factor=forget_factor, forget_val=forget_val)
    train_idx, val_idx, test_idx = _splits(len(dataset), val_fraction, test_fraction, seed)
    color = factor_values(len(dataset), forget_factor)

    train_arr = np.array(list(train_idx))
    val_arr = np.array(list(val_idx))
    retain_train = train_arr[color[train_arr] != forget_val].tolist()
    retain_val = val_arr[color[val_arr] != forget_val].tolist()

    common = _common(batch_size, num_workers)
    train_loader = DataLoader(Subset(dataset, retain_train), shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(Subset(dataset, retain_val),   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(Subset(dataset, list(test_idx)), shuffle=False, drop_last=False, **common)

    log.info("MPI3D retain split | train=%d val=%d test=%d",
             len(retain_train), len(retain_val), len(list(test_idx)))
    return train_loader, val_loader, test_loader


def get_labeled_dataloader(
    npz_path: str | Path,
    forget_factor: int = DEFAULT_FORGET_FACTOR,
    forget_val: int = DEFAULT_FORGET_VALUE,
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = SEED,
) -> DataLoader:
    """Single (image, 0/1) loader over the TEST split, for encode-and-probe.

    The test split (~10%, ~104k images) is used rather than the full 1.04 M so
    per-checkpoint encoding stays comparable to the other datasets; the forget
    class is 1/6 of images so a test split still holds ~17k positives, ample for
    probing and kNN. Deterministic so the score is reproducible.
    """
    _, _, test_loader = get_dataloaders(
        npz_path=npz_path, batch_size=batch_size, val_fraction=val_fraction,
        test_fraction=test_fraction, num_workers=num_workers, seed=seed,
        include_labels=True, forget_factor=forget_factor, forget_val=forget_val,
    )
    return test_loader


def load_factor_matrix(npz_path: str | Path, indices: np.ndarray | None = None
                       ) -> dict[str, np.ndarray]:
    """Ground-truth factor values, as {factor_name: (N,) int array}.

    Used only by the disentanglement evaluation (MIG/DCI), which -- unlike the
    binary concept metrics -- needs every factor. MPI3D-real supplies these
    where CelebA cannot, giving the disentanglement metrics their first
    real-world data point. If ``indices`` is given, factors are returned for
    that subset (e.g. the test split the encodings were computed on).
    """
    # Prefer the memmap header (cheap) over the .npz (which would load 12.7 GB
    # just to read a length); fall back to the factor-size product, which is the
    # dataset's fixed cardinality.
    npy = memmap_path(npz_path)
    if npy.exists():
        n = len(np.load(npy, mmap_mode="r", allow_pickle=False))
    else:
        n = int(np.prod(FACTOR_SIZES))
    grid = np.unravel_index(np.arange(n), FACTOR_SIZES)
    if indices is not None:
        grid = [g[indices] for g in grid]
    return {name: g.astype(np.int64) for name, g in zip(FACTOR_NAMES, grid)}


def download_dataset(data_dir: str | Path) -> None:
    """Download MPI3D-real (~12.7 GB) if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "mpi3d_real.npz"
    if out.exists():
        print(f"Already exists: {out}")
        return
    print(f"Downloading MPI3D-real (~12.7 GB) to {out} ...")
    urllib.request.urlretrieve(MPI3D_REAL_URL, out)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MPI3D-real dataset utilities")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--convert", action="store_true",
                        help="Build the memory-mappable .npy from the .npz (one-time).")
    parser.add_argument("--data-dir", default="data/mpi3d", type=str)
    args = parser.parse_args()

    if args.download:
        download_dataset(args.data_dir)
    if args.convert:
        ensure_memmap(Path(args.data_dir) / "mpi3d_real.npz")
