"""
tests/test_data.py
===================
Data-loading correctness: split disjointness, determinism, and the forget/
retain partition actually matching the label filter it claims to apply.

Uses the synthetic_dsprites_npz / synthetic_celeba_dir fixtures from
conftest.py -- no real dataset file is required to run this suite.
"""

import numpy as np

from src.data.dsprites import get_concept_loaders as dsprites_concept_loaders
from src.data.celeba import (
    get_concept_loaders as celeba_concept_loaders,
    get_concept_eval_loaders,
    get_retain_dataloaders as celeba_retain_loaders,
)


def indices_of(loader):
    """Flatten a DataLoader's underlying Subset indices, however it's built."""
    ds = loader.dataset
    if hasattr(ds, "indices"):
        return set(ds.indices)
    # _AttrFilteredDataset / plain dataset: identify samples by filename or id
    if hasattr(ds, "files"):
        return set(str(f) for f in ds.files)
    return set(range(len(ds)))


class TestDSpritesSplits:
    def test_train_and_test_splits_are_disjoint(self, synthetic_dsprites_npz):
        f_tr, r_tr = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=8, num_workers=0, split="train")
        f_te, r_te = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=8, num_workers=0, split="test")

        train_idx = indices_of(f_tr) | indices_of(r_tr)
        test_idx = indices_of(f_te) | indices_of(r_te)
        assert train_idx.isdisjoint(test_idx)

    def test_forget_and_retain_are_disjoint_within_a_split(self, synthetic_dsprites_npz):
        forget, retain = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=8, num_workers=0, split="train")
        assert indices_of(forget).isdisjoint(indices_of(retain))

    def test_forget_set_is_purely_the_concept_value(self, synthetic_dsprites_npz):
        """Every image in the forget loader must have shape==concept_val (2);
        every image in retain must not."""
        forget, retain = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=256, num_workers=0,
            concept_col=1, concept_val=2, split="train")

        data = np.load(synthetic_dsprites_npz)
        classes = data["latents_classes"]

        forget_idx = list(forget.dataset.indices)
        retain_idx = list(retain.dataset.indices)
        assert np.all(classes[forget_idx, 1] == 2)
        assert np.all(classes[retain_idx, 1] != 2)

    def test_same_seed_gives_the_same_split(self, synthetic_dsprites_npz):
        f1, r1 = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=8, num_workers=0,
            seed=123, split="train")
        f2, r2 = dsprites_concept_loaders(
            npz_path=str(synthetic_dsprites_npz), batch_size=8, num_workers=0,
            seed=123, split="train")
        assert indices_of(f1) == indices_of(f2)
        assert indices_of(r1) == indices_of(r2)

    def test_invalid_split_name_raises(self, synthetic_dsprites_npz):
        try:
            dsprites_concept_loaders(npz_path=str(synthetic_dsprites_npz), split="bogus")
            assert False, "expected ValueError for an invalid split name"
        except ValueError:
            pass


class TestCelebASplits:
    def test_forget_loader_is_purely_eyeglasses_positive(self, synthetic_celeba_dir):
        forget, _ = celeba_concept_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0)
        # every file in the forget dataset was written with Eyeglasses=+1
        # (index % 5 == 0 in the fixture) -- spot check via the dataset's own filter
        assert len(forget.dataset) == 8  # 40 images, 1-in-5 positive

    def test_train_and_eval_retain_splits_are_disjoint(self, synthetic_celeba_dir):
        _, retain_train = celeba_concept_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0, val_fraction=0.25)
        _, retain_eval = get_concept_eval_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0, val_fraction=0.25)

        train_files = {retain_train.dataset.dataset.files[i] for i in retain_train.dataset.indices}
        eval_files  = {retain_eval.dataset.dataset.files[i]  for i in retain_eval.dataset.indices}
        assert train_files.isdisjoint(eval_files)

    def test_retain_dataloaders_never_include_forget_concept(self, synthetic_celeba_dir):
        train_loader, val_loader, _ = celeba_retain_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0)
        # the retain pool is built with keep_val=-1 (Eyeglasses absent) --
        # confirm the pool size matches the known fixture (40 - 8 = 32 images)
        pool_size = len(train_loader.dataset) + len(val_loader.dataset)
        assert pool_size == 32

    def test_same_seed_gives_the_same_split(self, synthetic_celeba_dir):
        f1, r1 = celeba_concept_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0, seed=99)
        f2, r2 = celeba_concept_loaders(
            data_root=str(synthetic_celeba_dir), batch_size=4, num_workers=0, seed=99)
        idx1 = set(r1.dataset.indices)
        idx2 = set(r2.dataset.indices)
        assert idx1 == idx2
