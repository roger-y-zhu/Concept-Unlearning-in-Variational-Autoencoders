"""
tests/test_config.py
=====================
src/utils/config.py: the real experiment configs still load, and the dead
`unlearning:` config path (removed during the unlearning refactor) stays
rejected rather than silently reappearing and training a from-scratch model
into a sweep-results directory.  See docs/ARCHITECTURE.md "Dead ends".
"""

import pytest

from src.utils.config import load_experiment_config


REAL_EXPERIMENTS = [
    "dsprites_baseline", "dsprites_oracle", "dsprites200k",
    "celeba_baseline", "celeba_oracle", "celeba75k",
]


@pytest.mark.parametrize("experiment_name", REAL_EXPERIMENTS)
def test_real_experiment_configs_load(experiment_name):
    cfg = load_experiment_config(experiment_name)
    assert cfg["name"] == experiment_name
    assert cfg["results_dir"].startswith("results/")


def test_missing_experiment_raises():
    # _load_yaml opens the file directly, so a missing experiment surfaces as
    # FileNotFoundError, not the ValueError raised for an empty-but-present file.
    with pytest.raises(FileNotFoundError):
        load_experiment_config("does_not_exist_at_all")


def test_empty_experiment_file_raises_value_error(tmp_path, monkeypatch):
    import src.utils.config as config_module
    root = tmp_path / "fake_project3"
    (root / "configs" / "experiment").mkdir(parents=True)
    (root / "configs" / "experiment" / "blank.yaml").write_text("")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    with pytest.raises(ValueError, match="Missing experiment config"):
        config_module.load_experiment_config("blank")


def test_unlearning_key_is_rejected(tmp_path, monkeypatch):
    """Build a throwaway configs/ tree with a minimal experiment that
    declares `unlearning: finetune`, and confirm load_experiment_config
    refuses it -- this is the exact shape that used to silently train an
    oracle into a sweep-results directory. PROJECT_ROOT is monkeypatched so
    PROJECT_ROOT/configs resolves exactly to our throwaway tree."""
    import src.utils.config as config_module

    root = tmp_path / "fake_project"
    cfgs = root / "configs"
    for sub in ("experiment", "base", "dataset", "model", "unlearning"):
        (cfgs / sub).mkdir(parents=True)

    (cfgs / "base" / "training.yaml").write_text("training:\n  epochs: 10\n")
    (cfgs / "dataset" / "toy.yaml").write_text("dataset:\n  name: toy\n")
    (cfgs / "model" / "vae.yaml").write_text("model:\n  latent_dim: 4\n")
    (cfgs / "unlearning" / "finetune.yaml").write_text("unlearning:\n  epochs: 5\n")
    (cfgs / "experiment" / "toy_finetune.yaml").write_text(
        "name: toy_finetune\nmodel: vae\ndataset: toy\nunlearning: finetune\n"
    )

    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)

    with pytest.raises(ValueError, match="does not implement unlearning"):
        config_module.load_experiment_config("toy_finetune")


def test_experiment_without_unlearning_key_still_loads(tmp_path, monkeypatch):
    import src.utils.config as config_module

    root = tmp_path / "fake_project2"
    cfgs = root / "configs"
    for sub in ("experiment", "base", "dataset", "model"):
        (cfgs / sub).mkdir(parents=True)

    (cfgs / "base" / "training.yaml").write_text("training:\n  epochs: 10\n")
    (cfgs / "dataset" / "toy.yaml").write_text("dataset:\n  name: toy\n")
    (cfgs / "model" / "vae.yaml").write_text("model:\n  latent_dim: 4\n")
    (cfgs / "experiment" / "toy_baseline.yaml").write_text(
        "name: toy_baseline\nmodel: vae\ndataset: toy\n"
    )

    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    cfg = config_module.load_experiment_config("toy_baseline")
    assert cfg["training"]["epochs"] == 10
    assert cfg["model"]["latent_dim"] == 4
