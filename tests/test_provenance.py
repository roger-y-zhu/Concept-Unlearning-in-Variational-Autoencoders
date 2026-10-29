"""
tests/test_provenance.py
=========================
run_meta.json is the record that makes a result traceable. These tests pin the
properties that make it worth trusting -- above all that it captures the
RESOLVED configuration, not just what the operator typed.
"""

import argparse
import json

from src.unlearning.provenance import write_run_meta, read_run_meta, git_state


def _args(**over):
    """A namespace shaped like the real parser's output, defaults included."""
    base = dict(dataset="dsprites", method="leace", lr=5e-5, epochs=1,
                val_tolerance=0.10, erase_strength=1.0, fit_samples=0,
                train_seed=None, out_dir=None, u_seed=3, fisher_seed=0)
    base.update(over)
    return argparse.Namespace(**base)


def _write(tmp_path, **over):
    write_run_meta(tmp_path, args=_args(**over), method_name="leace",
                   dataset="dsprites", run_name="leace_ep1",
                   seeds={"split": 48025845, "train": 48025845, "method": {}})
    return json.loads((tmp_path / "run_meta.json").read_text())


class TestRunMetaCapturesConfiguration:
    def test_records_defaults_the_directory_name_omits(self, tmp_path):
        """The whole point: a run name like `leace_ep1` says nothing about lr
        or val_tolerance, so changing a default would silently rewrite the
        meaning of every existing directory."""
        meta = _write(tmp_path)
        assert meta["args"]["lr"] == 5e-5
        assert meta["args"]["val_tolerance"] == 0.10
        assert meta["args"]["fit_samples"] == 0

    def test_records_all_three_seed_kinds_separately(self, tmp_path):
        """Split / training / method-intrinsic seeds are different experimental
        variables and their variances are not interchangeable."""
        meta = _write(tmp_path)
        assert meta["seeds"]["split"] == 48025845
        assert meta["seeds"]["train"] == 48025845
        assert "method" in meta["seeds"]

    def test_survives_non_json_argument_values(self, tmp_path):
        """argparse namespaces carry Paths and similar; serialisation must not
        throw partway and leave no provenance at all."""
        from pathlib import Path
        meta = _write(tmp_path, out_dir=Path("results/x"))
        assert isinstance(meta["args"]["out_dir"], str)
        assert "results" in meta["args"]["out_dir"]

    def test_written_before_training_so_it_exists_for_interrupted_runs(self, tmp_path):
        _write(tmp_path)
        assert (tmp_path / "run_meta.json").exists()
        assert "started_utc" in json.loads((tmp_path / "run_meta.json").read_text())

    def test_schema_is_versioned(self, tmp_path):
        """Readers must be able to tell an old layout from a new one."""
        assert _write(tmp_path)["_schema"] == 1


class TestGitState:
    def test_dirty_unknown_is_not_reported_as_clean(self):
        """None (git unavailable) and False (verified clean) must stay
        distinct -- collapsing them makes an unknown state read as a clean one,
        which is exactly the claim a provenance record must never make."""
        g = git_state()
        assert g["dirty"] is None or isinstance(g["dirty"], bool)
        if g["dirty"] is None:
            assert g["dirty_files"] is None

    def test_reports_commit_and_branch_keys(self):
        g = git_state()
        assert set(g) == {"commit", "branch", "dirty", "dirty_files"}


class TestReadBack:
    def test_roundtrips(self, tmp_path):
        _write(tmp_path)
        assert read_run_meta(tmp_path)["method"] == "leace"

    def test_returns_none_for_runs_predating_provenance(self, tmp_path):
        """Every run made before this module exists has no run_meta.json.
        Readers must degrade to None rather than raising, or the whole existing
        results tree becomes unreadable."""
        assert read_run_meta(tmp_path) is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        (tmp_path / "run_meta.json").write_text("{not json")
        assert read_run_meta(tmp_path) is None


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

class TestMetricCacheIsKeyedByCheckpoint:
    """Two consumers, one directory, two different right answers.

    Sweep notebooks read final.pt; the master table reads the trade-off epoch.
    Both are correct for their own question, so the cache must be able to hold
    both. A single per-directory file cannot, and that is precisely how the
    dSprites RMU champion came to be reported with final.pt's metrics.
    """

    def _fake_ckpt(self, d, name):
        import torch
        p = d / name
        torch.save({"epoch": 1}, p)
        return p

    def test_two_checkpoints_get_two_cache_files(self, tmp_path):
        import numpy as np
        from notebooks.nb_utils import bespoke_metrics_cached

        a = self._fake_ckpt(tmp_path, "final.pt")
        b = self._fake_ckpt(tmp_path, "best_tradeoff.pt")
        rng = np.random.default_rng(0)
        mu, sv = rng.normal(size=(400, 4)), rng.integers(0, 3, 400)

        bespoke_metrics_cached(mu, sv, a, forget_val=2)
        bespoke_metrics_cached(mu, sv, b, forget_val=2)

        names = {p.name for p in (tmp_path / "cache").glob("bespoke_metrics*.json")}
        assert names == {"bespoke_metrics_final.json",
                         "bespoke_metrics_best_tradeoff.json"}

    def test_each_cache_records_its_own_checkpoint(self, tmp_path):
        import json
        import numpy as np
        from notebooks.nb_utils import bespoke_metrics_cached

        a = self._fake_ckpt(tmp_path, "final.pt")
        rng = np.random.default_rng(1)
        bespoke_metrics_cached(rng.normal(size=(400, 4)),
                               rng.integers(0, 3, 400), a, forget_val=2)
        data = json.loads((tmp_path / "cache" / "bespoke_metrics_final.json").read_text())
        assert data["_ckpt"] == "final.pt"

    def test_unstamped_legacy_cache_is_not_trusted(self, tmp_path):
        """A legacy file that does not say which checkpoint it describes cannot
        be attributed to one. Trusting it is the original bug."""
        import json
        import numpy as np
        from notebooks.nb_utils import bespoke_metrics_cached

        ckpt = self._fake_ckpt(tmp_path, "best_tradeoff.pt")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "bespoke_metrics.json").write_text(json.dumps(
            {"linear_auroc": 0.999, "_version": 2}))          # no _ckpt

        rng = np.random.default_rng(2)
        out = bespoke_metrics_cached(rng.normal(size=(400, 4)),
                                     rng.integers(0, 3, 400), ckpt, forget_val=2)
        assert out["linear_auroc"] != 0.999, "unstamped legacy cache must be recomputed"


class TestRecoveryCacheIsKeyedByCheckpoint:
    """probe_recovery_cached had the identical directory-not-checkpoint keying
    bug as bespoke_metrics_cached, and it was live: the dSprites RMU champion's
    cached recovery curve was final.pt's, served to every best_tradeoff.pt
    reader, because the only freshness check was mtime-vs-current-checkpoint
    and a cache file is always newer than the checkpoints once written once.
    """

    def _fake_ckpt(self, d, name):
        import torch
        p = d / name
        torch.save({"epoch": 1}, p)
        return p

    def test_two_checkpoints_get_two_cache_files(self, tmp_path):
        import numpy as np
        from notebooks.nb_utils import probe_recovery_cached

        a = self._fake_ckpt(tmp_path, "final.pt")
        b = self._fake_ckpt(tmp_path, "best_tradeoff.pt")
        rng = np.random.default_rng(0)
        mu, sv = rng.normal(size=(300, 4)), rng.integers(0, 2, 300)

        probe_recovery_cached(mu, sv, a, forget_val=1, n_grid=(8, 16), n_seeds=1)
        probe_recovery_cached(mu, sv, b, forget_val=1, n_grid=(8, 16), n_seeds=1)

        names = {p.name for p in (tmp_path / "cache").glob("probe_recovery*.json")}
        assert names == {"probe_recovery_final.json", "probe_recovery_best_tradeoff.json"}

    def test_stale_cache_from_a_different_checkpoint_is_not_served(self, tmp_path):
        """The exact failure mode: a cache computed for one checkpoint, whose
        mtime is newer than a LATER checkpoint someone asks about."""
        import time
        import numpy as np
        from notebooks.nb_utils import probe_recovery_cached

        a = self._fake_ckpt(tmp_path, "final.pt")
        rng = np.random.default_rng(0)
        mu_a, sv_a = rng.normal(loc=0, size=(300, 4)), rng.integers(0, 2, 300)
        r_a = probe_recovery_cached(mu_a, sv_a, a, forget_val=1, n_grid=(8, 16), n_seeds=1)

        time.sleep(0.05)
        b = self._fake_ckpt(tmp_path, "best_tradeoff.pt")  # newer mtime than the cache
        rng2 = np.random.default_rng(99)
        mu_b, sv_b = rng2.normal(loc=50, size=(300, 4)), rng2.integers(0, 2, 300)
        r_b = probe_recovery_cached(mu_b, sv_b, b, forget_val=1, n_grid=(8, 16), n_seeds=1)

        assert r_b["_ckpt"] == "best_tradeoff.pt"
        assert r_a["_ckpt"] == "final.pt"


class TestElboCacheIsKeyedByCheckpoint:
    """per_shape_elbo_cached / per_concept_elbo_cached had the SAME
    directory-not-checkpoint keying as bespoke_metrics (single file per dir,
    mtime-only), plus no code-version stamp. A multi-checkpoint dir could serve
    one checkpoint's ELBO for another, and a change in how the ELBO is computed
    was invisible. These pin the stem-keying + _ckpt + _version fixes.
    """

    def _fake_ckpt(self, d, name):
        import torch
        p = d / name
        torch.save({"epoch": 1}, p)
        return p

    def _shape_loader(self):
        import torch
        imgs = torch.rand(16, 1, 64, 64)
        cls = torch.zeros(16, 6, dtype=torch.long)
        cls[:8, 1] = 2                       # half hearts (forget_col=1, value 2)
        return [(imgs, torch.zeros(16, 6), cls)]

    def test_two_checkpoints_get_two_shape_cache_files(self, tmp_path, tiny_vae_bce):
        import torch
        from notebooks.nb_utils import per_shape_elbo_cached
        a = self._fake_ckpt(tmp_path, "final.pt")
        b = self._fake_ckpt(tmp_path, "best_tradeoff.pt")
        loader = self._shape_loader()
        per_shape_elbo_cached(tiny_vae_bce, loader, torch.device("cpu"), a)
        per_shape_elbo_cached(tiny_vae_bce, loader, torch.device("cpu"), b)
        names = {p.name for p in (tmp_path / "cache").glob("shape_elbo*.json")}
        assert names == {"shape_elbo_final.json", "shape_elbo_best_tradeoff.json"}

    def test_shape_cache_records_checkpoint_and_version(self, tmp_path, tiny_vae_bce):
        import json, torch
        from notebooks.nb_utils import per_shape_elbo_cached, ELBO_CACHE_VERSION
        a = self._fake_ckpt(tmp_path, "final.pt")
        per_shape_elbo_cached(tiny_vae_bce, self._shape_loader(), torch.device("cpu"), a)
        data = json.loads((tmp_path / "cache" / "shape_elbo_final.json").read_text())
        assert data["_ckpt"] == "final.pt"
        assert data["_version"] == ELBO_CACHE_VERSION

    def test_stale_version_is_recomputed(self, tmp_path, tiny_vae_bce):
        """A cache newer than the checkpoint but written by an older code
        version must be recomputed, not trusted."""
        import json, time, torch
        from notebooks.nb_utils import per_shape_elbo_cached
        a = self._fake_ckpt(tmp_path, "final.pt")
        cache = tmp_path / "cache"; cache.mkdir()
        time.sleep(0.02)
        (cache / "shape_elbo_final.json").write_text(json.dumps(
            {"0": 999.0, "1": 999.0, "2": 999.0, "_version": -1, "_ckpt": "final.pt"}))
        out = per_shape_elbo_cached(tiny_vae_bce, self._shape_loader(), torch.device("cpu"), a)
        assert out[2] != 999.0, "stale-version ELBO cache must be recomputed"

    def test_two_checkpoints_get_two_concept_cache_files(self, tmp_path, tiny_vae_bce):
        import torch
        from notebooks.nb_utils import per_concept_elbo_cached
        a = self._fake_ckpt(tmp_path, "final.pt")
        b = self._fake_ckpt(tmp_path, "best_tradeoff.pt")
        forget = [(torch.rand(8, 1, 64, 64),)]
        retain = [(torch.rand(8, 1, 64, 64),)]
        per_concept_elbo_cached(tiny_vae_bce, forget, retain, torch.device("cpu"), a)
        per_concept_elbo_cached(tiny_vae_bce, forget, retain, torch.device("cpu"), b)
        names = {p.name for p in (tmp_path / "cache").glob("concept_elbo*.json")}
        assert names == {"concept_elbo_final.json", "concept_elbo_best_tradeoff.json"}
