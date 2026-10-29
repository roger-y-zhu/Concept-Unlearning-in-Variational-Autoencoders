"""
tests/test_recovery.py
=======================
Probe-recovery curves (src/evaluation/recovery.py).

The metric's job is to price residual concept information in attacker-samples,
so the tests plant known amounts of information and check the price comes out
right:

  * a linearly separable concept must be recovered, and faster with more data
    (curve reaches high AUROC, n_to_recover populated);
  * shuffled labels -- zero information by construction -- must stay at
    chance for every n, and never "recover" (the metric must not manufacture
    signal out of overfitting small probes);
  * the LEACE interaction: on LEACE-erased data the LINEAR curve must stay at
    chance for all n (that is the guarantee being purchased) while the MLP
    curve still climbs on planted nonlinear structure -- this is the exact
    reading the thesis will put on these curves, so it is pinned here;
  * protocol honesty: deterministic under seed0, n_grid entries the data
    cannot support are dropped not padded, degenerate inputs raise.
"""

import numpy as np
import pytest
import torch

from src.evaluation.recovery import probe_recovery_curve
from src.unlearning.erasure import erase, fit_leace

from test_erasure import planted_data, xor_data


def small_curve(mu, y, **kw):
    """Recovery curve with test-suite-sized defaults."""
    defaults = dict(n_grid=(8, 32, 128), n_seeds=2, test_size=256)
    defaults.update(kw)
    return probe_recovery_curve(mu, y, **defaults)


class TestRecoversPlantedSignal:
    def test_linear_curve_climbs_and_recovers(self):
        x, z = planted_data(n=3000, d=8, gap=3.0, seed=0)
        out = small_curve(x.numpy(), z.numpy())
        mean = out["curves"]["linear"]["auroc_mean"]
        assert mean[-1] > 0.9, f"separable concept not recovered: {mean}"
        assert mean[-1] >= mean[0] - 0.05, f"curve should not fall with n: {mean}"
        assert out["n_to_recover"]["linear"]["0.9"] is not None

    def test_mlp_curve_recovers_too(self):
        x, z = planted_data(n=3000, d=8, gap=3.0, seed=1)
        out = small_curve(x.numpy(), z.numpy(), probes=("mlp",))
        assert out["curves"]["mlp"]["auroc_mean"][-1] > 0.85

    def test_forget_val_binarisation(self):
        # dSprites-style labels {0,1,2} with forget_val=2.
        x, z = planted_data(n=2000, d=6, gap=3.0, seed=2)
        labels = np.where(z.numpy() == 1, 2, 0)
        out = small_curve(x.numpy(), labels, forget_val=2, probes=("linear",))
        assert out["curves"]["linear"]["auroc_mean"][-1] > 0.9


class TestNoSignalNoRecovery:
    def test_shuffled_labels_stay_at_chance(self):
        x, z = planted_data(n=3000, d=8, gap=3.0, seed=3)
        rng = np.random.RandomState(0)
        y = z.numpy().copy()
        rng.shuffle(y)
        out = small_curve(x.numpy(), y)
        for probe in ("linear", "mlp"):
            for m in out["curves"][probe]["auroc_mean"]:
                assert 0.35 < m < 0.65, (
                    f"{probe} probe manufactured signal from shuffled labels: "
                    f"{out['curves'][probe]['auroc_mean']}")
            assert out["n_to_recover"][probe]["0.8"] is None
            assert out["n_to_recover"][probe]["0.9"] is None


class TestLeaceInteraction:
    """The reading the thesis puts on these curves, pinned as a test."""

    def test_linear_flat_at_chance_mlp_climbs_after_leace(self):
        x, z = xor_data(n=4000, seed=4, leak=2.0)
        p, b = fit_leace(x, z)
        xe = erase(x, p, b).numpy()
        y = z.numpy()

        out = small_curve(xe, y, n_grid=(32, 128, 512), n_seeds=2)

        # One-sided: the guarantee is "cannot BEAT chance"; small-n probes
        # fitting noise legitimately wander below 0.5 on held-out data.
        for m in out["curves"]["linear"]["auroc_mean"]:
            assert m < 0.57, (
                "linear probe beat chance on LEACE-erased data at some n -- "
                f"guarantee is for ALL n: {out['curves']['linear']['auroc_mean']}")
        assert out["curves"]["mlp"]["auroc_mean"][-1] > 0.75, (
            "MLP should still recover the planted nonlinear structure")


class TestProtocol:
    def test_deterministic_given_seed(self):
        x, z = planted_data(n=1500, d=6, seed=5)
        a = small_curve(x.numpy(), z.numpy(), probes=("linear",), seed0=7)
        b = small_curve(x.numpy(), z.numpy(), probes=("linear",), seed0=7)
        assert a == b

    def test_unsupported_ns_are_dropped(self):
        x, z = planted_data(n=400, d=4, seed=6)
        out = probe_recovery_curve(x.numpy(), z.numpy(),
                                   n_grid=(8, 32, 100000), probes=("linear",),
                                   n_seeds=2, test_size=64)
        assert 100000 not in out["n_grid"]
        assert len(out["curves"]["linear"]["auroc_mean"]) == len(out["n_grid"])

    def test_single_class_raises(self):
        x, _ = planted_data(n=200, d=4, seed=7)
        with pytest.raises(ValueError):
            probe_recovery_curve(x.numpy(), np.zeros(200), n_grid=(8,))

    def test_nothing_fits_raises(self):
        x, z = planted_data(n=60, d=4, seed=8)
        with pytest.raises(ValueError):
            probe_recovery_curve(x.numpy(), z.numpy(), n_grid=(4096,),
                                 test_size=16)

    def test_curve_shapes_align(self):
        x, z = planted_data(n=1200, d=5, seed=9)
        out = small_curve(x.numpy(), z.numpy())
        k = len(out["n_grid"])
        for probe, c in out["curves"].items():
            assert len(c["auroc_mean"]) == k
            assert len(c["auroc_sd"]) == k
            assert all(len(row) == k for row in c["per_seed"])
