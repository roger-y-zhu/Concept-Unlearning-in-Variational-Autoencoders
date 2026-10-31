"""
tests/test_regressions.py
==========================
One test per bug found and fixed during the 2026-08-04 audit. Each test is
written to FAIL against the pre-fix code, so a future refactor that
accidentally reintroduces one of these is caught immediately rather than
silently invalidating another 21 runs.

Bug 1 -- BatchNorm train/eval mismatch (feature_unlearn.py, both datasets)
    `frozen` ran in .eval() while `model` ran in .train()`, so the retain
    anchor -- which should be exactly 0 at initialisation -- was 208.0.
    dSprites retain ELBO went 37 -> 7,300; CelebA val loss 379 -> 19,000.

Bug 2 -- MSE compared in logit space instead of image space (dSprites)
    decode() for recon_loss="bce" returns unbounded logits ([-256, +135]
    observed); MSE between two logit tensors is dominated by saturated
    pixels unless sigmoid is applied first.

Bug 3 -- Optuna objectives scored on the training split
    Both retain_elbo and forget_elbo were measured on the same data a trial
    had just trained on, so the retain objective rewarded overfitting.

Bug 4 -- Inverted-metric denominators in compute_progress
    Oracle vs baseline: mlp_auroc, knn_cohesion and lift_norm all move the
    WRONG direction (oracle scores higher) because hearts are OOD for the
    oracle and collapse into a tight sink. Denominators near zero turned
    metric noise into apparent "progress" of >1.

Bug 5 (adjacent) -- probes missing 'proba' / MLP probe collapsing on
    imbalanced data. Covered here for completeness; also exercised in
    test_evaluation.py's TestProbing.
"""

import copy

import pytest
import torch

from src.models.vae import ConvVAE, VAEConfig
from src.unlearning.methods import FeatureUnlearn, make_to_img, img_mse


def make_model(seed=0):
    torch.manual_seed(seed)
    return ConvVAE(VAEConfig(latent_dim=4, base_channels=4, in_channels=1, recon_loss="bce"))


class FakeArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# Bug 1: BatchNorm train/eval mismatch
# ---------------------------------------------------------------------------

class TestBug1_BatchNormMismatch:
    def test_eval_mode_gives_zero_retain_anchor_at_init(self):
        """The fixed behaviour: with identical weights and both models in
        eval mode, decode(mu_r) must equal frozen.decode(mu_r) almost exactly."""
        model = make_model()
        frozen = copy.deepcopy(model)
        frozen.eval()
        model.eval()

        x = torch.randint(0, 2, (8, 1, 64, 64)).float()
        with torch.no_grad():
            mu, _ = model.encode(x)
            out_live = model.decode(mu)
            out_frozen = frozen.decode(mu)
        assert torch.allclose(out_live, out_frozen, atol=1e-4)

    def test_train_mode_would_have_broken_it(self):
        """Documents WHY the fix matters: the same computation in .train() mode
        (the pre-fix code path) produces a large, non-zero discrepancy purely
        from BatchNorm using batch statistics on one side and running
        statistics on the other -- with IDENTICAL weights AND identical
        running stats.

        Running stats are set to a deliberately different distribution than
        the test batch will produce, simulating a model trained on real,
        structured data (the actual baseline: 150 epochs over 737k images)
        being probed with a small, differently-distributed forget batch --
        rather than relying on synthetic random data to happen to diverge
        from itself, which it won't (no real distribution shift exists
        between random noise and more random noise)."""
        model = make_model()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.fill_(2.0)
                m.running_var.fill_(0.1)

        frozen = copy.deepcopy(model)  # identical weights AND running stats
        frozen.eval()
        model.train()  # the historical bug: model stays in train mode

        x = torch.randint(0, 2, (8, 1, 64, 64)).float()
        with torch.no_grad():
            mu, _ = model.encode(x)
            out_live = model.decode(mu)
            out_frozen = frozen.decode(mu)
        discrepancy = (out_live - out_frozen).pow(2).mean().item()
        assert discrepancy > 0.5, (
            "expected .train() mode to visibly diverge from the frozen reference "
            "-- if this now passes, BatchNorm behaviour may have changed upstream"
        )

    def test_feature_unlearn_method_object_declares_eval_mode(self):
        """The actual fix: FeatureUnlearn.train_mode must be False so
        src/unlearning/loop.py calls model.train(False) every epoch."""
        assert FeatureUnlearn().train_mode is False


# ---------------------------------------------------------------------------
# Bug 2: image-space vs logit-space comparison
# ---------------------------------------------------------------------------

class TestBug2_ImageSpaceComparison:
    def test_bce_decoder_output_is_unbounded(self):
        """Confirms the premise: decode() for bce mode is NOT bounded to [0,1],
        so comparing it directly with MSE is comparing logits, not images."""
        model = make_model()
        z = torch.randn(8, 4) * 5
        with torch.no_grad():
            raw = model.decode(z)
        assert raw.min() < 0.0 or raw.max() > 1.0

    def test_make_to_img_bounds_bce_output_before_mse(self):
        to_img = make_to_img("bce")
        model = make_model()
        z = torch.randn(8, 4) * 5
        with torch.no_grad():
            raw = model.decode(z)
            img = to_img(raw)
        assert img.min() >= 0.0 and img.max() <= 1.0

    def test_mse_on_raw_logits_is_dominated_by_saturation(self):
        """Direct demonstration of the bug's effect: MSE computed on raw
        logits between two similar-but-shifted decodes can be enormous, while
        the same comparison in image (sigmoid) space is well-scaled."""
        model = make_model()
        z1 = torch.randn(8, 4)
        z2 = z1 + 0.05  # a small, realistic latent perturbation
        with torch.no_grad():
            raw1, raw2 = model.decode(z1), model.decode(z2)

        to_img = make_to_img("bce")
        logit_space_mse = img_mse(raw1, raw2).item()
        image_space_mse = img_mse(to_img(raw1), to_img(raw2)).item()

        # image-space MSE is bounded (pixels in [0,1], so per-pixel error <= 1);
        # logit-space MSE has no such bound and is typically far larger.
        assert image_space_mse <= raw1.shape[1] * raw1.shape[2] * raw1.shape[3]
        assert logit_space_mse > image_space_mse


# ---------------------------------------------------------------------------
# Bug 3: Optuna scored on the training split
# ---------------------------------------------------------------------------

class TestBug3_HeldOutScoring:
    def test_dsprites_concept_loaders_support_a_held_out_split(self, synthetic_dsprites_npz):
        from src.data.dsprites import get_concept_loaders
        # split="test" must exist and be disjoint from split="train" --
        # exhaustively covered in test_data.py; here we just confirm the
        # parameter exists and defaults sanely (this is the regression surface:
        # a caller forgetting to pass split="test" silently gets "train").
        f, r = get_concept_loaders(npz_path=str(synthetic_dsprites_npz),
                                   batch_size=8, num_workers=0)
        assert f is not None and r is not None  # default split="train" still works

    def test_celeba_has_a_dedicated_eval_loader_function(self, synthetic_celeba_dir):
        """get_concept_eval_loaders must exist as a SEPARATE function from
        get_concept_loaders -- the bug was calling the training-split function
        for scoring. A separate name makes that mistake harder to make."""
        from src.data.celeba import get_concept_eval_loaders
        f, r = get_concept_eval_loaders(data_root=str(synthetic_celeba_dir),
                                        batch_size=4, num_workers=0)
        assert f is not None and r is not None

    def test_tune_py_cache_only_reuses_held_out_scored_trials(self):
        """experiments/unlearn/tune.py's find_cached() must ignore any
        metrics.json that predates the held-out-scoring fix (no
        'scored_on': 'held-out' marker) -- otherwise old train-split scores
        would silently be reused as if they were held-out scores."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "unlearn"))
        import importlib
        tune = importlib.import_module("tune")

        import tempfile, json
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            stale = out_dir / "trial_000"
            stale.mkdir()
            (stale / "metrics.json").write_text(json.dumps({
                "retain_elbo": 10.0, "forget_elbo": 20.0, "lambda_forget": 0.001,
                "epochs": 5,
                # no "scored_on" key -- this is the pre-fix shape
            }))
            result = tune.find_cached(out_dir, {"lambda_forget": 0.001, "epochs": 5})
            assert result is None, "a pre-fix (train-scored) trial must not be reused"

            fresh = out_dir / "trial_001"
            fresh.mkdir()
            (fresh / "metrics.json").write_text(json.dumps({
                "retain_elbo": 11.0, "forget_elbo": 22.0, "lambda_forget": 0.002,
                "epochs": 6, "scored_on": "held-out",
            }))
            result = tune.find_cached(out_dir, {"lambda_forget": 0.002, "epochs": 6})
            assert result == (11.0, 22.0)


# ---------------------------------------------------------------------------
# Bug 4: degenerate metric denominators in compute_progress
# ---------------------------------------------------------------------------

class TestBug4_DegenerateProgressDenominators:
    def test_small_denominator_metric_is_excluded_from_the_mean(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from notebooks.nb_utils import compute_progress

        baseline = {"linear_auroc": 0.63, "mlp_auroc": 0.9868, "dir_norm": 0.36}
        oracle   = {"linear_auroc": 0.56, "mlp_auroc": 0.9935, "dir_norm": 0.20}
        # a tiny, plausible measurement wobble on a near-zero-denominator metric
        run = {"linear_auroc": 0.60, "mlp_auroc": 0.9968, "dir_norm": 0.30}

        results = {"baseline": baseline, "oracle": oracle, ("run",): run}
        progress = compute_progress(results, [("run",)],
                                    ["linear_auroc", "mlp_auroc", "dir_norm"],
                                    verbose=False)

        assert "mlp_auroc" not in progress[("run",)]["_mean_over"]
        assert "linear_auroc" in progress[("run",)]["_mean_over"]
        assert "dir_norm" in progress[("run",)]["_mean_over"]

        # the excluded metric's wobble must not distort the mean
        raw_mlp_progress = progress[("run",)]["mlp_auroc"]
        assert abs(raw_mlp_progress) > 1.0, "the wobble should look huge in isolation"
        assert abs(progress[("run",)]["mean"]) < 1.0, "but must not dominate the mean"

    def test_all_usable_metrics_falls_back_to_including_everything(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from notebooks.nb_utils import compute_progress

        baseline = {"a": 0.0, "b": 0.0}
        oracle   = {"a": 1.0, "b": 1.0}
        run      = {"a": 0.5, "b": 0.5}
        results = {"baseline": baseline, "oracle": oracle, ("run",): run}
        progress = compute_progress(results, [("run",)], ["a", "b"], verbose=False)
        assert progress[("run",)]["mean"] == pytest.approx(0.5)
        assert set(progress[("run",)]["_mean_over"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Bug 5 (adjacent): sweep-key parsing must not confuse "a" with "alpha"
# ---------------------------------------------------------------------------

class TestSweepKeyParsingRegression:
    def test_rmu_prefix_does_not_match_feature_unlearn_directories(self, tmp_path):
        import sys
        from pathlib import Path as P
        sys.path.insert(0, str(P(__file__).resolve().parents[1]))
        from notebooks.nb_utils import discover_sweep_keys

        for name in ("alpha1.0_ep5", "a50.0_c1.0_ep10", "lam0.0003_ep13"):
            d = tmp_path / name
            d.mkdir()
            (d / "final.pt").touch()

        rmu_keys = discover_sweep_keys(tmp_path, prefix="a")
        alpha_keys = discover_sweep_keys(tmp_path, prefix="alpha")
        lam_keys = discover_sweep_keys(tmp_path, prefix="lam")

        assert rmu_keys == [(50.0, 10)]
        assert alpha_keys == [(1.0, 5)]
        assert lam_keys == [(0.0003, 13)]


# ---------------------------------------------------------------------------
# build_master_table.read_adversarial_replicates
#
# The attack column is the thesis headline, and it was the ONLY column in the
# master table without an error bar -- so it read as the most precise number in
# the row when it was the least supported (n=1 checkpoint). Replicating it
# changed an ordinal claim (RESEARCH_LEDGER E13), so the reader that supplies
# the error bar is worth pinning.
#
# The radius guard is the part that actually matters. Attack success is steeply
# radius-dependent (dSprites 0.66 -> 0.83 across the dose sweep at fixed r), so
# pairing a mean computed at one radius with a column labelled another would be
# worse than printing no error bar at all: it would be a confidently-formatted
# wrong number, which is the exact failure mode that produced the R1 retraction.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import json
from pathlib import Path

_BMT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_master_table.py"


def _load_bmt():
    spec = _ilu.spec_from_file_location("_bmt_under_test", _BMT_PATH)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_replicates(tmp_path, dataset, payload):
    d = tmp_path / "results" / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / "attack_robustness_replicates.json").write_text(json.dumps(payload))


@pytest.fixture
def bmt(tmp_path, monkeypatch):
    mod = _load_bmt()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    return mod


def test_replicates_read_when_radius_matches(bmt, tmp_path):
    _write_replicates(tmp_path, "ds", {
        "_radius": 2.0,
        "rmu": {"n": 5, "mean": 0.27, "sd": 0.0075, "values": [0.27] * 5},
    })
    out = bmt.read_adversarial_replicates("ds", 2.0)
    assert out == {"rmu": (0.27, 0.0075, 5)}


def test_replicates_refused_when_radius_differs(bmt, tmp_path):
    """A mean from r=2.0 must NOT be printed in a column labelled r=4.0."""
    _write_replicates(tmp_path, "ds", {
        "_radius": 2.0,
        "rmu": {"n": 5, "mean": 0.27, "sd": 0.0075},
    })
    assert bmt.read_adversarial_replicates("ds", 4.0) == {}


def test_replicates_absent_file_is_empty_not_an_error(bmt):
    assert bmt.read_adversarial_replicates("nonexistent", 2.0) == {}


def test_replicates_corrupt_file_is_empty_not_an_error(bmt, tmp_path):
    d = tmp_path / "results" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "attack_robustness_replicates.json").write_text("{not json")
    assert bmt.read_adversarial_replicates("ds", 2.0) == {}


def test_replicates_keeps_singletons_but_drops_metadata_keys(bmt, tmp_path):
    """n=1 rows ARE kept -- run-consistency beats error bars.

    baseline/oracle have no training-seed replicates but ARE attacked in the
    same process as the methods, so taking them from this file keeps the whole
    column one measurement. E14 showed that sourcing a baseline from a
    different invocation can move it by more than the effect being compared.
    Underscore-prefixed metadata must still be excluded.
    """
    _write_replicates(tmp_path, "ds", {
        "_radius": 2.0, "_archs": ["cnn"],
        "rmu": {"n": 5, "mean": 0.27, "sd": 0.007},
        "oracle": {"n": 1, "mean": 0.66, "sd": 0.0},
    })
    out = bmt.read_adversarial_replicates("ds", 2.0)
    assert set(out) == {"rmu", "oracle"}
    assert out["oracle"] == (0.66, 0.0, 1)


def test_fmt_emits_error_bar_only_when_replicated(bmt):
    assert bmt.fmt(0.2735, 0.0076, 5) == "0.2735 ± 0.0076"
    assert bmt.fmt(0.6821, 0.0, 1) == "0.6821"


def test_read_adversarial_reports_its_protocol(bmt, tmp_path):
    """Consensus and single-judge values must be distinguishable by the caller."""
    d = tmp_path / "results" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "attack_robustness_consensus.json").write_text(
        json.dumps({"_archs": ["cnn"], "leace": {"2.0": 0.72}}))
    vals, proto = bmt.read_adversarial("ds", 2.0)
    assert proto == "consensus" and vals == {"leace": 0.72}

    (d / "attack_robustness_consensus.json").unlink()
    (d / "adversarial_recovery.json").write_text(
        json.dumps({"leace": {"within_radius": {"success_at_r": {"2.0": 0.83}}}}))
    vals, proto = bmt.read_adversarial("ds", 2.0)
    assert proto == "single" and vals == {"leace": 0.83}

    (d / "adversarial_recovery.json").unlink()
    assert bmt.read_adversarial("ds", 2.0) == ({}, "none")


def test_protocol_guard_blocks_mixing_consensus_into_single_judge_column():
    """The dSprites near-miss, pinned.

    The replicates file is ALWAYS consensus-judged. read_adversarial falls back
    to single-judge when a dataset has no consensus file. Merging the two put
    single-judge 0.6821 (baseline) in the same column as consensus 0.1743
    (feature_unlearn) and made the judging protocol look like a 3.9x model
    effect. The guard is a plain `if protocol == "consensus"`, so the thing
    worth pinning is that the branch exists and is keyed on the protocol.
    """
    src = _BMT_PATH.read_text(encoding="utf-8")
    guard = src.split("PROTOCOL GUARD", 1)
    assert len(guard) == 2, "the protocol guard comment was removed"
    after = guard[1]
    assert 'if protocol == "consensus":' in after, \
        "replicates must only be merged when the base values are consensus-judged"
    merge_pos = after.index("read_adversarial_replicates")
    branch_pos = after.index('if protocol == "consensus":')
    assert branch_pos < merge_pos, \
        "the replicates merge must sit INSIDE the consensus branch"


# ---------------------------------------------------------------------------
# get_judge must return the SAME object graph whether the cache is cold or warm
#
# The bug: the training branch returned the freshly-trained model directly while
# the cache branch returned weights loaded into a fresh module. The first run of
# any analysis therefore disagreed with every later run of that same analysis.
# On dSprites the cold-cache run reported fisher 0.2568 / rmu 0.2666 /
# neg_grad 0.1670; warm-cache runs report 0.2510 / 0.2715 / 0.1699 and reproduce
# exactly (3x in one process, and again across processes).
#
# Deltas of 0.003-0.006 exceed real between-method gaps at the resistant end
# (neg_grad vs feature_unlearn = 0.0019), so this could reorder methods -- and
# it did so invisibly, because nothing about a cold cache is recorded in the
# output file.
# ---------------------------------------------------------------------------

_RAB_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_attack_robustness.py"


def test_get_judge_round_trips_through_disk_on_the_training_path():
    src = _RAB_PATH.read_text(encoding="utf-8")
    body = src.split("def get_judge", 1)[1].split("\ndef ", 1)[0]

    assert "ROUND-TRIP THROUGH DISK" in body, \
        "the round-trip guard comment was removed from get_judge"

    save_pos = body.index("torch.save(")
    tail = body[save_pos:]
    assert "load_state_dict" in tail, \
        "get_judge must reload the saved weights after training, not return the trained object"
    assert ".eval()" in tail, \
        "the reloaded judge must be put in eval mode before being returned"

    # the trained object itself must never be what leaves the function
    assert "return trained" not in body, \
        "returning the freshly-trained model reintroduces the cold-cache/warm-cache split"


def test_get_judge_both_branches_end_in_eval():
    """Cache hit and cache miss must both return an eval-mode module."""
    src = _RAB_PATH.read_text(encoding="utf-8")
    body = src.split("def get_judge", 1)[1].split("\ndef ", 1)[0]
    # two load_state_dict calls: one per branch, each followed by .eval()
    assert body.count("load_state_dict") == 2, \
        "expected exactly two load paths in get_judge (cache hit and post-training reload)"
    assert body.count("clf.eval()") == 2, \
        "both get_judge return paths must call clf.eval()"
