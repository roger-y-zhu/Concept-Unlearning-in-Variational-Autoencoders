"""
tests/test_selection.py
=======================
src/evaluation/selection.py is the single home of the trade-off checkpoint rule,
shared by the audit script and the analysis notebooks (and mirroring what
src/unlearning/loop.py applies online).  If these three ever disagree, the
notebooks silently analyse different checkpoints than the loop saved.
"""

import json

from src.evaluation.selection import (
    tradeoff_epoch, resolve_checkpoint, champion_run, summarise_runs,
)


def hist(forget, val):
    return {"forget_loss": list(forget), "val_loss": list(val)}


def write_run(root, name, forget, val, files=()):
    d = root / name
    d.mkdir(parents=True)
    (d / "history.json").write_text(json.dumps(hist(forget, val)))
    for f in files:
        (d / f).write_text("x")
    return d


class TestTradeoffEpoch:
    def test_picks_max_forget_when_higher_is_better(self):
        h = hist([1.0, 5.0, 3.0], [10.0, 10.0, 10.0])
        assert tradeoff_epoch(h, "higher", budget=20.0) == 2

    def test_picks_min_forget_when_lower_is_better(self):
        h = hist([5.0, 1.0, 3.0], [10.0, 10.0, 10.0])
        assert tradeoff_epoch(h, "lower", budget=20.0) == 2

    def test_ignores_epochs_over_budget(self):
        """The best-forgetting epoch is excluded when it blows the retain budget --
        this is the whole point of the rule."""
        h = hist([5.0, 1.0], [10.0, 99.0])
        assert tradeoff_epoch(h, "lower", budget=20.0) == 1

    def test_returns_none_when_nothing_fits(self):
        h = hist([5.0, 1.0], [99.0, 99.0])
        assert tradeoff_epoch(h, "lower", budget=20.0) is None

    def test_skips_nan_forget(self):
        """Finetune at lambda=0 never computes a forget loss; there is nothing
        to trade off against."""
        h = hist([float("nan"), float("nan")], [10.0, 10.0])
        assert tradeoff_epoch(h, "higher", budget=20.0) is None

    def test_budget_boundary_is_inclusive(self):
        h = hist([1.0], [20.0])
        assert tradeoff_epoch(h, "lower", budget=20.0) == 1

    def test_matches_the_loops_online_rule(self, tiny_spec):
        """The offline rule must agree with what loop.py wrote as
        best_tradeoff.pt -- otherwise notebooks analyse a different epoch."""
        import torch
        from types import SimpleNamespace
        from src.unlearning.loop import run_unlearning
        from src.unlearning.methods import NegGrad

        args = SimpleNamespace(lr=1e-3, grad_clip=1.0, save_every=0, num_workers=0,
                               out_dir=None, epochs=3, max_batches=0,
                               val_tolerance=1e6, lambda_forget=1.0)
        out = run_unlearning(tiny_spec, NegGrad(), args, torch.device("cpu"))
        h = json.loads((out / "history.json").read_text())
        sel = h["selection"]
        assert tradeoff_epoch(h, "higher", sel["val_budget"]) == sel["tradeoff_epoch"]


class TestResolveCheckpoint:
    def test_prefers_exact_epoch_file(self, tmp_path):
        d = write_run(tmp_path, "r", [1], [1], files=["epoch_0002.pt", "final.pt"])
        p, exact = resolve_checkpoint(d, 2, 5)
        assert p.name == "epoch_0002.pt" and exact

    def test_uses_best_tradeoff_when_present(self, tmp_path):
        d = write_run(tmp_path, "r", [1], [1], files=["best_tradeoff.pt", "final.pt"])
        p, exact = resolve_checkpoint(d, 2, 5)
        assert p.name == "best_tradeoff.pt" and exact

    def test_final_is_exact_only_for_the_last_epoch(self, tmp_path):
        d = write_run(tmp_path, "r", [1], [1], files=["final.pt"])
        assert resolve_checkpoint(d, 5, 5)[1] is True
        assert resolve_checkpoint(d, 2, 5)[1] is False

    def test_inexact_fallback_still_returns_a_usable_path(self, tmp_path):
        """Callers must be able to analyse *something*, but must be told it is
        not the trade-off epoch."""
        d = write_run(tmp_path, "r", [1], [1], files=["final.pt"])
        p, exact = resolve_checkpoint(d, 2, 5)
        assert p.name == "final.pt" and not exact


class TestChampionRun:
    def test_picks_the_best_trading_off_run_in_a_sweep(self, tmp_path):
        write_run(tmp_path, "a", [5.0], [10.0], files=["final.pt"])
        write_run(tmp_path, "b", [1.0], [10.0], files=["final.pt"])   # forgets most
        write_run(tmp_path, "c", [0.1], [99.0], files=["final.pt"])   # over budget
        champ = champion_run(tmp_path, "lower", budget=20.0)
        assert champ["name"] == "b"

    def test_returns_none_when_no_run_fits(self, tmp_path):
        write_run(tmp_path, "a", [1.0], [99.0], files=["final.pt"])
        assert champion_run(tmp_path, "lower", budget=20.0) is None

    def test_missing_sweep_dir_is_not_an_error(self, tmp_path):
        assert champion_run(tmp_path / "nope", "lower", budget=20.0) is None

    def test_name_filter_excludes_runs(self, tmp_path):
        write_run(tmp_path, "a10_ep5", [1.0], [10.0], files=["final.pt"])
        write_run(tmp_path, "a10_ep5_s3", [0.5], [10.0], files=["final.pt"])
        champ = champion_run(tmp_path, "lower", budget=20.0,
                             name_filter=lambda n: "_s" not in n)
        assert champ["name"] == "a10_ep5"

    def test_reports_checkpoint_exactness(self, tmp_path):
        # best epoch is 1 of 2, and only final.pt exists -> inexact
        write_run(tmp_path, "a", [1.0, 5.0], [10.0, 10.0], files=["final.pt"])
        champ = champion_run(tmp_path, "lower", budget=20.0)
        assert champ["epoch"] == 1 and champ["ckpt_is_exact"] is False

    def test_require_exact_skips_runs_whose_tradeoff_weights_are_missing(self, tmp_path):
        """Guards a real failure that produced a spectacular-looking artefact.

        Run "a" trades off best but only has final.pt, so resolve_checkpoint
        falls back to the LAST epoch while the caller believes it holds the
        trade-off epoch. On the real CelebA finetune sweep that meant analysing
        a val=8022 model (twenty times past budget) under the label of a
        val=407 one, reporting a concept direction norm of 50.0 against the
        baseline's 2.1 -- pure artefact.

        With require_exact, the best run that actually HAS its weights wins.
        """
        # "a" forgets most (lower is better) but its epoch-1 weights are absent
        write_run(tmp_path, "a", [1.0, 5.0], [10.0, 10.0], files=["final.pt"])
        # "b" forgets less, but its trade-off epoch IS on disk
        write_run(tmp_path, "b", [2.0, 6.0], [10.0, 10.0],
                  files=["final.pt", "epoch_0001.pt"])

        loose = champion_run(tmp_path, "lower", budget=20.0)
        assert loose["name"] == "a" and loose["ckpt_is_exact"] is False

        strict = champion_run(tmp_path, "lower", budget=20.0, require_exact=True)
        assert strict["name"] == "b"
        assert strict["ckpt_is_exact"] is True
        assert strict["ckpt"].name == "epoch_0001.pt"

    def test_require_exact_returns_none_when_no_run_has_usable_weights(self, tmp_path):
        write_run(tmp_path, "a", [1.0, 5.0], [10.0, 10.0], files=["final.pt"])
        assert champion_run(tmp_path, "lower", budget=20.0, require_exact=True) is None

    def test_require_exact_accepts_best_tradeoff_and_final_at_last_epoch(self, tmp_path):
        """resolve_checkpoint counts two other cases as exact: an explicit
        best_tradeoff.pt, and final.pt when the trade-off epoch IS the last."""
        write_run(tmp_path, "a", [5.0, 1.0], [10.0, 10.0], files=["final.pt"])
        champ = champion_run(tmp_path, "lower", budget=20.0, require_exact=True)
        assert champ["name"] == "a" and champ["epoch"] == 2

        write_run(tmp_path, "b", [9.0, 0.5], [10.0, 10.0],
                  files=["final.pt", "best_tradeoff.pt"])
        champ = champion_run(tmp_path, "lower", budget=20.0, require_exact=True)
        assert champ["name"] == "b" and champ["ckpt"].name == "best_tradeoff.pt"


class TestSummariseRuns:
    def test_reports_both_selection_rules_per_run(self, tmp_path):
        write_run(tmp_path, "a", [5.0, 1.0], [10.0, 12.0], files=["final.pt"])
        rows = summarise_runs(tmp_path, "lower", budget=20.0)
        assert len(rows) == 1
        assert rows[0]["best_val_epoch"] == 1      # lowest val
        assert rows[0]["tradeoff_epoch"] == 2      # most forgetting in budget

    def test_run_without_history_is_skipped(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert summarise_runs(tmp_path, "lower", budget=20.0) == []


class TestReplicatesDoNotCompeteAsConfigurations:
    """`_ts<N>` runs are the same config under another training seed.

    Letting them into selection means the reported champion is the max over
    seeds rather than the configuration's actual performance -- the estimate
    gets better the more times you measure, which is exactly backwards. This
    was observed for real: one dSprites Fisher replicate at train-seed 1
    displaced the seed-0 run as champion on forget loss.
    """

    def test_replicate_does_not_displace_its_own_base_run(self, tmp_path):
        write_run(tmp_path, "sig0.01_ep1", forget=[10.0], val=[1.0], files=("final.pt",))
        # Same config, luckier seed -- must not be selectable.
        write_run(tmp_path, "sig0.01_ep1_ts1", forget=[99.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0)
        assert champ["name"] == "sig0.01_ep1"
        assert champ["forget"] == 10.0

    def test_include_replicates_restores_old_behaviour(self, tmp_path):
        write_run(tmp_path, "sig0.01_ep1", forget=[10.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "sig0.01_ep1_ts1", forget=[99.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0, include_replicates=True)
        assert champ["name"] == "sig0.01_ep1_ts1"

    def test_a_genuinely_better_config_still_wins(self, tmp_path):
        """Guard against over-filtering: only _ts suffixes are excluded."""
        write_run(tmp_path, "sig0.01_ep1", forget=[10.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "sig0.02_ep1", forget=[50.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "sig0.01_ep1_ts1", forget=[99.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0)
        assert champ["name"] == "sig0.02_ep1"

    def test_summarise_runs_lists_configs_not_replicates(self, tmp_path):
        write_run(tmp_path, "a_ep1", forget=[1.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "a_ep1_ts1", forget=[2.0], val=[1.0], files=("final.pt",))
        names = {r["name"] for r in summarise_runs(tmp_path, "higher", budget=2.0)}
        assert names == {"a_ep1"}
        names_all = {r["name"] for r in
                     summarise_runs(tmp_path, "higher", budget=2.0, include_replicates=True)}
        assert names_all == {"a_ep1", "a_ep1_ts1"}

    def test_names_without_ts_suffix_are_untouched(self):
        from src.evaluation.selection import is_replicate
        assert is_replicate("sig0.01_ep1_ts1")
        assert is_replicate("lam0.008_ep10_ts12")
        assert not is_replicate("sig0.01_ep1")
        assert not is_replicate("a200.0_c1.0_ep20_s5")   # method-level seed, not train-seed
        assert not is_replicate("lam0.02_ep20_b150")


class TestDoseVariantsDoNotCompeteAsConfigurations:
    """`_a<strength>` runs are LEACE interpolation points (--erase-strength),
    not the method at full strength. Letting them compete for champion status
    is worse than the replicate winner's-curse case: a dose point can be
    PROVABLY unguarded (any a<1 has an invertible P_a) and still win purely by
    being cheap enough to land in budget, producing a champion row that looks
    like "LEACE, in budget" while buying zero guardedness.

    Found live: dsprites_leace's a=0.1 point (val 39.79, chance-level linear
    AUROC) won champion selection over the full method (a=1.0, val 467.20,
    out of budget) because it was the only in-budget candidate.
    """

    def test_dose_variant_does_not_win_over_the_full_method(self, tmp_path):
        # Full method: out of budget, but IS the method.
        write_run(tmp_path, "leace_ep1", forget=[99.0], val=[100.0], files=("final.pt",))
        # Dose point: in budget, but provably not what "leace" means.
        write_run(tmp_path, "leace_ep1_a0.1", forget=[10.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0)
        assert champ is None, (
            "no run should win: the only in-budget candidate is a dose variant "
            "and must be excluded, not silently promoted to champion"
        )

    def test_include_dose_variants_restores_old_behaviour(self, tmp_path):
        write_run(tmp_path, "leace_ep1", forget=[99.0], val=[100.0], files=("final.pt",))
        write_run(tmp_path, "leace_ep1_a0.1", forget=[10.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0, include_dose_variants=True)
        assert champ["name"] == "leace_ep1_a0.1"

    def test_a_genuinely_better_full_run_still_wins_among_full_runs(self, tmp_path):
        write_run(tmp_path, "leace_ep1", forget=[10.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "leace_ep1_a0.5", forget=[99.0], val=[1.0], files=("final.pt",))

        champ = champion_run(tmp_path, "higher", budget=2.0)
        assert champ["name"] == "leace_ep1"

    def test_summarise_runs_excludes_dose_variants_by_default(self, tmp_path):
        write_run(tmp_path, "leace_ep1", forget=[1.0], val=[1.0], files=("final.pt",))
        write_run(tmp_path, "leace_ep1_a0.3", forget=[2.0], val=[1.0], files=("final.pt",))
        names = {r["name"] for r in summarise_runs(tmp_path, "higher", budget=2.0)}
        assert names == {"leace_ep1"}

    def test_is_dose_variant_recognises_leace_naming_only(self):
        from src.evaluation.selection import is_dose_variant
        assert is_dose_variant("leace_ep1_a0.1")
        assert is_dose_variant("leace_ep1_a0.95_fs500")  # fs suffix after a-suffix
        assert not is_dose_variant("leace_ep1")
        assert not is_dose_variant("a200.0_c1.0_ep20_s5")  # RMU: "a" is a PREFIX
        assert not is_dose_variant("lam0.008_ep10")


# ---------------------------------------------------------------------------
# `_bl<tag>` baseline refits must not compete for champion status.
#
# The third instance of the same trap. `_ts` replicates were the first
# (winner's curse), `_a` dose points the second (a provably-unguarded run
# became the LEACE champion). This is the third: `--baseline-ckpt` refits exist
# so the closed-form erasure methods can carry an error bar, and each one
# derives its retain budget from ITS OWN baseline -- CelebA ts1 gives 417.33
# against the canonical 417.18 -- so a refit can sit marginally further inside
# its own budget and displace the canonical run.
#
# Observed live before the guard existed: `leace_ep1_blts3` became the LEACE
# champion at attack@4 = 0.8564, replacing `leace_ep1` at 0.7300. That is a 17%
# swing in a headline number, caused purely by letting a replicate compete with
# the configuration it is a replicate of.
# ---------------------------------------------------------------------------

from src.evaluation.selection import is_baseline_refit


class TestBaselineRefitExclusion:
    def test_recognises_refit_names(self):
        assert is_baseline_refit("leace_ep1_blts3")
        assert is_baseline_refit("k8_ep1_blts1")

    def test_does_not_catch_other_suffixes(self):
        assert not is_baseline_refit("leace_ep1")
        assert not is_baseline_refit("lam0.0_ep30_ts4")
        assert not is_baseline_refit("leace_ep1_a0.5")
        assert not is_baseline_refit("a200.0_c1.0_ep20_s5")

    def test_refit_cannot_displace_canonical_champion(self, tmp_path):
        """The live failure, reproduced: a refit that forgets MORE must still
        not be selected, because it is a replicate rather than a configuration."""
        sweep = tmp_path / "celeba_leace"
        sweep.mkdir(parents=True)
        for name, forget, val in (("leace_ep1", 500.0, 380.0),
                                  ("leace_ep1_blts3", 600.0, 380.0)):
            d = sweep / name
            d.mkdir()
            (d / "history.json").write_text(json.dumps(
                {"val_loss": [val], "forget_loss": [forget]}))
            (d / "final.pt").write_text("")

        champ = champion_run(sweep, "higher", budget=417.18)
        assert champ is not None
        assert champ["name"] == "leace_ep1", (
            "a _bl refit displaced the canonical run -- the exclusion is missing")

        # and with the escape hatch, the refit IS visible
        both = summarise_runs(sweep, "higher", budget=417.18, include_replicates=True)
        assert {r["name"] for r in both} == {"leace_ep1", "leace_ep1_blts3"}
