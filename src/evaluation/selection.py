"""
src/evaluation/selection.py
===========================
Which checkpoint of a run should be analysed?

`best.pt` minimises retain validation loss.  That criterion is monotonically
optimised by not unlearning at all, so it systematically returns the LEAST
unlearned epoch -- for RMU, always epoch 1.  Analysing it answers "which run
did the least", not "which run traded off best".

The rule used here instead is the one the literature actually uses (Kurmanji et
al.: tune "to get a high forget-error while not destroying retain-error"):

    among epochs whose retain val stays within `budget`, take the one that
    forgets most.

`src/unlearning/loop.py` applies this online and writes `best_tradeoff.pt`.
This module applies the same rule *after the fact*, to runs that predate that
file, so the audit script and the analysis notebooks agree with the loop rather
than each reimplementing it.

A run that has NO epoch inside budget returns None -- that absence is a result,
not an error: every setting of that configuration damages retain past tolerance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: A ``_ts<N>`` suffix marks a TRAINING-SEED REPLICATE of another run in the
#: same sweep directory (run.py --train-seed), not a distinct configuration.
_REPLICATE_SUFFIX = re.compile(r"_ts\d+$")


def is_replicate(run_name: str) -> bool:
    """True for ``<config>_ts<N>`` -- a re-run of <config> under another seed."""
    return bool(_REPLICATE_SUFFIX.search(run_name))


#: A ``_a<number>`` suffix marks a LEACE DOSE-RESPONSE point (--erase-strength,
#: queue26) -- an interpolation toward the method, not the method itself.
#:
#: Found live: the dSprites LEACE sweep directory holds both the full method
#: (erase_strength=1.0, no suffix, val 467.20, 12.7x over budget) and 14 dose
#: points a=0.01..0.95. Every dose below a=1.0 is PROVABLY not linearly
#: guarded (src/unlearning/erasure.py's P_a is invertible for a<1 --
#: the thesis, Chapter 7), yet several land inside the retain budget
#: (a=0.01..0.10 do on dSprites). Champion selection has no notion of "this
#: run is a diagnostic sweep point, not a competing configuration" -- exactly
#: the problem is_replicate solves for _ts -- so without this exclusion the
#: champion selector picks a=0.1 (val 39.79, linear AUROC at CHANCE, i.e.
#: literally the baseline's value, zero guardedness bought) as "the LEACE
#: champion" purely because it is the only in-budget point, and reports it as
#: though it were the method. That would have been, verbatim,
#: "the most misleading row in the thesis" (partial_erasure_invariance memory
#: note written when this exact scenario was anticipated and warned against).
#: Matches LeaceErase.run_name's exact construction order: "_a<strength>"
#: optionally followed by "_fs<fit_samples>" (--fit-samples caps the fitting
#: sample count and is independent of the dose).
_DOSE_SUFFIX = re.compile(r"_a[\d.]+(_fs\d+)?$")


def is_dose_variant(run_name: str) -> bool:
    """True for ``<config>_a<strength>`` -- an interpolation point, not the
    method at full strength."""
    return bool(_DOSE_SUFFIX.search(run_name))


#: A ``_bl<tag>`` suffix marks a run REFIT ON A REPLICATE BASELINE
#: (run.py --baseline-ckpt). It exists for the closed-form erasure methods,
#: which have no --train-seed axis: LEACE and INLP are post-hoc transforms of a
#: trained encoder, so the only way to give them an error bar is to refit them
#: on each of the queue28 baseline replicates.
#:
#: Excluded from champion selection for exactly the reason ``_ts`` is, and it
#: bit immediately when the exclusion was missing: each refit derives its retain
#: budget from ITS OWN baseline (CelebA ts1 gives 417.33 against the canonical
#: 417.18), so a refit can land marginally further inside its own budget and
#: displace the canonical run. Observed live -- `leace_ep1_blts3` became the
#: LEACE champion at attack 0.8564, replacing `leace_ep1` at 0.7300, which would
#: have reported a 17% swing in a headline number purely because a replicate was
#: allowed to compete with the thing it is a replicate OF.
_BASELINE_REFIT_SUFFIX = re.compile(r"_bl\w+$")


def is_baseline_refit(run_name: str) -> bool:
    """True for ``<config>_bl<tag>`` -- the same method refit on another baseline."""
    return bool(_BASELINE_REFIT_SUFFIX.search(run_name))


def tradeoff_epoch(history: dict, forget_better: str, budget: float) -> int | None:
    """1-indexed epoch that forgets most while retain val stays within budget.

    `forget_better` is the method's UnlearnMethod.forget_better:
      "higher" -- forget_loss is a negative ELBO being pushed up (NegGrad+)
      "lower"  -- forget_loss is a distance to a target (Feature Unlearning, RMU)
    """
    best_ep, best_val = None, None
    for i, (f, v) in enumerate(zip(history["forget_loss"], history["val_loss"]), start=1):
        if f is None or f != f or v > budget:        # NaN forget, or over budget
            continue
        if best_val is None or (f > best_val if forget_better == "higher" else f < best_val):
            best_ep, best_val = i, f
    return best_ep


def load_history(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    return h if h.get("val_loss") else None


def resolve_checkpoint(run_dir: Path, epoch: int, n_epochs: int) -> tuple[Path, bool]:
    """Path to the weights for `epoch`, and whether they are actually that epoch.

    Sweeps run with --save-every 0 kept only best.pt and final.pt, so the
    trade-off epoch's weights often do not exist on disk.  The bool lets the
    caller report that honestly instead of silently analysing final.pt while
    labelling it the trade-off checkpoint.
    """
    run_dir = Path(run_dir)
    exact = run_dir / f"epoch_{epoch:04d}.pt"
    if exact.exists():
        return exact, True
    if (run_dir / "best_tradeoff.pt").exists():
        return run_dir / "best_tradeoff.pt", True
    if epoch == n_epochs and (run_dir / "final.pt").exists():
        return run_dir / "final.pt", True
    return run_dir / "final.pt", False


def champion_run(sweep_dir: Path, forget_better: str, budget: float,
                 *, name_filter=None, require_exact: bool = False,
                 include_replicates: bool = False,
                 include_dose_variants: bool = False) -> dict | None:
    """The single best-trading-off run in a sweep directory.

    Returns a dict with run_dir / epoch / forget / val / ckpt / ckpt_is_exact,
    or None if no run in the sweep has any epoch inside budget.

    include_dose_variants
        OFF by default, and it matters just as much as include_replicates.
        ``_a<strength>`` directories are LEACE dose-response points
        (--erase-strength < 1.0), not the method -- and unlike replicates,
        which merely bias an estimate, letting a dose point compete for
        champion status can select a run that is PROVABLY not linearly
        guarded (any a<1 has an invertible P_a, so guardedness is exactly
        zero) purely because it is cheap enough to land in budget. See
        is_dose_variant's docstring for the concrete number this produced.

    include_replicates
        OFF by default, and it matters.  ``_ts<N>`` directories are the SAME
        configuration re-run under a different training seed, so letting them
        compete here picks the luckiest seed and reports it as the method's
        result -- winner's curse, and it gets worse the more seeds are added.
        The point of replicates is to put an error bar ON a configuration, and
        that is destroyed if selection first skims the maximum off the same
        spread.

        Concretely: adding a single dSprites Fisher replicate at train-seed 1
        was enough to displace the seed-0 run as champion on forget loss --
        changing a method's reported champion purely by measuring it twice.

        Selection therefore runs over configurations only; replicates are
        gathered afterwards, against the already-selected config.

    require_exact
        Only consider runs whose trade-off epoch's weights actually exist on
        disk (``ckpt_is_exact``).  Off by default so existing callers keep the
        old behaviour and can report the flag themselves.

        Turn it ON for any analysis that loads the weights and computes metrics
        from them.  Otherwise ``resolve_checkpoint`` silently falls back to
        ``final.pt``, and the run is analysed at its LAST epoch while being
        labelled with the trade-off epoch's numbers.  That is not a small
        discrepancy: on CelebA, ``celeba_finetune/lam0.02_ep20`` trades off at
        epoch 15 with val 407.14 (inside a 417.18 budget), but its final.pt is
        epoch 20 at val 8022 -- a model destroyed twenty times past budget.
        Analysed unknowingly, it reported a concept direction norm of 50.0
        against the baseline's 2.1 and CKA-to-baseline of 0.06, which looks
        like a spectacular unlearning result and is purely an artefact of
        loading the wrong epoch.

        Selecting the best run that HAS usable weights is more honest than
        picking the best run overall and then quietly analysing something else.
    """
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        return None

    best = None
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if not include_replicates and is_replicate(run_dir.name):
            continue
        if not include_dose_variants and is_dose_variant(run_dir.name):
            continue
        if not include_replicates and is_baseline_refit(run_dir.name):
            continue
        if name_filter and not name_filter(run_dir.name):
            continue
        h = load_history(run_dir)
        if h is None:
            continue
        ep = tradeoff_epoch(h, forget_better, budget)
        if ep is None:
            continue
        ckpt, exact = resolve_checkpoint(run_dir, ep, len(h["val_loss"]))
        if require_exact and not exact:
            continue
        f = h["forget_loss"][ep - 1]
        better = best is None or (
            f > best["forget"] if forget_better == "higher" else f < best["forget"])
        if better:
            best = {"run_dir": run_dir, "name": run_dir.name, "epoch": ep,
                    "forget": f, "val": h["val_loss"][ep - 1],
                    "n_epochs": len(h["val_loss"]),
                    "ckpt": ckpt, "ckpt_is_exact": exact}
    return best


def summarise_runs(sweep_dir: Path, forget_better: str, budget: float,
                   *, name_filter=None, include_replicates: bool = False,
                   include_dose_variants: bool = False) -> list[dict]:
    """Every run in a sweep with its trade-off epoch (or None). For audit tables.

    Excludes ``_ts<N>`` training-seed replicates and ``_a<strength>`` dose
    points by default so an audit table lists configurations rather than
    repeating one config once per seed/dose; pass include_replicates=True /
    include_dose_variants=True to see the raw spread.
    """
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        return []
    out = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if not include_replicates and is_replicate(run_dir.name):
            continue
        if not include_dose_variants and is_dose_variant(run_dir.name):
            continue
        if not include_replicates and is_baseline_refit(run_dir.name):
            continue
        if name_filter and not name_filter(run_dir.name):
            continue
        h = load_history(run_dir)
        if h is None:
            continue
        ep = tradeoff_epoch(h, forget_better, budget)
        out.append({
            "name": run_dir.name, "run_dir": run_dir,
            "best_val_epoch": h["val_loss"].index(min(h["val_loss"])) + 1,
            "tradeoff_epoch": ep,
            "forget": h["forget_loss"][ep - 1] if ep else None,
            "val": h["val_loss"][ep - 1] if ep else None,
            "final_val": h["val_loss"][-1],
            "final_forget": h["forget_loss"][-1],
        })
    return out
