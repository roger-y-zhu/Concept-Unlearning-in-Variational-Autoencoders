"""
scripts/build_master_table.py
==============================
The cross-method master table -- the central table of the thesis. Regenerates
it from disk so
it stays correct as new runs land, rather than being hand-maintained prose that
silently goes stale. Earlier hand-written versions predated LEACE/INLP, probe
recovery and adversarial recovery, and carried no error bars.

WHAT IT DOES
------------
For each dataset: baseline, oracle, and every method's budget-selected champion
under the shared selection rule (require_exact=True -- see
nb_utils.collect_method_champions for why loose selection silently analyses the
wrong epoch), scored on the same metrics.

SEED AGGREGATION
----------------
A champion config replicated across --train-seed values (queue24/25) is
collapsed to mean +- sd rather than shown as separate rows, because the
replicates are the same configuration and their spread IS the error bar.

Each replicate is re-selected independently: every run has its own history.json
and therefore its own trade-off epoch. Forcing them all to the champion's epoch
would understate the spread, since "which epoch the budget rule picks" is
itself part of what varies between seeds.

A method with n=1 is reported without a +- and flagged, so a reader can see at
a glance which numbers are single-run.

MISSING CACHES
--------------
Metrics are read from the same `<ckpt>.parent/cache/` files the notebooks use.
Anything uncached is reported as missing rather than computed inline -- filling
it is the warmers' job (scripts/warm/), which can run many at once, whereas
this script is a single process and would serialise it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.selection import load_history, tradeoff_epoch, resolve_checkpoint  # noqa: E402
from notebooks.nb_utils import collect_method_champions  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])

# One definition, shared -- this dict previously existed in six files, so
# adding a method meant six edits and missing one produced a table quietly
# short a row rather than an error.
from src.unlearning.spec import METHODS_FORGET_BETTER as METHODS  # noqa: E402
from src.unlearning.spec import REPORT_RADIUS  # noqa: E402  -- single source of truth

DATASETS = {
    "dsprites": {
        "baseline": "results/dsprites/dsprites_baseline/epoch_0150.pt",
        "oracle":   "results/dsprites/dsprites_oracle/epoch_0150.pt",
        "split":    "test",
        "forget_val": 2,
        # Radius at which the ball-constrained attack curves are furthest
        # apart. Reporting one column off a curve always needs an operating
        # point stated with it -- the curves CROSS, so a different r reorders
        # the methods (on CelebA, LEACE sits below baseline at r=2 and well
        # above it at r=4).
        "adv_r": 2.0,
    },
    "celeba": {
        "baseline": "results/celeba/celeba_baseline/epoch_0250.pt",
        "oracle":   "results/celeba/celeba_oracle/epoch_0250.pt",
        "split":    "test",
        "forget_val": 1,
        "adv_r": 4.0,
    },
    "mpi3d": {
        "baseline": "results/mpi3d/mpi3d_baseline/epoch_0120.pt",
        "oracle":   "results/mpi3d/mpi3d_oracle/epoch_0120.pt",
        "split":    "test",
        # Concept label is BINARY (is-red), like CelebA -- not the raw factor
        # value -- so the forget class is 1 on the encoded labels.
        "forget_val": 1,
        # Calibrated 2026-08-30 from run_adversarial_recovery.py: r=8.0 maximises
        # the across-method spread of the ball-constrained success curve (std
        # 0.353, peak over radii 0.5–12), the same max-separation rule that set
        # dSprites=2.0 and CelebA=4.0. MPI3D's optimum is larger because its
        # non-resistant methods saturate slowly (real images, concept harder to
        # reach through the decoder). NOTE the curves CROSS: feature_unlearn is
        # resistant at small r (0.179 vs baseline 0.302 at r=3) but the LEAST
        # resistant by r=8 — so this single column is read alongside the curve.
        "adv_r": 8.0,
    },
}

# The per-dataset "adv_r" comments above document the max-separation calibration,
# but the VALUE is sourced from spec.REPORT_RADIUS so this table can never report
# at a different radius than run_dense_recovery / run_attack_robustness.
for _ds in DATASETS:
    DATASETS[_ds]["adv_r"] = REPORT_RADIUS[_ds]

# Metric key -> (display name, whether lower is better). None = no direction.
METRIC_CFG = [
    ("val",            "retain val",   True),
    ("linear_auroc",   "Linear AUROC", None),
    ("mlp_auroc",      "MLP AUROC",    None),
    ("knn_cohesion",   "kNN cohesion", None),
    ("dir_norm",       "dir norm",     None),
    ("adv_at_r",       "attack@r",     True),
]

# A train-seed replicate directory is the champion's name plus _ts<N>.
_TS = re.compile(r"^(?P<base>.+?)_ts(?P<seed>\d+)$")


def sibling_seed_runs(champ_dir: Path) -> list[Path]:
    """The champion run plus every _ts<N> replicate of the SAME config."""
    champ_dir = Path(champ_dir)
    base = champ_dir.name
    parent = champ_dir.parent
    if not parent.is_dir():
        return [champ_dir]
    out = [champ_dir]
    for d in sorted(parent.iterdir()):
        if not d.is_dir() or d == champ_dir:
            continue
        m = _TS.match(d.name)
        if m and m.group("base") == base:
            out.append(d)
    return out


def read_cached_metrics(ckpt: Path, split: str) -> dict | None:
    """bespoke_metrics for a checkpoint, or None if not warmed yet.

    THIS BUG CLASS AGAIN, ONE LAYER UP: an earlier version of this function
    read the single legacy `cache/bespoke_metrics.json` directly, with NO
    checkpoint-identity check at all -- exactly the failure mode
    nb_utils.bespoke_metrics_cached was fixed to prevent by stem-keying its
    cache files (`bespoke_metrics_<stem>.json`) and stamping `_ckpt`. This
    reader bypassed that fix entirely by reading the file on disk directly
    instead of going through the safe function.

    Audited every run this table reads (52 champion/replicate directories):
    none were actually wrong, because each legacy file happened to still hold
    the write from before nb_utils switched to stem-keyed caches, and that
    write happened to be for the right checkpoint. That is luck of write
    ordering, not a guarantee -- the RMU sweep rewarm (queue's cache warmer,
    unrelated to this table) has since added `bespoke_metrics_final.json`
    alongside those same legacy files, for a DIFFERENT checkpoint, in the same
    directories. The next warm of any of these runs for a still-different
    checkpoint would silently invalidate this reader's assumption again.

    Fixed the same way as nb_utils: prefer the checkpoint-specific file, and
    only trust the legacy path if its own `_ckpt` stamp matches what was asked
    for.
    """
    cache_dir = Path(ckpt).parent / "cache"
    stem_file = cache_dir / f"bespoke_metrics_{Path(ckpt).stem}.json"
    if stem_file.exists():
        try:
            return json.loads(stem_file.read_text())
        except Exception:
            pass
    legacy = cache_dir / "bespoke_metrics.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text())
        except Exception:
            return None
        if data.get("_ckpt") == Path(ckpt).name:
            return data
    return None


def read_recovery(ckpt: Path) -> dict | None:
    """Same checkpoint-identity fix as read_cached_metrics -- see its
    docstring. probe_recovery_cached had the identical bug (single
    per-directory file, mtime-only freshness check) and it was live: the
    dsprites RMU champion's cached curve was final.pt's, not
    best_tradeoff.pt's."""
    cache_dir = Path(ckpt).parent / "cache"
    stem_file = cache_dir / f"probe_recovery_{Path(ckpt).stem}.json"
    if stem_file.exists():
        try:
            return json.loads(stem_file.read_text())
        except Exception:
            pass
    legacy = cache_dir / "probe_recovery.json"
    if legacy.exists():
        try:
            data = json.loads(legacy.read_text())
        except Exception:
            return None
        if data.get("_ckpt") == Path(ckpt).name:
            return data
    return None


def row_for_run(run_dir: Path, forget_better: str, budget: float, split: str) -> dict | None:
    """One replicate's metrics, selected by the shared budget rule."""
    hist = load_history(run_dir)
    if hist is None:
        return None
    ep = tradeoff_epoch(hist, forget_better, budget)
    if ep is None:
        return None  # no epoch in budget -- a result, but not a table row
    ckpt, exact = resolve_checkpoint(Path(run_dir), ep, len(hist["val_loss"]))
    if not exact or not Path(ckpt).exists():
        return None
    m = read_cached_metrics(ckpt, split)
    if m is None:
        return {"_uncached": True, "ckpt": str(ckpt), "val": hist["val_loss"][ep - 1]}
    out = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    out["val"] = hist["val_loss"][ep - 1]
    out["forget"] = hist["forget_loss"][ep - 1]
    rec = read_recovery(ckpt)
    if rec:
        n2r = rec.get("n_to_recover", {})
        for probe in ("linear", "mlp"):
            v = n2r.get(probe, {}).get("0.7")
            out[f"n70_{probe}"] = float(v) if v else float("nan")
    out["ckpt"] = str(ckpt)
    return out


def agg(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0), len(vals)


def fmt(mean: float, sd: float, n: int, places: int = 4) -> str:
    if n == 0:
        return "—"
    if n == 1:
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f} ± {sd:.{places}f}"


def read_adversarial(dataset: str, adv_r: float) -> tuple[dict, str]:
    """({method: attack success within adv_r}, judging_protocol).

    Read from the dataset-level results file rather than a per-checkpoint
    cache, because the attack is run once across all champions by
    scripts/run_adversarial_recovery.py (it needs one shared classifier).

    Returns the PROTOCOL alongside the numbers ("consensus", "single", or
    "none"). Callers must not merge values across protocols -- see build().
    """
    # Prefer the CONSENSUS-judged attack where it exists. Single-judge success
    # mixes concept reachability with classifier fragility, and does so more for
    # the models that resist best (feature_unlearn inflated 3.35x, inlp 1.09x),
    # so the single-judge column understates the very effect the table reports.
    cons = ROOT / f"results/{dataset}/attack_robustness_consensus.json"
    if cons.exists():
        try:
            d = json.loads(cons.read_text())
            out = {k: v[str(adv_r)] for k, v in d.items()
                   if not k.startswith("_") and str(adv_r) in v}
            if out:
                return out, "consensus"
        except Exception:
            pass

    p = ROOT / f"results/{dataset}/adversarial_recovery.json"
    if not p.exists():
        return {}, "none"
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}, "none"
    out = {}
    for k, v in d.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        wr = v.get("within_radius", {}).get("success_at_r", {})
        if str(adv_r) in wr:
            out[k] = wr[str(adv_r)]
    return out, ("single" if out else "none")


def read_adversarial_replicates(dataset: str, adv_r: float) -> dict:
    """{method: (mean, sd, n)} over training-seed replicates of the champion.

    Written by run_attack_robustness.py --mode replicates.  Returns {} when the
    file is absent or was produced at a DIFFERENT radius than the table reports:
    attack success is strongly radius-dependent, so silently pairing a mean from
    one radius with a table column labelled another would be worse than having
    no error bar at all.
    """
    p = ROOT / f"results/{dataset}/attack_robustness_replicates.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    if float(d.get("_radius", -1)) != float(adv_r):
        return {}
    # n=1 rows (baseline, oracle, and the post-hoc methods) ARE returned, even
    # though they carry no error bar. This file is written by a single process
    # that attacks every model against one judge instance, so taking the whole
    # column from it makes the column internally comparable -- which matters
    # more than which file a row came from. Mixing a baseline from one
    # invocation with methods from another is exactly what E14 showed can shift
    # values by more than the effects being compared.
    return {k: (v["mean"], v["sd"], v["n"]) for k, v in d.items()
            if not k.startswith("_") and isinstance(v, dict) and "mean" in v}


def build(dataset: str) -> dict:
    cfg = DATASETS[dataset]
    split = cfg["split"]
    base_ckpt = ROOT / cfg["baseline"]
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    baseline_val = float(ck["metrics"]["loss"])
    budget = baseline_val * 1.10

    champions = collect_method_champions(dataset, METHODS, budget, require_exact=True)

    rows: list[dict] = []
    # baseline / oracle are single models with no sweep and no budget rule.
    for label, key in (("baseline", "baseline"), ("oracle", "oracle")):
        ckpt = ROOT / cfg[key]
        m = read_cached_metrics(ckpt, split) or {}
        rec = read_recovery(ckpt) or {}
        r = {"method": label, "n_seeds": 1, "runs": [str(ckpt)],
             "val": baseline_val if label == "baseline" else float("nan")}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                r[k] = v
        n2r = rec.get("n_to_recover", {})
        for probe in ("linear", "mlp"):
            v = n2r.get(probe, {}).get("0.7")
            r[f"n70_{probe}"] = float(v) if v else float("nan")
        # The oracle has no comparable retain val -- it is trained from scratch
        # on the retain set rather than unlearned from the baseline, so there is
        # no budget for it to sit inside. Drop the key rather than carrying a
        # nan into a thesis table.
        rows.append({"agg": {k: (v, 0.0, 1) for k, v in r.items()
                             if isinstance(v, (int, float))
                             and not (isinstance(v, float) and math.isnan(v))},
                     "method": label, "n_seeds": 1, "note": "", "runs": r["runs"]})

    for method, forget_better in METHODS.items():
        champ = champions.get(method)
        if champ is None:
            rows.append({"agg": {}, "method": method, "n_seeds": 0,
                         "note": "no in-budget champion", "runs": []})
            continue
        run_dirs = sibling_seed_runs(Path(champ["run_dir"]))
        per_run = []
        uncached = 0
        for rd in run_dirs:
            rr = row_for_run(rd, forget_better, budget, split)
            if rr is None:
                continue
            if rr.get("_uncached"):
                uncached += 1
                continue
            per_run.append(rr)
        keys = set()
        for rr in per_run:
            keys |= {k for k, v in rr.items() if isinstance(v, (int, float))}
        aggregated = {k: agg([rr.get(k) for rr in per_run]) for k in sorted(keys)}
        note = ""
        if uncached:
            note = f"{uncached} replicate(s) not yet cached"
        if len(per_run) == 1:
            note = (note + "; " if note else "") + "latent metrics: single run"
        rows.append({"agg": aggregated, "method": method, "n_seeds": len(per_run),
                     "note": note, "runs": [str(Path(r["ckpt"]).parent.name) for r in per_run]})

    adv, protocol = read_adversarial(dataset, cfg["adv_r"])
    for r in rows:
        if r["method"] in adv:
            r["agg"]["adv_at_r"] = (adv[r["method"]], 0.0, 1)

    # Overwrite with the MODEL-LEVEL error bar where training-seed replicates of
    # the champion exist (RESEARCH_LEDGER E13).  Every other column in this table
    # already carries a +- across seeds; this column alone was a single number,
    # so it read as the most precise entry in the row when it was the least
    # supported.  Replicating it changed one ordinal claim on dSprites
    # (neg_grad vs feature_unlearn is a tie, not an ordering), which is exactly
    # the failure mode a missing error bar hides.
    #
    # PROTOCOL GUARD -- do not delete.  The replicates file is always
    # CONSENSUS-judged, but read_adversarial falls back to SINGLE-judge values
    # when a dataset has no consensus file.  Merging them puts two incompatible
    # measurements in one column: on dSprites the single-judge baseline is
    # 0.6821 while the consensus feature_unlearn is 0.1743, and most of that
    # apparent 3.9x gap is the judging protocol rather than the model.  Caught
    # live -- this exact mix shipped in one build of the dSprites table before
    # the guard existed.  A column that silently mixes protocols is the same
    # class of error as the R1 retraction: correctly formatted, confidently
    # labelled, and not measuring one thing.
    if protocol == "consensus":
        for m, (mean, sd, n) in read_adversarial_replicates(dataset, cfg["adv_r"]).items():
            for r in rows:
                if r["method"] == m:
                    r["agg"]["adv_at_r"] = (mean, sd, n)
        adv_note = "consensus-judged"
        # `n` in the table counts LATENT-metric runs. The attack column can have
        # a different n -- the closed-form erasure methods have no training-seed
        # axis, so their attack replicates come from refitting on replicate
        # BASELINES instead (E13). Leaving that implicit would put a "+-" next
        # to an "n=1" row and look like a contradiction, so each row states its
        # own attack n when the two disagree.
        for m, (_mean, _sd, _n) in read_adversarial_replicates(dataset, cfg["adv_r"]).items():
            for r in rows:
                if r["method"] == m and _n != r["n_seeds"] and _n > 1:
                    r["note"] = ((r["note"] + "; ") if r["note"] else "") +                                 f"attack@r n={_n} (refit on replicate baselines)"
    else:
        adv_note = (f"{protocol}-judge only -- consensus file missing, so the "
                    f"model-level error bars from E13 are NOT merged in "
                    f"(they are consensus-judged and would not be comparable)")

    return {"dataset": dataset, "baseline_val": baseline_val, "budget": budget,
            "adv_r": cfg["adv_r"], "adv_protocol": protocol,
            "adv_note": adv_note, "rows": rows}


def render(table: dict) -> str:
    ds = table["dataset"]
    lines = [
        f"### {ds} — cross-method master table",
        "",
        f"Budget {table['budget']:.2f} (baseline {table['baseline_val']:.2f} + 10%). "
        f"Champions selected by the shared trade-off rule with `require_exact=True`. "
        f"`n` is the number of training seeds aggregated; `±` is the sample sd across them. "
        f"`attack@r` is the ball-constrained adversarial-recovery success within "
        f"L2 radius {table.get('adv_r')} (lower = harder to attack); it is one point "
        f"off a curve that CROSSES, so the radius is part of the number. "
        f"**Judging protocol: {table.get('adv_note')}.** Its `±` "
        f"is over training-seed replicates of the champion where those exist, and "
        f"is therefore a MODEL-level error bar (E13) — distinct from the "
        f"judge-level spread in E4/E9, which is larger and is reported separately.",
        "",
    ]
    cols = [c for c in METRIC_CFG]
    header = "| method | n | " + " | ".join(name for _, name, _ in cols) + " | notes |"
    sep = "|---|---|" + "---|" * len(cols) + "---|"
    lines += [header, sep]
    for r in table["rows"]:
        cells = []
        for key, _, _ in cols:
            if key in r["agg"]:
                mean, sd, n = r["agg"][key]
                places = 2 if key in ("val", "forget") else 4
                cells.append(fmt(mean, sd, n, places))
            else:
                cells.append("—")
        lines.append(f"| {r['method']} | {r['n_seeds']} | " + " | ".join(cells)
                     + f" | {r['note']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsprites", "celeba", "mpi3d", "all", "both"],
                    default="all")
    ap.add_argument("--out", type=str, default="docs/MASTER_TABLE.md")
    args = ap.parse_args()
    # "both" kept as a backwards-compatible alias for the original two datasets.
    if args.dataset in ("all", "both"):
        names = ["dsprites", "celeba"] + (["mpi3d"] if args.dataset == "all" else [])
    else:
        names = [args.dataset]

    blocks, tables = [], {}
    for n in names:
        t = build(n)
        tables[n] = t
        blocks.append(render(t))
        print(render(t))
        print()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Cross-method master table\n\n"
        "Generated by `scripts/build_master_table.py` — do not hand-edit, it is "
        "regenerated as runs land.\n\n" + "\n\n".join(blocks) + "\n",
        encoding="utf-8",
    )
    print(f"-> {out.relative_to(ROOT)}")
    (ROOT / "results/master_table.json").write_text(json.dumps(tables, indent=2, default=str))


if __name__ == "__main__":
    main()
