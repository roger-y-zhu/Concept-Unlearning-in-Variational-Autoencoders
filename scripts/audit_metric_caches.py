"""
scripts/audit_metric_caches.py
===============================
Find (and optionally delete) `bespoke_metrics*.json` / `probe_recovery*.json`
caches that may describe a different checkpoint than the analysis reads.

THE ORIGINAL BUG
-----------------
A run directory holds several checkpoints -- best.pt, best_tradeoff.pt,
final.pt, epoch_NNNN.pt -- but each metric file used to be ONE file per
directory. Whichever checkpoint was warmed first won, and every later reader
got those numbers regardless of which checkpoint it resolved.

Found on `dsprites_rmu/a200.0_c1.0_ep20_s5` bespoke_metrics: the cache held
final.pt's metrics (epoch 20, val 46.26 -- 14% OVER budget) while the master
table resolved best_tradeoff.pt (epoch 11, val 40.21). The documented finding
"RMU makes hearts more linearly decodable than baseline" rested entirely on
that mismatch; at the real champion values RMU is indistinguishable from
baseline.

nb_utils.bespoke_metrics_cached and probe_recovery_cached are both now
stem-keyed (`bespoke_metrics_<stem>.json`, `probe_recovery_<stem>.json`) and
stamp `_ckpt`, so a freshly written cache cannot collide across checkpoints.
This script finds pre-fix leftovers: legacy unsuffixed files that predate
stem-keying, still sitting next to the new files, and would be silently
trusted by any reader that does not go through the safe cached functions
(build_master_table.py's readers do; this script is the general-purpose check
for everything else, including future code that might make the same mistake).

VERSION 2 BUG, worth recording so it is not reintroduced
----------------------------------------------------------
An earlier version of this script kept checking the LEGACY unsuffixed path
even after the stem-keying fix landed, so it flagged 120 "suspects" that were
really just runs where the stem-keyed cache for the trade-off checkpoint had
simply never been computed yet -- not wrong, just absent, and any read
recomputes correctly on demand. That false-positive flood nearly buried a REAL
finding surfaced the same session: `probe_recovery_cached` had the identical
directory-not-checkpoint keying bug independently, and it WAS live on the
dSprites RMU champion (see RESEARCH_LEDGER.md's E10 retraction). Fixed here by
checking the checkpoint-specific stem-keyed file first, mirroring exactly how
build_master_table.py's readers resolve a checkpoint's cache -- a MISSING
stem-keyed file is not a suspect, only a WRONG or absent-but-legacy-present one
is.

WHAT COUNTS AS AT RISK
------------------------
A directory is at risk when the trade-off epoch is not the final epoch AND
final.pt is present (two genuinely different models could share a legacy
cache). Directories whose trade-off epoch IS the final epoch are safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.selection import load_history, tradeoff_epoch, resolve_checkpoint  # noqa: E402
from src.unlearning.spec import val_budget, METHODS_FORGET_BETTER  # noqa: E402

BUDGETS = {ds: val_budget(ds) for ds in ("dsprites", "celeba")}
FORGET_BETTER = METHODS_FORGET_BETTER

# metric name -> (stem-keyed prefix, legacy filename)
METRIC_FILES = {
    "bespoke_metrics": ("bespoke_metrics_", "bespoke_metrics.json"),
    "probe_recovery": ("probe_recovery_", "probe_recovery.json"),
}


def _check_metric(cache_dir: Path, prefix: str, legacy_name: str, ckpt_name: str) -> dict | None:
    """None if safe (stem-keyed cache exists and is correct, OR nothing to
    check at all), else a dict describing the risk."""
    stem_file = cache_dir / f"{prefix}{Path(ckpt_name).stem}.json"
    if stem_file.exists():
        return None  # self-identified by filename; trust it (that is the fix)

    legacy = cache_dir / legacy_name
    if not legacy.exists():
        return None  # nothing cached for this checkpoint at all -- not a risk

    try:
        data = json.loads(legacy.read_text())
    except Exception:
        return {"stamped": "CORRUPT", "legacy": legacy}
    stamped = data.get("_ckpt")
    if stamped == ckpt_name:
        return None  # legacy file is correctly stamped for this checkpoint
    return {"stamped": stamped, "legacy": legacy}


def audit() -> list[dict]:
    out = []
    for ds, budget in BUDGETS.items():
        root = ROOT / "results" / ds
        if not root.is_dir():
            continue
        for sweep in sorted(root.iterdir()):
            if not sweep.is_dir() or "PREBNFIX" in sweep.name:
                continue
            method = sweep.name.replace(f"{ds}_", "")
            fb = FORGET_BETTER.get(method)
            if fb is None:
                continue
            for run in sorted(sweep.iterdir()):
                cache_dir = run / "cache"
                if not run.is_dir() or not cache_dir.is_dir():
                    continue
                hist = load_history(run)
                if hist is None:
                    continue
                ep = tradeoff_epoch(hist, fb, budget)
                if ep is None:
                    continue
                n_epochs = len(hist["val_loss"])
                if ep == n_epochs or not (run / "final.pt").exists():
                    continue  # only one distinct model in play -- safe
                ckpt, exact = resolve_checkpoint(run, ep, n_epochs)
                if not exact:
                    continue

                for metric, (prefix, legacy_name) in METRIC_FILES.items():
                    risk = _check_metric(cache_dir, prefix, legacy_name, Path(ckpt).name)
                    if risk is not None:
                        out.append({
                            "run": str(run.relative_to(ROOT)),
                            "metric": metric,
                            "reads": Path(ckpt).name,
                            "tradeoff_epoch": ep,
                            "final_epoch": n_epochs,
                            "stamped": risk["stamped"],
                            "legacy": risk["legacy"],
                        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="Delete the suspect legacy caches so the next warm recomputes them")
    args = ap.parse_args()

    rows = audit()
    by_sweep: dict[str, int] = {}
    for r in rows:
        key = "/".join(Path(r["run"]).parts[1:3])
        by_sweep[key] = by_sweep.get(key, 0) + 1

    print(f"suspect legacy caches: {len(rows)}")
    for k, v in sorted(by_sweep.items(), key=lambda t: -t[1]):
        print(f"   {k:40s} {v}")
    if rows[:5]:
        print("\nexamples:")
        for r in rows[:5]:
            print(f"   {r['run']}  [{r['metric']}]  reads {r['reads']} "
                  f"(ep {r['tradeoff_epoch']} of {r['final_epoch']}), "
                  f"legacy stamped {r['stamped']}")

    if args.delete:
        deleted = set()
        for r in rows:
            p = r["legacy"]
            if p not in deleted and p.exists():
                p.unlink()
                deleted.add(p)
        print(f"\ndeleted {len(deleted)} legacy cache files; re-run the warmers to recompute")
    else:
        print("\n(dry run -- pass --delete to remove them)")


if __name__ == "__main__":
    main()
