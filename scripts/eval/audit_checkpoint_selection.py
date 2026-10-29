"""
scripts/eval/audit_checkpoint_selection.py
==========================================
Which completed runs would a trade-off selection rule have picked differently?

best.pt selects on retain val loss alone.  That criterion is monotonically
optimised by not unlearning, so it systematically returns the least-unlearned
epoch of a run.  src/unlearning/loop.py now also writes best_tradeoff.pt -- the
epoch that forgets most while retain val stays within --val-tolerance of the
baseline -- but runs completed before that change have no such file.

This script recomputes the trade-off epoch from history.json (no retraining, no
GPU) and reports where it disagrees with best.pt.  A disagreement means the
analysis for that run was reading the least-unlearned checkpoint.

Runs swept with --save-every 0 only kept best.pt and final.pt, so a disagreeing
run cannot simply be re-read -- it has to be rerun with --save-every 1 to
recover the weights at the trade-off epoch.  The script says which those are.

    python scripts/eval/audit_checkpoint_selection.py
    python scripts/eval/audit_checkpoint_selection.py --val-tolerance 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.selection import tradeoff_epoch
from src.models.vae import VAEConfig
from src.unlearning.methods import METHODS
from src.unlearning.spec import DATASETS

torch.serialization.add_safe_globals([VAEConfig])


def baseline_val(spec) -> float | None:
    try:
        ckpt = torch.load(spec.baseline_ckpt, map_location="cpu", weights_only=True)
        return float(ckpt["metrics"]["loss"])
    except (FileNotFoundError, KeyError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-tolerance", type=float, default=0.10,
                    help="Fraction the retain val loss may exceed the baseline")
    ap.add_argument("--results", type=str, default="results")
    args = ap.parse_args()

    root = Path(args.results)
    total = disagree = no_valid = 0
    rerun: list[str] = []

    for ds_name, spec in sorted(DATASETS.items()):
        base = baseline_val(spec)
        if base is None:
            print(f"\n{ds_name}: baseline checkpoint missing -- skipped")
            continue
        budget = base * (1.0 + args.val_tolerance)

        for m_name, method in sorted(METHODS.items()):
            sweep_dir = root / ds_name / f"{ds_name}_{m_name}"
            runs = sorted(p for p in sweep_dir.glob("*/history.json")) if sweep_dir.exists() else []
            if not runs:
                continue

            print(f"\n{'='*74}")
            print(f"{ds_name} / {m_name}   baseline val {base:.2f}  budget {budget:.2f} "
                  f"(+{args.val_tolerance:.0%})  forget-better={method.forget_better}")
            print(f"{'='*74}")
            print(f"  {'run':32s} {'best.pt':>8s} {'tradeoff':>9s} {'val@tr':>9s}")

            for hp in runs:
                hist = json.loads(hp.read_text())
                if not hist.get("val_loss"):
                    continue
                total += 1
                best_ep = hist["val_loss"].index(min(hist["val_loss"])) + 1
                tr_ep = tradeoff_epoch(hist, method.forget_better, budget)

                if tr_ep is None:
                    no_valid += 1
                    mark, tr_s, val_s = "  no epoch in budget", "--", "--"
                else:
                    val_s = f"{hist['val_loss'][tr_ep-1]:.2f}"
                    tr_s = str(tr_ep)
                    if tr_ep != best_ep:
                        disagree += 1
                        mark = "  <-- DISAGREE"
                        rerun.append(f"{ds_name}/{m_name}/{hp.parent.name} (epoch {tr_ep})")
                    else:
                        mark = ""
                print(f"  {hp.parent.name:32s} {best_ep:>8d} {tr_s:>9s} {val_s:>9s}{mark}")

    print(f"\n{'='*74}")
    print(f"{total} runs | {disagree} where the trade-off epoch differs from best.pt "
          f"| {no_valid} with no epoch inside budget")
    if rerun:
        print("\nThese need rerunning with --save-every 1 to recover the weights at\n"
              "the trade-off epoch (their sweeps used --save-every 0):")
        for r in rerun:
            print(f"  {r}")


if __name__ == "__main__":
    main()
