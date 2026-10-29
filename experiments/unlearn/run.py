"""
experiments/unlearn/run.py
==========================
Single entry point for every unlearning method on every dataset.

Replaces experiments/unlearn/{dsprites,celeba}/{neg_grad,finetune,feature_unlearn}.py
-- six near-identical scripts whose shared training loop had drifted apart.

Usage
-----
    python experiments/unlearn/run.py --dataset dsprites --method neg_grad \
        --lambda-forget 0.00029 --epochs 13

    python experiments/unlearn/run.py --dataset celeba --method feature_unlearn \
        --alpha 1 --epochs 10 --elbo-anchor 1

    python experiments/unlearn/run.py --dataset dsprites --method rmu \
        --c 1.0 --alpha 1.0 --epochs 10

    python experiments/unlearn/run.py --list        # show datasets and methods

Output paths are unchanged from the old scripts:
    results/<dataset>/<dataset>_<method>/<run_name>/{best,final,epoch_NNNN}.pt

Re-running a completed configuration exits 0 without recomputing (final.pt is
the done-marker), so sweeps are safe to restart.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.unlearning.loop import run_unlearning
from src.unlearning.methods import METHODS
from src.unlearning.spec import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%m-%d %I:%M:%S %p",
)
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a concept-unlearning method on a VAE baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Method-specific flags are added by --method; run --list to see them.",
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), required=False)
    p.add_argument("--method",  choices=sorted(METHODS),  required=False)
    p.add_argument("--list", action="store_true",
                   help="List available datasets and methods, then exit")

    # Shared across every method.
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--save-every",  type=int,   default=1,
                   help="Save a checkpoint every N epochs (0 = only final.pt)")
    p.add_argument("--max-batches", type=int,   default=0,
                   help="Cap optimiser steps per epoch (0 = full pass). RMU's "
                        "reference implementation uses 80-150 TOTAL, versus 768 "
                        "(dSprites) / 1331 (CelebA) in one full epoch here. "
                        "Appends _b<N> to the run directory.")
    p.add_argument("--val-tolerance", type=float, default=0.10,
                   help="Retain val loss may exceed the baseline by this fraction "
                        "before a checkpoint stops being eligible for "
                        "best_tradeoff.pt (default 0.10 = 10%%)")
    p.add_argument("--train-seed", type=int, default=None,
                   help="Seed for TRAINING stochasticity -- shuffle order and the "
                        "VAE's reparameterisation noise. Default None = the fixed "
                        "spec.SEED every run has used so far, so omitting this "
                        "reproduces existing results exactly. Appends _ts<N> to the "
                        "run directory. NOTE this is a different axis from the "
                        "method-level --u-seed/--fisher-seed/--inlp-seed, which "
                        "reseed only that method's own random vector and leave "
                        "training identical; the data SPLIT is always spec.SEED and "
                        "is deliberately not varied, or runs stop being comparable.")
    p.add_argument("--baseline-ckpt", type=str, default=None,
                   help="Start from this checkpoint instead of the spec's "
                        "canonical baseline. Used to refit the closed-form "
                        "erasure methods (leace/inlp) on replicate baselines "
                        "so they can carry an error bar; appends _bl<tag> to "
                        "the run name.")
    p.add_argument("--num-workers", type=int,   default=None,
                   help="DataLoader workers (default: per-dataset)")
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--out-dir",     type=str,   default=None,
                   help="Override the output directory entirely")
    return p


def show_listing() -> None:
    print("\nDatasets")
    print("-" * 66)
    for name, spec in sorted(DATASETS.items()):
        print(f"  {name:10s} concept={spec.concept_desc:20s} batch={spec.batch_size}")
    print("\nMethods")
    print("-" * 66)
    for name, m in sorted(METHODS.items()):
        sub = argparse.ArgumentParser(add_help=False)
        m.add_args(sub)
        flags = "  ".join(a.option_strings[0] for a in sub._actions)
        print(f"  {name:16s} {m.description}")
        print(f"  {'':16s}   flags: {flags}")
        # "compares vs frozen ref" is the reason for eval mode in Feature
        # Unlearning and RMU, but NOT in Fisher scrubbing, which has no
        # reference model -- it needs eval mode because its FIM pass runs at
        # batch size 1.  Don't state a reason that isn't this method's.
        if m.train_mode:
            bn = "updating (train mode)"
        elif getattr(m, "one_shot", False):
            bn = "frozen (one-shot; no reference model)"
        else:
            bn = "frozen (compares vs frozen ref)"
        print(f"  {'':16s}   encoder BatchNorm: {bn}")
    print()


def main() -> None:
    parser = build_parser()
    known, _ = parser.parse_known_args()

    if known.list or not (known.dataset and known.method):
        if known.list:
            show_listing()
            return
        parser.error("--dataset and --method are required (or use --list)")

    method = METHODS[known.method]
    spec   = DATASETS[known.dataset]

    # Let the method contribute its own hyperparameters, then reparse.
    method.add_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = run_unlearning(spec, method, args, device)
    if out_dir is None:
        sys.exit(0)


if __name__ == "__main__":
    main()
