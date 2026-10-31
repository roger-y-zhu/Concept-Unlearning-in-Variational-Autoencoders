"""
experiments/unlearn/tune.py
===========================
Optuna multi-objective hyperparameter search for any unlearning method on any
dataset.

Replaces experiments/unlearn/{dsprites,celeba}/optuna_tune.py, which each
reimplemented the training loop for every method -- the copy that let the
feature-unlearning BatchNorm bug survive in four places at once.  Training now
goes through the same src/unlearning/loop.py that experiments/unlearn/run.py
uses, so a fix lands everywhere.

Objectives (optimised simultaneously, NSGA-II builds the Pareto front)
    minimise  retain ELBO   -- model still reconstructs the retain set
    maximise  forget ELBO   -- model has "unlearned" the forget concept

Both are scored on HELD-OUT data (DatasetSpec.eval_loaders).  Scoring on the
training split makes the retain objective reward overfitting instead of
retention.

Usage
-----
    python experiments/unlearn/tune.py --dataset dsprites --method neg_grad --n-trials 40
    python experiments/unlearn/tune.py --dataset celeba  --method rmu      --n-trials 20
    # resume / run parallel workers:
    python experiments/unlearn/tune.py ... --storage sqlite:///tune.db --study-name ng_v1

Results
-------
    results/<dataset>/<dataset>_<method>_optuna/
        trial_NNN/final.pt
        trial_NNN/metrics.json
        study_results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import torch
from optuna.samplers import NSGAIISampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.unlearning.loop import load_baseline
from src.unlearning.methods import METHODS
from src.unlearning.spec import DATASETS, SEED
from src.utils.seed import seed_all

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%m-%d %I:%M:%S %p")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search spaces  --  one place, keyed by (dataset, method)
# ---------------------------------------------------------------------------
# Ranges reflect measured collapse boundaries:
#   dsprites neg_grad  collapses above lambda ~= 1e-3
#   dsprites feature_unlearn: alpha=1 -> retain val ~50, alpha=10 -> ~193
#     (baseline 36.9), so the old alpha<=100 range was mostly wasted trials.

SEARCH_SPACES = {
    ("dsprites", "neg_grad"):        {"lambda_forget": ("float", 1e-4, 5e-3, True),
                                      "epochs":        ("int",   4,    20,   False)},
    ("dsprites", "finetune"):        {"lambda_forget": ("float", 5e-5, 5e-4, True),
                                      "epochs":        ("int",   30,   150,  False)},
    # Moon et al. report alpha = 3 (MNIST VAE) and 300 (CelebA VDVAE); loss
    # normalisation differs so those are not directly transferable, but the old
    # 20 ceiling was well below anything they tested.
    ("dsprites", "feature_unlearn"): {"alpha":         ("float", 0.25, 100.0, True),
                                      "epochs":        ("int",   3,    30,   False)},
    # alpha (retain anchor) dominates: alpha=1 destroys the model, alpha>=100
    # leaves it unchanged.  c is swept over a narrow band around the latent scale.
    # epochs is capped at 3 because every sweep run peaks at epoch 1 and decays
    # after -- the reference implementation budgets 80-150 batches TOTAL, so an
    # epochs 3-30 range spent every trial deep in the overtrained regime.
    ("dsprites", "rmu"):             {"alpha":         ("float", 5.0,  150.0, True),
                                      "c":             ("float", 0.5,  5.0,   True),
                                      "epochs":        ("int",   1,    3,     False)},

    ("celeba",   "neg_grad"):        {"lambda_forget": ("float", 5e-3, 3e-2, True),
                                      "epochs":        ("int",   10,   80,   False)},
    ("celeba",   "finetune"):        {"lambda_forget": ("float", 5e-3, 5e-2, True),
                                      "epochs":        ("int",   10,   60,   False)},
    ("celeba",   "feature_unlearn"): {"alpha":         ("float", 0.25, 100.0, True),
                                      "epochs":        ("int",   3,    30,   False)},
    ("celeba",   "rmu"):             {"alpha":         ("float", 5.0,  150.0, True),
                                      "c":             ("float", 0.5,  5.0,   True),
                                      "epochs":        ("int",   1,    3,     False)},
}


def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuning for unlearning methods")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--method",  choices=sorted(METHODS),  required=True)
    p.add_argument("--n-trials",    type=int,   default=30)
    p.add_argument("--lr",          type=float, default=5e-5)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--num-workers", type=int,   default=None)
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--storage",     type=str,   default=None,
                   help="Optuna storage URL, e.g. sqlite:///tune.db")
    p.add_argument("--study-name",  type=str,   default=None)
    p.add_argument("--out-dir",     type=str,   default=None)
    return p.parse_args()


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        kind, lo, hi, use_log = spec
        if kind == "float":
            out[name] = trial.suggest_float(name, lo, hi, log=use_log)
        else:
            out[name] = trial.suggest_int(name, lo, hi)
    return out


@torch.no_grad()
def avg_elbo(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        out = model(x, kl_weight=1.0)
        total += out.loss.item() * x.size(0)
        n     += x.size(0)
    return total / n if n else float("nan")


def find_cached(out_dir: Path, hparams: dict):
    """Reuse a previous trial with identical hyperparameters, if one exists.

    Only accepts entries scored on held-out data -- older trials recorded
    train-split scores and mixing the two would corrupt the front.
    """
    for mfile in sorted(out_dir.glob("trial_*/metrics.json")):
        try:
            m = json.loads(mfile.read_text())
        except Exception:
            continue
        if m.get("scored_on") != "held-out":
            continue
        if all(k in m and abs(m[k] - v) <= 1e-9 + 1e-6 * abs(v)
               for k, v in hparams.items()):
            return m["retain_elbo"], m["forget_elbo"]
    return None


def main():
    args = parse_args()
    seed_all(SEED)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    spec   = DATASETS[args.dataset]
    method = METHODS[args.method]
    space  = SEARCH_SPACES.get((args.dataset, args.method))
    if space is None:
        sys.exit(f"No search space defined for ({args.dataset}, {args.method}). "
                 f"Add one to SEARCH_SPACES in {__file__}.")

    out_dir = Path(args.out_dir) if args.out_dir else spec.optuna_dir(args.method)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loaders are built once and shared across trials.
    forget_loader, retain_loader = spec.concept_loaders(args.num_workers)
    eval_forget, eval_retain     = spec.eval_loaders(args.num_workers)
    log.info("Train | forget %d  retain %d batches", len(forget_loader), len(retain_loader))
    log.info("Eval  | forget %d  retain %d batches  (held-out)",
             len(eval_forget), len(eval_retain))

    def objective(trial):
        hparams = suggest(trial, space)

        cached = find_cached(out_dir, hparams)
        if cached is not None:
            r, f = cached
            log.info("Trial %3d | %s | CACHED retain=%.2f forget=%.2f",
                     trial.number, "  ".join(f"{k}={v:.5g}" for k, v in hparams.items()), r, f)
            return r, f

        # Fresh model + frozen reference for every trial.
        model, frozen, _ = load_baseline(spec, device)

        trial_args = argparse.Namespace(
            lr=args.lr, grad_clip=args.grad_clip, num_workers=args.num_workers,
            save_every=0, out_dir=None, **hparams,
        )
        ctx       = method.prepare(model, frozen, forget_loader, retain_loader,
                                   device, trial_args)
        trainable = method.trainable_params(model)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

        t0 = time.time()
        from itertools import cycle
        import torch.nn as nn
        for _ in range(hparams["epochs"]):
            model.train(method.train_mode)
            for rb, fb in zip(retain_loader, cycle(forget_loader)):
                x_r = (rb[0] if isinstance(rb, (list, tuple)) else rb).to(device)
                x_f = (fb[0] if isinstance(fb, (list, tuple)) else fb).to(device)
                loss, _ = method.step(model, frozen, ctx, x_r, x_f, trial_args)
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()

        retain_elbo = avg_elbo(model, eval_retain, device)
        forget_elbo = avg_elbo(model, eval_forget, device)
        elapsed     = time.time() - t0

        metrics = {"trial": trial.number, "retain_elbo": retain_elbo,
                   "forget_elbo": forget_elbo, "elapsed_s": elapsed,
                   "scored_on": "held-out", **hparams}

        tdir = out_dir / f"trial_{trial.number:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": model.cfg,
                    "metrics": metrics, "trial": trial.number}, tdir / "final.pt")
        (tdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        log.info("Trial %3d | %s | retain=%.2f forget=%.2f | %.0fs",
                 trial.number, "  ".join(f"{k}={v:.5g}" for k, v in hparams.items()),
                 retain_elbo, forget_elbo, elapsed)
        return retain_elbo, forget_elbo

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=NSGAIISampler(seed=SEED),
        storage=args.storage,
        study_name=args.study_name or f"{args.dataset}_{args.method}",
        load_if_exists=True,
    )

    log.info("=" * 66)
    log.info("Optuna | %s | %s | n_trials=%d | device=%s",
             args.dataset, args.method, args.n_trials, device)
    log.info("  search space: %s", ", ".join(space))
    log.info("=" * 66)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    pareto = sorted(study.best_trials, key=lambda t: t.values[0])
    log.info("\nPareto front (%d points), held-out scores:", len(pareto))
    log.info("  %-6s %-12s %-12s %s", "trial", "retain", "forget", "hparams")
    for t in pareto:
        log.info("  %-6d %-12.4f %-12.4f %s", t.number, t.values[0], t.values[1],
                 "  ".join(f"{k}={v:.5g}" for k, v in t.params.items()))

    (out_dir / "study_results.json").write_text(json.dumps({
        "dataset": args.dataset, "method": args.method,
        "scored_on": "held-out", "n_trials": args.n_trials,
        "pareto": [{"trial": t.number, "retain_elbo": t.values[0],
                    "forget_elbo": t.values[1], **t.params} for t in pareto],
        "all_trials": [{"trial": t.number, "retain_elbo": t.values[0],
                        "forget_elbo": t.values[1], "state": t.state.name, **t.params}
                       for t in study.trials if t.values is not None],
    }, indent=2))
    log.info("Study results -> %s", out_dir / "study_results.json")


if __name__ == "__main__":
    main()
