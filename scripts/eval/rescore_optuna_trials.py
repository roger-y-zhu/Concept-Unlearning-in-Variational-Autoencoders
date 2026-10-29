"""
scripts/eval/rescore_optuna_trials.py
=====================================
Re-score existing Optuna trial checkpoints on the HELD-OUT test split.

Why this exists
---------------
Earlier runs of experiments/unlearn/*/optuna_tune.py measured both objectives
(retain ELBO, forget ELBO) on the *training* split -- the same data each trial
had just fitted.  The trained checkpoints are fine; only the recorded scores
were misleading, and `_find_cached_metrics` would have reused those stale
values in any later study.

This script reloads every trial_*/final.pt, re-evaluates it on the held-out
split, rewrites metrics.json, and regenerates study_results.json with a fresh
Pareto front.  No retraining -- it salvages the GPU hours already spent.

Usage
-----
    python scripts/eval/rescore_optuna_trials.py --dataset dsprites --method neg_grad
    python scripts/eval/rescore_optuna_trials.py --dataset dsprites --method finetune
    python scripts/eval/rescore_optuna_trials.py --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.vae import ConvVAE, VAEConfig
from src.utils.seed import seed_all

torch.serialization.add_safe_globals([VAEConfig])

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%m-%d %I:%M:%S %p")
log = logging.getLogger(__name__)

SEED = 48025845


def parse_args():
    p = argparse.ArgumentParser(description="Re-score Optuna trials on held-out data")
    p.add_argument("--dataset", choices=["dsprites", "celeba"])
    p.add_argument("--method",  choices=["neg_grad", "finetune", "feature_unlearn"])
    p.add_argument("--all", action="store_true",
                   help="Re-score every *_optuna directory found under results/")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


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


def build_eval_loaders(dataset: str, num_workers: int):
    """Return (forget_loader, retain_loader) drawn from held-out data."""
    if dataset == "dsprites":
        from src.data.dsprites import get_concept_loaders
        return get_concept_loaders(
            npz_path=str(ROOT / "data/dsprites/dsprites.npz"),
            concept_col=1, concept_val=2,
            batch_size=512, num_workers=num_workers, seed=SEED,
            split="test",
        )
    from src.data.celeba import get_concept_eval_loaders
    return get_concept_eval_loaders(
        data_root=str(ROOT / "data/celeba/img_align_celeba"),
        forget_attr="Eyeglasses",
        batch_size=128, num_workers=num_workers, seed=SEED,
    )


def rescore_dir(out_dir: Path, forget_loader, retain_loader, device) -> int:
    trials = sorted(out_dir.glob("trial_*/final.pt"))
    if not trials:
        log.info("  no trials in %s -- skipping", out_dir.name)
        return 0

    log.info("Re-scoring %d trials in %s", len(trials), out_dir.name)
    rows = []
    for ckpt_path in trials:
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = ConvVAE(ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        retain_elbo = avg_elbo(model, retain_loader, device)
        forget_elbo = avg_elbo(model, forget_loader, device)

        mfile = ckpt_path.parent / "metrics.json"
        metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
        old_r = metrics.get("retain_elbo", float("nan"))
        old_f = metrics.get("forget_elbo", float("nan"))

        metrics.update({
            "retain_elbo":       retain_elbo,
            "forget_elbo":       forget_elbo,
            "retain_elbo_train": old_r,   # kept for reference
            "forget_elbo_train": old_f,
            "scored_on":         "held-out",
        })
        mfile.write_text(json.dumps(metrics, indent=2))

        log.info("  %-12s train(%8.2f, %8.2f) -> held-out(%8.2f, %8.2f)",
                 ckpt_path.parent.name, old_r, old_f, retain_elbo, forget_elbo)
        rows.append({k: metrics[k] for k in metrics if k != "elapsed_s"})

    # ── Recompute the Pareto front from the held-out scores ──────────────────
    rows.sort(key=lambda r: r["retain_elbo"])
    pareto, best_f = [], float("-inf")
    for r in rows:
        if r["forget_elbo"] > best_f:
            pareto.append(r)
            best_f = r["forget_elbo"]

    summary = {
        "scored_on":  "held-out",
        "n_trials":   len(rows),
        "pareto":     pareto,
        "all_trials": rows,
    }
    (out_dir / "study_results.json").write_text(json.dumps(summary, indent=2))

    log.info("  Pareto front (%d points), held-out scores:", len(pareto))
    log.info("    %-8s %-12s %-12s  %s", "trial", "retain", "forget", "hparams")
    for r in pareto:
        hp = "  ".join(f"{k}={v:.5g}" for k, v in r.items()
                       if k in ("lambda_forget", "alpha", "epochs"))
        log.info("    %-8s %-12.4f %-12.4f  %s",
                 r.get("trial", "?"), r["retain_elbo"], r["forget_elbo"], hp)
    return len(rows)


def main():
    args = parse_args()
    seed_all(SEED)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.all:
        targets = sorted(p for p in (ROOT / "results").glob("*/*_optuna") if p.is_dir())
    else:
        if not (args.dataset and args.method):
            sys.exit("Give --dataset and --method, or use --all")
        targets = [ROOT / f"results/{args.dataset}/{args.dataset}_{args.method}_optuna"]

    # Group by dataset so loaders are built once each.
    by_dataset: dict[str, list[Path]] = {}
    for t in targets:
        by_dataset.setdefault(t.parent.name, []).append(t)

    total = 0
    for dataset, dirs in by_dataset.items():
        dirs = [d for d in dirs if d.exists()]
        if not dirs:
            continue
        log.info("Building held-out eval loaders for %s ...", dataset)
        fl, rl = build_eval_loaders(dataset, args.num_workers)
        log.info("  forget batches=%d  retain batches=%d", len(fl), len(rl))
        for d in dirs:
            total += rescore_dir(d, fl, rl, device)

    log.info("Done -- %d trials re-scored on held-out data.", total)


if __name__ == "__main__":
    main()
