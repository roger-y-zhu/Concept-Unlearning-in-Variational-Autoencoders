"""
scripts/run_disentanglement.py
================================
MIG and DCI-Disentanglement on the dSprites champions, over the factors that
were never the unlearning target (scale, orientation, x, y -- shape is the
forget concept, color is constant across the dataset).

WHY
---
docs/EVALUATION_LANDSCAPE.md's gap analysis: no unlearning-evaluation suite
surveyed asks whether an intervention damages disentanglement of factors it
was never aimed at. dSprites is the standard benchmark for these metrics and
this project already has the ground-truth ready; this is close to a free
question to answer.

`encode_dataset` (src/evaluation/probing.py) keeps only the shape column of
latents_classes, so the cached `encodings_test.npz` does not carry what this
script needs -- it re-encodes to also capture scale/orientation/x/y. This is a
small extra cost (encoding-only, no probe fits) run once per champion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.disentanglement import mutual_info_gap, dci_disentanglement  # noqa: E402
from src.unlearning.spec import val_budget, METHODS_FORGET_BETTER  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint, collect_method_champions  # noqa: E402
from src.data.dsprites import get_dataloaders  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 48025845
FACTOR_COLS = {"scale": 2, "orientation": 3, "x": 4, "y": 5}


@torch.no_grad()
def encode_with_factors(model, loader) -> tuple[np.ndarray, dict]:
    model.eval()
    mus, factor_lists = [], {k: [] for k in FACTOR_COLS}
    for batch in loader:
        imgs, _, cls = batch
        mu, _ = model.encode(imgs.to(DEVICE))
        mus.append(mu.cpu().numpy())
        cls_np = cls.numpy()
        for name, col in FACTOR_COLS.items():
            factor_lists[name].append(cls_np[:, col])
    mu = np.concatenate(mus, axis=0)
    factors = {k: np.concatenate(v).astype(np.float64) for k, v in factor_lists.items()}
    return mu, factors


@torch.no_grad()
def encode_mu(model, loader) -> np.ndarray:
    """Just the latent means, in loader order (factors handled separately)."""
    model.eval()
    mus = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        mu, _ = model.encode(imgs.to(DEVICE))
        mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def run_mpi3d(args):
    """MIG/DCI on MPI3D-real over the six factors that are NOT the forget target.

    MPI3D's unique value-add (cf. ledger E11 / MO3): unlike CelebA it has full
    ground-truth factor labels, so it can answer whether unlearning object_color
    damages disentanglement of the factors it never targeted (shape, size, camera
    height, background colour, the two DOF axes). Factors come from
    ``load_factor_matrix`` rather than a batch column; alignment with the encoder
    pass is guaranteed by driving BOTH from the same test Subset.
    """
    from torch.utils.data import DataLoader, Subset  # noqa: E402
    from src.data.mpi3d import (  # noqa: E402
        MPI3DDataset, _splits, load_factor_matrix, FACTOR_NAMES, DEFAULT_FORGET_FACTOR,
    )

    npz = str(ROOT / "data/mpi3d/mpi3d_real.npz")
    forget_name = FACTOR_NAMES[DEFAULT_FORGET_FACTOR]  # "object_color"

    dataset = MPI3DDataset(npz, include_labels=True)
    _, _, test_sub = _splits(len(dataset), 0.1, 0.1, SEED)
    test_indices = np.array(list(test_sub))           # global indices, split order
    # One Subset object drives the loader; the same indices drive the factors.
    test_loader = DataLoader(Subset(dataset, list(test_sub)),
                             batch_size=256, shuffle=False, num_workers=0)

    all_factors = load_factor_matrix(npz, test_indices)   # {name: (Ntest,)}
    factors = {k: v.astype(np.float64) for k, v in all_factors.items()
               if k != forget_name}                        # drop the forget target
    print(f"MPI3D test samples: {len(test_indices)} | "
          f"factors scored: {list(factors)}")

    models = {
        "baseline": ROOT / "results/mpi3d/mpi3d_baseline/epoch_0120.pt",
        "oracle": ROOT / "results/mpi3d/mpi3d_oracle/epoch_0120.pt",
    }
    budget = val_budget("mpi3d")
    for method, champ in collect_method_champions(
            "mpi3d", METHODS_FORGET_BETTER, budget).items():
        models[method] = Path(champ["ckpt"]).resolve()

    if args.replicates:
        for ts in (1, 2, 3):
            for tag, rel in ((f"baseline_ts{ts}", f"results/mpi3d/mpi3d_baseline_ts{ts}/epoch_0120.pt"),
                             (f"oracle_ts{ts}",   f"results/mpi3d/mpi3d_oracle_ts{ts}/epoch_0120.pt")):
                p = ROOT / rel
                if p.exists():
                    models[tag] = p

    results = {}
    print(f"{'model':<18}{'MIG':>8}{'DCI':>8}   per-factor MIG")
    for name, path in models.items():
        if not Path(path).exists():
            print(f"{name:<18}  (missing ckpt, skipped)"); continue
        model, _ = load_vae_checkpoint(path, DEVICE)
        mu = encode_mu(model, test_loader)
        mig = mutual_info_gap(mu, factors, n_bins=args.n_bins)
        dci = dci_disentanglement(mu, factors, n_estimators=args.n_estimators)
        results[name] = {"mig": mig, "dci": dci, "ckpt": str(path)}
        per_f = "  ".join(f"{k}={v['mig']:.3f}" for k, v in mig["per_factor"].items())
        print(f"{name:<18}{mig['mig']:>8.4f}{dci['disentanglement']:>8.4f}   {per_f}")

    out = ROOT / "results/mpi3d/disentanglement.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsprites", "mpi3d"], default="dsprites")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--replicates", action="store_true")
    args = ap.parse_args()

    if args.dataset == "mpi3d":
        run_mpi3d(args)
        return

    _, _, test_loader = get_dataloaders(
        npz_path=str(ROOT / "data/dsprites/dsprites.npz"), batch_size=512,
        val_fraction=0.1, test_fraction=0.1, num_workers=0, seed=SEED,
        include_labels=True,
    )

    models = {
        "baseline": ROOT / "results/dsprites/dsprites_baseline/epoch_0150.pt",
        "oracle": ROOT / "results/dsprites/dsprites_oracle/epoch_0150.pt",
    }
    budget = val_budget("dsprites")
    for method, champ in collect_method_champions(
            "dsprites", METHODS_FORGET_BETTER, budget, require_exact=True).items():
        models[method] = Path(champ["ckpt"]).resolve()

    if args.replicates:
        # Oracle and RMU/fisher (the two methods with the most striking single-run
        # numbers) already have --train-seed replicates on disk from queue24/27 --
        # this is a pure encode pass, no new training, so replication is nearly
        # free here even though it is expensive everywhere else in the project.
        for ts in (1, 2, 3):
            p = ROOT / f"results/dsprites/dsprites_oracle_ts{ts}/epoch_0150.pt"
            if p.exists():
                models[f"oracle_ts{ts}"] = p
        for ts in (1, 2, 3, 4):
            p = ROOT / f"results/dsprites/dsprites_rmu/a200.0_c1.0_ep20_s5_ts{ts}/epoch_0011.pt"
            if p.exists():
                models[f"rmu_ts{ts}"] = p

        # BASELINE replicates (queue28). Without these, E11's oracle-vs-baseline
        # comparison had n=4 on one side and n=1 on the other, so it could only
        # ask "does the baseline fall inside the oracle's range" -- which is not
        # a test. These are the missing half, and they matter more here than
        # anywhere else in the project: from-scratch training variance on
        # MIG/DCI is ~213x the fine-tuning variance that the METHODS' _ts
        # replicates measure (E5 scope correction), so a single baseline draw is
        # a particularly weak reference point for exactly this metric.
        for ts in (1, 2, 3):
            p = ROOT / f"results/dsprites/dsprites_baseline_ts{ts}/epoch_0150.pt"
            if p.exists():
                models[f"baseline_ts{ts}"] = p

    results = {}
    print(f"{'model':<18}{'MIG':>8}{'DCI':>8}   per-factor MIG")
    for name, path in models.items():
        model, _ = load_vae_checkpoint(path, DEVICE)
        mu, factors = encode_with_factors(model, test_loader)
        mig = mutual_info_gap(mu, factors, n_bins=args.n_bins)
        dci = dci_disentanglement(mu, factors, n_estimators=args.n_estimators)
        results[name] = {"mig": mig, "dci": dci, "ckpt": str(path)}
        per_f = "  ".join(f"{k}={v['mig']:.3f}" for k, v in mig["per_factor"].items())
        print(f"{name:<18}{mig['mig']:>8.4f}{dci['disentanglement']:>8.4f}   {per_f}")

    out = ROOT / "results/dsprites/disentanglement.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
