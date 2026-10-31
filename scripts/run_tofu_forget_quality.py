"""
scripts/run_tofu_forget_quality.py
==================================
A TOFU-style Forget Quality metric (Maini et al. 2024, arXiv:2401.06121;
papers/summaries/tofu2024.md), adapted to the VAE concept-unlearning setting, and
added to every dense operating point so it enters the incremental-validity study
(ledger D1/E18) as a modern SOTA *oracle-distance* criterion.

TOFU defines Forget Quality as the p-value of a two-sample Kolmogorov-Smirnov test
comparing a per-example likelihood statistic (the "truth ratio") on the FORGET set
between the unlearned model and a RETRAINED reference. High p = the unlearned
model's forget-set behaviour is statistically indistinguishable from the model that
never saw the concept = "successful unlearning" by the oracle-distance criterion.

VAE-faithful transfer: the per-example statistic is the per-sample negative ELBO
(reconstruction + KL, decoded from mu; src/evaluation/membership.py) on the FORGET
set (concept-positive images). The reference is our trained oracle. So

    forget_quality(model) = KS_2samp( negELBO_forget(model), negELBO_forget(oracle) ).pvalue

This is an oracle-distance criterion in exactly TOFU's sense (indistinguishability
from the retrained reference via a KS test on a per-example likelihood), differing
only in the per-example statistic (ELBO vs an LLM QA truth-ratio). We ALSO record
the KS statistic (a distance, higher = further from oracle). Both are written back
into results/<ds>/dense_recovery.json as `forget_quality` and `tofu_ks`.

Note (in TOFU too, their Sec 5.2): a model that outputs gibberish can score high
Forget Quality without genuine unlearning -- the same proxy-validity gap E18 shows
for geometry. Whether this metric tracks the attack criterion is the question E18
puts to it.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, Subset  # noqa: E402
from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.membership import per_sample_negative_elbo  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORGET_CAP = 3000   # forget-set images for the KS test; deterministic subset

CKPT = {
    "dsprites": dict(baseline="results/dsprites/dsprites_baseline/epoch_0150.pt",
                     oracle="results/dsprites/dsprites_oracle/epoch_0150.pt"),
    "celeba":   dict(baseline="results/celeba/celeba_baseline/epoch_0250.pt",
                     oracle="results/celeba/celeba_oracle/epoch_0250.pt"),
    "mpi3d":    dict(baseline="results/mpi3d/mpi3d_baseline/epoch_0120.pt",
                     oracle="results/mpi3d/mpi3d_oracle/epoch_0120.pt"),
    # beta=4 variants (poster refresh 2026-10-16): same data / forget set, own beta=4 oracle.
    "dsprites_b4": dict(baseline="results/dsprites_b4/dsprites_b4_baseline/epoch_0150.pt",
                        oracle="results/dsprites_b4/dsprites_b4_oracle/epoch_0150.pt"),
    "celeba_b4":   dict(baseline="results/celeba_b4/celeba_b4_baseline/epoch_0250.pt",
                        oracle="results/celeba_b4/celeba_b4_oracle/epoch_0250.pt"),
    "mpi3d_b4":    dict(baseline="results/mpi3d_b4/mpi3d_b4_baseline/epoch_0120.pt",
                        oracle="results/mpi3d_b4/mpi3d_b4_oracle/epoch_0120.pt"),
}


def forget_loader(dataset: str) -> DataLoader:
    """Deterministic, capped loader of CONCEPT-POSITIVE (forget) test images."""
    dataset = dataset.removesuffix("_b4")      # beta=4 variants use the identical data
    if dataset == "dsprites":
        from src.data.dsprites import get_concept_loaders
        f, _ = get_concept_loaders(npz_path=str(ROOT / "data/dsprites/dsprites.npz"),
                                   concept_col=1, concept_val=2, batch_size=256,
                                   num_workers=0, split="test")
    elif dataset == "celeba":
        from src.data.celeba import get_concept_loaders, EYEGLASSES_COL
        f, _ = get_concept_loaders(data_root=str(ROOT / "data/celeba/img_align_celeba"),
                                   forget_attr=EYEGLASSES_COL, batch_size=256, num_workers=0)
    elif dataset == "mpi3d":
        from src.data.mpi3d import get_concept_loaders
        f, _ = get_concept_loaders(npz_path=str(ROOT / "data/mpi3d/mpi3d_real.npz"),
                                   forget_factor=0, forget_val=2, batch_size=256,
                                   num_workers=0, split="test")
    else:
        raise ValueError(dataset)
    ds = f.dataset
    n = min(FORGET_CAP, len(ds))
    return DataLoader(Subset(ds, range(n)), batch_size=256, shuffle=False)


def row_ckpt(dataset: str, row: dict) -> Path:
    m, run = row["method"], row["run"]
    if m == "baseline":
        return ROOT / CKPT[dataset]["baseline"]
    if m == "oracle":
        return ROOT / CKPT[dataset]["oracle"]
    return ROOT / f"results/{dataset}/{dataset}_{m}/{run}/final.pt"


# Bump when the TOFU computation changes. The skip below is version-aware so a
# code change forces recompute instead of being silently skipped as "already
# present" (the stale-on-code-change cache class).
TOFU_VERSION = 1


def run_dataset(dataset: str) -> None:
    agg = ROOT / f"results/{dataset}/dense_recovery.json"
    if not agg.exists():
        print(f"  {dataset}: no dense_recovery.json, skipping"); return
    rows = json.loads(agg.read_text())
    if rows and all(r.get("_tofu_v") == TOFU_VERSION for r in rows):
        print(f"  {dataset}: forget_quality up to date (v{TOFU_VERSION}, {len(rows)} rows)"); return

    loader = forget_loader(dataset)
    oracle, _ = load_vae_checkpoint(ROOT / CKPT[dataset]["oracle"], DEVICE)
    elbo_oracle = per_sample_negative_elbo(oracle, loader, DEVICE)
    print(f"\n{dataset}: forget set n={len(elbo_oracle)}  oracle negELBO "
          f"mean={elbo_oracle.mean():.2f}")

    for r in rows:
        ck = row_ckpt(dataset, r)
        if not ck.exists():
            r["forget_quality"], r["tofu_ks"], r["_tofu_v"] = None, None, TOFU_VERSION
            continue
        model, _ = load_vae_checkpoint(ck, DEVICE)
        elbo = per_sample_negative_elbo(model, loader, DEVICE)
        ks = ks_2samp(elbo, elbo_oracle)
        r["forget_quality"] = float(ks.pvalue)      # high = indistinguishable from oracle
        r["tofu_ks"] = float(ks.statistic)          # distance from oracle (0 = identical)
        r["_tofu_v"] = TOFU_VERSION
        print(f"  [{r['method']:<14} {r['run'][:26]:<26}] "
              f"KS={ks.statistic:.3f} forget_quality(p)={ks.pvalue:.3g} "
              f"attack@{r['adv_r']}={r['attack_at_r']:.3f}")

    agg.write_text(json.dumps(rows, indent=2))
    print(f"  -> updated {agg.relative_to(ROOT)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=[
        "dsprites", "celeba", "mpi3d", "dsprites_b4", "celeba_b4", "mpi3d_b4", "all"])
    args = ap.parse_args()
    datasets = (["dsprites", "celeba", "mpi3d", "dsprites_b4", "celeba_b4", "mpi3d_b4"]
                if args.dataset == "all" else [args.dataset])
    for ds in datasets:
        run_dataset(ds)


if __name__ == "__main__":
    main()
