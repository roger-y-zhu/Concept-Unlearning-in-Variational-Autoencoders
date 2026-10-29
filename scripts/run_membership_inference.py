"""
scripts/run_membership_inference.py
=====================================
Population loss-threshold MIA (src/evaluation/membership.py) across the
dSprites champions. dSprites-only -- see that module's docstring for why
CelebA's forget set has no genuine held-out non-member split.

Members    : TRAIN-split hearts (concept_col=1, concept_val=2, split="train")
              -- in baseline's original training set AND every method's
              forget_loader during unlearning.
Non-members: TEST-split hearts, split="test" -- never in any model's
              gradient updates, for baseline OR the oracle OR any method.

No new training: every checkpoint here already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.membership import per_sample_negative_elbo, loss_threshold_mia  # noqa: E402
from src.unlearning.spec import val_budget, METHODS_FORGET_BETTER  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint, collect_method_champions  # noqa: E402
from src.data.dsprites import get_concept_loaders  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 48025845


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", action="store_true",
                    help="also score the oracle/RMU --train-seed replicates")
    args = ap.parse_args()

    npz = str(ROOT / "data/dsprites/dsprites.npz")
    member_loader, _ = get_concept_loaders(
        npz_path=npz, concept_col=1, concept_val=2, batch_size=512,
        num_workers=0, seed=SEED, split="train",
    )
    nonmember_loader, _ = get_concept_loaders(
        npz_path=npz, concept_col=1, concept_val=2, batch_size=512,
        num_workers=0, seed=SEED, split="test",
    )
    print(f"members (train hearts): {len(member_loader.dataset)}  "
          f"non-members (test hearts): {len(nonmember_loader.dataset)}")

    models = {
        "baseline": ROOT / "results/dsprites/dsprites_baseline/epoch_0150.pt",
        "oracle": ROOT / "results/dsprites/dsprites_oracle/epoch_0150.pt",
    }
    budget = val_budget("dsprites")
    for method, champ in collect_method_champions(
            "dsprites", METHODS_FORGET_BETTER, budget, require_exact=True).items():
        models[method] = Path(champ["ckpt"]).resolve()

    if args.replicates:
        for ts in (1, 2, 3):
            p = ROOT / f"results/dsprites/dsprites_oracle_ts{ts}/epoch_0150.pt"
            if p.exists():
                models[f"oracle_ts{ts}"] = p
        for ts in (1, 2, 3, 4):
            p = ROOT / f"results/dsprites/dsprites_rmu/a200.0_c1.0_ep20_s5_ts{ts}/epoch_0011.pt"
            if p.exists():
                models[f"rmu_ts{ts}"] = p

    results = {}
    print(f"\n{'model':<18}{'MIA AUROC':>11}{'member loss':>13}{'nonmember loss':>16}")
    for name, path in models.items():
        model, _ = load_vae_checkpoint(path, DEVICE)
        m_loss = per_sample_negative_elbo(model, member_loader, DEVICE)
        nm_loss = per_sample_negative_elbo(model, nonmember_loader, DEVICE)
        r = loss_threshold_mia(m_loss, nm_loss)
        results[name] = r
        print(f"{name:<18}{r['auroc']:>11.4f}{r['member_loss_mean']:>13.3f}"
              f"{r['nonmember_loss_mean']:>16.3f}")

    out = ROOT / "results/dsprites/membership_inference.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
