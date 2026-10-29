"""Direction-robustness of the MPI3D RMU champion across steering-vector seeds.

Every earlier MPI3D RMU number used u_seed=0 (one random steering direction). This
recomputes the headline quantities -- kNN cohesion, dir_norm, linear AUROC, and
consensus attack@8 -- for the champion (a300 c1.0 ep10) at u_seed 0..4, so RMU's
"max cohesion / zero attack resistance / concentrates-not-removes" claim carries a
spread over directions rather than resting on a single draw.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.evaluation.adversarial import ConsensusJudge  # noqa: E402
from src.data.mpi3d import get_labeled_dataloader  # noqa: E402
from notebooks.nb_utils import (  # noqa: E402
    load_vae_checkpoint, encode_dataset_cached, bespoke_metrics_cached, rmu_run_name,
)
from scripts.run_attack_robustness import mpi3d_setup, get_judge, attack, DEVICE  # noqa: E402

REPORT_R = 8.0
SEEDS = [0, 1, 2, 3, 4]
SWEEP = ROOT / "results/mpi3d/mpi3d_rmu"

cfg = mpi3d_setup()
judge = ConsensusJudge([get_judge(cfg, 0, arch=a) for a in ("cnn", "gap", "mlp")]).to(DEVICE).eval()
loader = get_labeled_dataloader(npz_path=str(ROOT / "data/mpi3d/mpi3d_real.npz"),
                                batch_size=256, num_workers=0)

rows = []
for s in SEEDS:
    run = rmu_run_name(300.0, 10, c=1.0, u_seed=s)
    ck = SWEEP / run / "best_tradeoff.pt"
    if not ck.exists():
        print(f"MISSING {run}"); continue
    model, _ = load_vae_checkpoint(ck, DEVICE)
    mu, sv = encode_dataset_cached(model, loader, DEVICE, ck, split="labeled")
    m = bespoke_metrics_cached(mu, sv, ck, forget_val=1)
    atk = attack(ck.resolve(), judge, cfg, 0.5)["success_at_r"][str(REPORT_R)]
    rows.append({"seed": s, "linear_auroc": m["linear_auroc"], "knn_cohesion": m["knn_cohesion"],
                 "dir_norm": m["dir_norm"], "attack8": float(atk)})
    print(f"u_seed={s}  AUROC={m['linear_auroc']:.4f}  cohesion={m['knn_cohesion']:.4f}  "
          f"dir_norm={m['dir_norm']:.4f}  attack@8={atk:.4f}")

print("\n=== RMU champion across n={} steering directions ===".format(len(rows)))
for k, lab in [("linear_auroc", "linear AUROC"), ("knn_cohesion", "kNN cohesion"),
               ("dir_norm", "dir_norm"), ("attack8", "consensus attack@8")]:
    v = np.array([r[k] for r in rows])
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"  {lab:20} {v.mean():.4f} +- {sd:.4f}   (min {v.min():.4f}, max {v.max():.4f})")

import json  # noqa: E402
(SWEEP / "u_seed_variance.json").write_text(json.dumps(rows, indent=2))
print(f"\n-> results/mpi3d/mpi3d_rmu/u_seed_variance.json")
