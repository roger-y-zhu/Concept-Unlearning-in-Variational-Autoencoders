"""
scripts/run_attack_sensitivity.py
=================================
How sensitive is attack@r to the strength of the optimiser? (added 2026-10-20 after the thesis examination)

The headline attack is a 60-step signed-gradient ascent per radius, started at the posterior mean, one run, no restarts.
That is a fixed, modest attacker, and a failed attack is not proof that the concept is unreachable. This script re-attacks the
champion models of Table 5.1 (baseline, oracle and each method's budget-respecting champion, beta=1, three datasets) with
stronger settings and reports attack@r at the dataset's report radius:

  default       60 steps per radius, stride 2.5 r / 60, 1 start (the headline setting; reproduces dense_recovery)
  more_steps    240 steps per radius at the SAME stride 2.5 r / 60 (four times as many iterations)
  finer_stride  240 steps per radius with stride 2.5 r / 240 (the same total path length, finer steps)
  restarts      60 steps per radius, 4 starts (the posterior mean plus 3 uniformly random points in the ball)

Same consensus judge (CNN, GAP-CNN, MLP; seed 0), same 2,048 concept-free source images, same success rule as the headline
attack. Nothing is retrained. Writes results/analysis_oct2026/attack_sensitivity_<dataset>.json (one file per dataset, so the datasets
can be run in parallel; resumable).
Models: the baseline and the oracle, and the champions of NegGrad+, Feature Unlearning, RMU and (where one exists, CelebA) LEACE,
which act in three different places (all weights, the decoder only, the encoder only or in closed form). The shared GPU made a run over
all champions too slow. The finer-stride setting is run on the baseline and the oracle only.
Run: python scripts/run_attack_sensitivity.py [--datasets dsprites celeba mpi3d]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_attack_robustness as R  # noqa: E402
from src.evaluation.adversarial import ConsensusJudge, adversarial_recovery_within_radius  # noqa: E402
from src.unlearning.spec import REPORT_RADIUS  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis_oct2026"
MODELS = ("baseline", "oracle", "neg_grad", "feature_unlearn", "rmu", "leace")
REFERENCE_ONLY = {"finer_stride": {"baseline", "oracle"}}      # settings run on the reference models only
SETTINGS = {"default": dict(steps_per_radius=60, restarts=1),
            "more_steps": dict(steps_per_radius=240, restarts=1, step_per_radius=2.5 / 60),
            "finer_stride": dict(steps_per_radius=240, restarts=1),
            "restarts": dict(steps_per_radius=60, restarts=4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dsprites", "celeba", "mpi3d"])
    a = ap.parse_args()
    for ds in a.datasets:
        out_path = OUT_DIR / f"attack_sensitivity_{ds}.json"
        res = {ds: json.loads(out_path.read_text())} if out_path.exists() else {}
        cfg = R.SETUPS[ds]()
        r = float(REPORT_RADIUS[ds])
        judge = ConsensusJudge([R.get_judge(cfg, 0, arch=x) for x in ("cnn", "gap", "mlp")]).to(R.DEVICE).eval()
        models = {k: v for k, v in R.champion_models(ds, cfg).items() if k in MODELS}
        print(f"{ds}: r={r}, {len(models)} models: {list(models)}", flush=True)
        d = res.setdefault(ds, {"report_radius": r, "models": {}})
        for name, path in models.items():
            row = d["models"].setdefault(name, {"ckpt": str(path)})
            model, _ = load_vae_checkpoint(path, R.DEVICE)
            for sname, kw in SETTINGS.items():
                if sname in row or (sname in REFERENCE_ONLY and name not in REFERENCE_ONLY[sname]):
                    continue
                t0 = time.time()
                out = adversarial_recovery_within_radius(
                    model, judge, cfg["attack_source_loader"], R.DEVICE, to_img=cfg["to_img"], radii=R.DEFAULT_RADII,
                    success_threshold=0.5, max_batches=R.ATTACK_MAX_BATCHES, seed=0, **kw)
                row[sname] = out["success_at_r"][str(r)]
                print(f"  {name:16s} {sname:10s} attack@{r:g} = {row[sname]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
                out_path.write_text(json.dumps(res[ds], indent=1))
        names = list(d["models"])
        for s in SETTINGS:
            if s == "default":
                continue
            ns = [n for n in names if s in d["models"][n]]
            if len(ns) >= 3:
                d.setdefault("spearman_vs_default", {})[s] = float(spearmanr([d["models"][n]["default"] for n in ns], [d["models"][n][s] for n in ns]).statistic)
            d.setdefault("max_abs_change", {})[s] = float(max(abs(d["models"][n][s] - d["models"][n]["default"]) for n in ns))
            d.setdefault("n_models", {})[s] = len(ns)
        out_path.write_text(json.dumps(res[ds], indent=1))
        print(json.dumps({k: v for k, v in d.items() if k != "models"}, indent=1))


if __name__ == "__main__":
    main()
