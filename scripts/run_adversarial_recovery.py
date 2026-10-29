"""
scripts/run_adversarial_recovery.py
====================================
First real run of `src/evaluation/adversarial.py` (proposal S3.3.4) -- an
active latent-space PGD attack, distinct from every passive probe metric in
the project. It was implemented and unit-tested (tests/test_adversarial.py)
but never pointed at a real checkpoint. It is
the entire evidence base for H2 (does latent geometry predict recoverability),
so it is the highest-value experiment left that touches no new code.

WHAT IT ANSWERS THAT THE PASSIVE PROBES DO NOT
------------------------------------------------
`mlp_probe` (bespoke_metrics) asks "is the concept decodable from mu right
now". `probe_recovery_curve` (this project's Aug-2026 addition) asks the same
question as a function of attacker sample budget. Both are PASSIVE: they read
the representation, they never touch the decoder in the attack itself.

Adversarial recovery is ACTIVE: starting from a retain-set image's mu, take
signed-gradient steps to find the nearest latent point whose DECODED image a
fixed external classifier calls the concept. A model can score at chance on
every passive probe while the forget region still sits one small step away in
latent space -- this is the only metric here that would catch that.

PROCEDURE
---------
1. Train (or load a cached) ConceptClassifier per dataset on REAL images with
   ground-truth labels -- one fixed judge, reused for every model attacked.
2. Run the validity gate (`reconstruction_concept_scores`) on the BASELINE.
   If the classifier never fires above 0.5 on the baseline's own decoded
   forget images, the attack is vacuous and every model would report
   success_rate=0 for the same non-reason. Read `attack_is_meaningful` and
   fall back to `suggested_threshold` if the default is unreachable, exactly
   as the module's own docstring prescribes -- do not skip this step.
3. Run the attack itself on baseline, oracle, and every method's
   budget-selected champion (the same selection rule -- and the same
   require_exact=True guard against loading the wrong epoch -- as every
   cross_method notebook), starting from RETAIN-set images only.

Skip-if-done: the classifier is cached to disk and the per-model results are
written incrementally, so an interrupted run only repeats the models after the
last one written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.adversarial import (  # noqa: E402
    ConceptClassifier, train_concept_classifier, eval_concept_classifier,
    reconstruction_concept_scores, adversarial_recovery,
    adversarial_recovery_within_radius,
)
from notebooks.nb_utils import load_vae_checkpoint, collect_method_champions  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 48025845
BATCH_SIZE = 256
CLF_EPOCHS = 5
ATTACK_MAX_BATCHES = 8      # 8 * 256 = 2048 retain-set starting points per model
GATE_MAX_BATCHES = 20       # generous, this is a cheap forward-only pass

# ---------------------------------------------------------------------------
# Step size must be derived from the latent dimension, NOT fixed
# ---------------------------------------------------------------------------
# adversarial_recovery steps z <- z + step_size * grad.sign().  Every entry of
# grad.sign() is +-1, so its L2 norm is exactly sqrt(d) and the distance the
# attack actually travels per step is step_size * sqrt(d) -- it scales with the
# latent dimension.
#
# The first run of this script used a fixed step_size=0.1 for both datasets,
# which meant:
#     dSprites (d=8)    0.283 L2 per step
#     CelebA   (d=256)  1.600 L2 per step
# So (a) the two datasets' distances were never on a comparable scale, and
# (b) on CelebA the measurement resolution (1.6) was COARSER than the spread
# between methods (4.85 to 6.72) -- about one step separated the best model
# from the worst, and the observed minimum of exactly 1.60 is simply one step.
# Reported means looked precise and were quantised.
#
# Fixing step_size per dataset so the L2 travelled per step is a constant
# L2_PER_STEP makes resolution uniform, fine relative to the effect, and makes
# dSprites and CelebA directly comparable for the first time.
L2_PER_STEP = 0.05          # target L2 distance travelled per attack step
MAX_REACH = 15.0            # ceiling on total distance searched, both datasets


def attack_params(latent_dim: int) -> tuple[float, int]:
    """(step_size, max_steps) giving L2_PER_STEP resolution out to MAX_REACH."""
    step_size = L2_PER_STEP / (latent_dim ** 0.5)
    max_steps = int(round(MAX_REACH / L2_PER_STEP))
    return step_size, max_steps

# method -> forget_better, mirroring cross_method_{dsprites,celeba}.ipynb's
# METHODS dict plus the two erasure methods added this session (both
# forget_better="higher" -- see src/unlearning/methods.py's
# _LinearConceptErasure).
# One definition, shared -- this dict previously existed in six files, so
# adding a method meant six edits and missing one produced a table quietly
# short a row rather than an error.
from src.unlearning.spec import METHODS_FORGET_BETTER as METHODS  # noqa: E402


RADII = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
STEPS_PER_RADIUS = 60       # PGD steps inside each ball; stride scales with r


def distance_stats(distances: list) -> dict:
    """Percentiles and success-within-radius, which `mean_distance` hides.

    Two reasons the mean alone is not enough:

    * Once the attack budget is unconstrained, success_rate saturates to 1.0
      for every model whose decoder is smooth -- which is every VAE here. On
      CelebA all nine models scored exactly 1.0, so the headline number
      carried no information at all and only the distances distinguished them.
    * `success_at_r` is the quantity with an actual security reading: the
      fraction of retain points sitting within `r` of the concept region. That
      is a curve, and two models with the same mean can have very different
      curves.

    `at_zero` is a validity check, not a result: a sample "succeeding" at
    distance 0 means the classifier already fired on decode(mu(x)) for a
    RETAIN image, before a single step was taken. That is a classifier
    false positive, and if it is a large fraction the attack is measuring the
    judge rather than the model.
    """
    import numpy as np
    ok = np.array([d for d in distances if d is not None], dtype=float)
    n = len(distances)
    out = {
        "n": n,
        "n_success": int(ok.size),
        "success_rate": float(ok.size / n) if n else float("nan"),
        "at_zero_frac": float((ok == 0).sum() / n) if n else float("nan"),
    }
    if ok.size:
        for p in (10, 25, 50, 75, 90):
            out[f"p{p}"] = float(np.percentile(ok, p))
        out["mean"] = float(ok.mean())
        out["min"] = float(ok.min())
        out["max"] = float(ok.max())
    # Censored samples (never found) count as NOT within any radius, which is
    # correct: they were not reached anywhere along a trajectory longer than r.
    out["success_at_r"] = {
        str(r): float((ok <= r).sum() / n) if n else float("nan") for r in RADII
    }
    return out


def dsprites_setup():
    from src.data.dsprites import get_dataloaders, get_concept_loaders
    npz = str(ROOT / "data/dsprites/dsprites.npz")
    train_loader, _, test_loader = get_dataloaders(
        npz_path=npz, batch_size=BATCH_SIZE, val_fraction=0.1, test_fraction=0.1,
        num_workers=0, seed=SEED, include_labels=True,
    )
    _, retain_loader = get_concept_loaders(
        npz_path=npz, concept_col=1, concept_val=2, batch_size=BATCH_SIZE,
        num_workers=0, split="test",
    )
    return dict(
        clf_train_loader=train_loader, gate_loader=test_loader,
        attack_source_loader=retain_loader,
        in_channels=1, forget_col=1, forget_val=2,
        to_img=torch.sigmoid,  # BCE decoder is raw logits; real images are {0,1}
        baseline_ckpt=ROOT / "results/dsprites/dsprites_baseline/epoch_0150.pt",
        oracle_ckpt=ROOT / "results/dsprites/dsprites_oracle/epoch_0150.pt",
        classifier_path=ROOT / "results/dsprites/concept_classifier.pt",
        results_path=ROOT / "results/dsprites/adversarial_recovery.json",
    )


def celeba_setup():
    from src.data.celeba import LabeledCelebADataset, get_concept_loaders, EYEGLASSES_COL
    data_root = str(ROOT / "data/celeba/img_align_celeba")

    full = LabeledCelebADataset(data_root, attr_name=EYEGLASSES_COL, image_size=64)
    n_test = int(len(full) * 0.1)
    n_train = len(full) - n_test
    train_ds, test_ds = random_split(
        full, [n_train, n_test], generator=torch.Generator().manual_seed(SEED)
    )
    common = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=torch.cuda.is_available())
    clf_train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    gate_loader = DataLoader(test_ds, shuffle=False, **common)

    _, retain_loader = get_concept_loaders(
        data_root=data_root, forget_attr=EYEGLASSES_COL, batch_size=BATCH_SIZE, num_workers=0,
    )

    # DETERMINISTIC attack source -- do not pass `retain_loader` straight in.
    # get_concept_loaders builds TRAINING loaders: shuffle=True, drop_last=True.
    # The attack consumes only the first ATTACK_MAX_BATCHES batches, so with
    # shuffling every model is attacked on a DIFFERENT random subset -- both
    # across runs and, because the loader is re-iterated per model and the
    # global RNG advances, between models inside a single run. That makes the
    # cross-method comparison unpaired for no reason, and adds binomial
    # sampling noise (~0.005-0.011 at these sample counts) on top of the real
    # effect. The between-method gaps this table reports at the resistant end
    # are of that same order.
    #
    # dSprites was already safe: it requests split="test", which returns
    # shuffle=False loaders. Only CelebA had the training loader wired in.
    #
    # Same dataset, same samples -- only the order is pinned, so every model
    # sees an identical batch sequence and the comparison becomes paired.
    attack_source_loader = DataLoader(
        retain_loader.dataset, shuffle=False, drop_last=False, **common)

    return dict(
        clf_train_loader=clf_train_loader, gate_loader=gate_loader,
        attack_source_loader=attack_source_loader,
        in_channels=3, forget_col=1, forget_val=1,
        to_img=lambda x: x,  # decoder is Tanh in [-1,1], matching Normalize(0.5,0.5,0.5)
        baseline_ckpt=ROOT / "results/celeba/celeba_baseline/epoch_0250.pt",
        oracle_ckpt=ROOT / "results/celeba/celeba_oracle/epoch_0250.pt",
        classifier_path=ROOT / "results/celeba/concept_classifier.pt",
        results_path=ROOT / "results/celeba/adversarial_recovery.json",
    )


def mpi3d_setup():
    from src.data.mpi3d import get_dataloaders, get_concept_loaders
    npz = str(ROOT / "data/mpi3d/mpi3d_real.npz")
    # Labeled (img, is-red) train/test splits -- 2-tuple batches like CelebA, so
    # _unpack_labeled_batch reads batch[1] and forget_val=1 selects the concept.
    train_loader, _, gate_loader = get_dataloaders(
        npz_path=npz, batch_size=BATCH_SIZE, val_fraction=0.1, test_fraction=0.1,
        num_workers=0, seed=SEED, include_labels=True,
    )
    # Retain (not-red) HELD-OUT images as the attack start points. split="test"
    # already returns a deterministic (shuffle=False) loader, so -- unlike the
    # CelebA fix above -- no re-wrapping is needed; every model is attacked on the
    # identical batch sequence.
    _, retain_loader = get_concept_loaders(
        npz_path=npz, forget_factor=0, forget_val=2, batch_size=BATCH_SIZE,
        num_workers=0, split="test",
    )
    return dict(
        clf_train_loader=train_loader, gate_loader=gate_loader,
        attack_source_loader=retain_loader,
        in_channels=3, forget_col=1, forget_val=1,
        to_img=lambda x: x,  # Tanh decoder in [-1,1], matching the [-1,1] input norm
        baseline_ckpt=ROOT / "results/mpi3d/mpi3d_baseline/epoch_0120.pt",
        oracle_ckpt=ROOT / "results/mpi3d/mpi3d_oracle/epoch_0120.pt",
        classifier_path=ROOT / "results/mpi3d/concept_classifier.pt",
        results_path=ROOT / "results/mpi3d/adversarial_recovery.json",
    )


def dsprites_b4_setup():
    """beta=4 dSprites variant: SAME data, concept, judge and loaders as dsprites_setup
    (the concept classifier detects hearts in dSprites IMAGES, independent of the VAE),
    only the baseline/oracle CHECKPOINTS and the results path point at the b4 tree.
    Reuses results/dsprites/concept_classifier.pt so no new judge is trained."""
    cfg = dsprites_setup()
    cfg["baseline_ckpt"] = ROOT / "results/dsprites_b4/dsprites_b4_baseline/epoch_0150.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/dsprites_b4/dsprites_b4_oracle/epoch_0150.pt"
    cfg["results_path"]  = ROOT / "results/dsprites_b4/adversarial_recovery.json"
    return cfg


def mpi3d_b4_setup():
    """beta=4 MPI3D variant: same data/concept/judge as mpi3d_setup; only the baseline/
    oracle checkpoints + results path point at the b4 tree (reuses the mpi3d judge)."""
    cfg = mpi3d_setup()
    cfg["baseline_ckpt"] = ROOT / "results/mpi3d_b4/mpi3d_b4_baseline/epoch_0120.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/mpi3d_b4/mpi3d_b4_oracle/epoch_0120.pt"
    cfg["results_path"]  = ROOT / "results/mpi3d_b4/adversarial_recovery.json"
    return cfg


def celeba_b4_setup():
    """beta=4 CelebA variant: same data/concept/judge as celeba_setup; only checkpoints
    + results path point at the b4 tree (reuses the celeba judge)."""
    cfg = celeba_setup()
    cfg["baseline_ckpt"] = ROOT / "results/celeba_b4/celeba_b4_baseline/epoch_0250.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/celeba_b4/celeba_b4_oracle/epoch_0250.pt"
    cfg["results_path"]  = ROOT / "results/celeba_b4/adversarial_recovery.json"
    return cfg


def dsprites_btc_setup():
    """beta-TCVAE dSprites variant: same data/concept/judge/loaders as dsprites_setup;
    only the baseline/oracle checkpoints + results path point at the btc tree."""
    cfg = dsprites_setup()
    cfg["baseline_ckpt"] = ROOT / "results/dsprites_btc/dsprites_btc_baseline/epoch_0150.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/dsprites_btc/dsprites_btc_oracle/epoch_0150.pt"
    cfg["results_path"]  = ROOT / "results/dsprites_btc/adversarial_recovery.json"
    return cfg


def celeba_btc_setup():
    """beta-TCVAE CelebA: same data/concept/judge as celeba_setup; btc checkpoints + results path."""
    cfg = celeba_setup()
    cfg["baseline_ckpt"] = ROOT / "results/celeba_btc/celeba_btc_baseline/epoch_0250.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/celeba_btc/celeba_btc_oracle/epoch_0250.pt"
    cfg["results_path"]  = ROOT / "results/celeba_btc/adversarial_recovery.json"
    return cfg


def mpi3d_btc_setup():
    """beta-TCVAE MPI3D: same data/concept/judge as mpi3d_setup; btc checkpoints + results path."""
    cfg = mpi3d_setup()
    cfg["baseline_ckpt"] = ROOT / "results/mpi3d_btc/mpi3d_btc_baseline/epoch_0120.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/mpi3d_btc/mpi3d_btc_oracle/epoch_0120.pt"
    cfg["results_path"]  = ROOT / "results/mpi3d_btc/adversarial_recovery.json"
    return cfg


def dsprites_flow_setup():
    """RealNVP flow dSprites: same data/concept/judge/loaders as dsprites_setup; the
    baseline/oracle checkpoints are the flow models (loaded via the dispatching loader)
    and results land in the flow tree."""
    cfg = dsprites_setup()
    # best-val checkpoints: the unregularised flow NLL overfits past ~epoch 9, so the
    # final-epoch model generalises far worse (val NLL ~2x best); best.pt is the
    # best-val model for BOTH, keeping the baseline/oracle recipe consistent.
    cfg["baseline_ckpt"] = ROOT / "results/dsprites_flow/flow_dsprites_baseline/best.pt"
    cfg["oracle_ckpt"]   = ROOT / "results/dsprites_flow/flow_dsprites_oracle/best.pt"
    cfg["results_path"]  = ROOT / "results/dsprites_flow/adversarial_recovery.json"
    # The flow's decode already returns images in [0,1] (sigmoid + un-dequant), whereas the
    # VAE decoder emits raw BCE logits -> dsprites_setup sets to_img=sigmoid. Applying sigmoid
    # AGAIN to the flow's [0,1] output would feed the judge washed-out [0.5,0.73] images, so
    # the flow uses identity.
    cfg["to_img"] = lambda x: x
    return cfg


SETUPS = {"dsprites": dsprites_setup, "celeba": celeba_setup, "mpi3d": mpi3d_setup,
           "dsprites_b4": dsprites_b4_setup, "mpi3d_b4": mpi3d_b4_setup,
           "celeba_b4": celeba_b4_setup, "dsprites_btc": dsprites_btc_setup,
           "celeba_btc": celeba_btc_setup, "mpi3d_btc": mpi3d_btc_setup,
           "dsprites_flow": dsprites_flow_setup}


def get_classifier(cfg: dict) -> ConceptClassifier:
    if cfg["classifier_path"].exists():
        print(f"  [cached]  concept classifier <- {cfg['classifier_path']}")
        clf = ConceptClassifier(in_channels=cfg["in_channels"]).to(DEVICE)
        clf.load_state_dict(torch.load(cfg["classifier_path"], map_location=DEVICE, weights_only=True))
        clf.eval()
        return clf

    print(f"  [training] concept classifier ({CLF_EPOCHS} epochs on real images)...")
    t0 = time.time()
    clf = train_concept_classifier(
        cfg["clf_train_loader"], DEVICE, in_channels=cfg["in_channels"],
        forget_col=cfg["forget_col"], forget_val=cfg["forget_val"], epochs=CLF_EPOCHS,
    )
    print(f"  [trained]  {time.time() - t0:.1f}s")
    cfg["classifier_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), cfg["classifier_path"])
    return clf


def run_dataset(name: str) -> dict:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    cfg = SETUPS[name]()

    clf = get_classifier(cfg)
    auroc = eval_concept_classifier(
        clf, cfg["gate_loader"], DEVICE, forget_col=cfg["forget_col"], forget_val=cfg["forget_val"],
    )["auroc"]
    print(f"  classifier AUROC on held-out real images: {auroc:.4f}")

    baseline_model, baseline_ckpt_d = load_vae_checkpoint(cfg["baseline_ckpt"], DEVICE)
    oracle_model, _ = load_vae_checkpoint(cfg["oracle_ckpt"], DEVICE)
    baseline_val = baseline_ckpt_d["metrics"]["loss"]
    budget = baseline_val * 1.10

    print("  [gate] reconstruction_concept_scores on baseline...")
    gate = reconstruction_concept_scores(
        baseline_model, clf, cfg["gate_loader"], DEVICE, to_img=cfg["to_img"],
        forget_col=cfg["forget_col"], forget_val=cfg["forget_val"], max_batches=GATE_MAX_BATCHES,
    )
    threshold = 0.5 if gate["attack_is_meaningful"] else gate["suggested_threshold"]
    print(f"  gate: attack_is_meaningful={gate['attack_is_meaningful']}  "
          f"p_forget_mean={gate['p_forget_mean']:.3f}  p_retain_mean={gate['p_retain_mean']:.3f}  "
          f"auroc(decoded)={gate['auroc']:.3f}  threshold={threshold:.3f}")
    if not gate["attack_is_meaningful"]:
        print("  WARNING: default threshold unreachable even on baseline decode(mu(x_forget)); "
              "falling back to the Youden-optimal cut. Interpret success_rate=0 results with this in mind.")

    champions = collect_method_champions(name, METHODS, budget, require_exact=True)
    model_map = {"baseline": (baseline_model, cfg["baseline_ckpt"]),
                 "oracle": (oracle_model, cfg["oracle_ckpt"])}
    for method, champ in champions.items():
        m, _ = load_vae_checkpoint(champ["ckpt"], DEVICE)
        model_map[method] = (m, champ["ckpt"])
    missing = [m for m in METHODS if m not in champions]
    if missing:
        print(f"  no in-budget champion for: {missing} (every setting damaged retain past tolerance)")

    latent_dim = baseline_model.cfg.latent_dim
    step_size, max_steps = attack_params(latent_dim)
    print(f"  attack: latent_dim={latent_dim}  step_size={step_size:.5f}  max_steps={max_steps}"
          f"  -> {L2_PER_STEP} L2/step, reach {MAX_REACH}")

    results = {
        "_gate": gate, "_threshold": threshold, "_classifier_auroc": auroc,
        "_baseline_val": baseline_val, "_budget": budget,
        "_attack": {"latent_dim": latent_dim, "step_size": step_size, "max_steps": max_steps,
                    "l2_per_step": L2_PER_STEP, "max_reach": MAX_REACH, "radii": list(RADII)},
    }
    for key, (model, ckpt) in model_map.items():
        ckpt = Path(ckpt).resolve()  # champion paths come back relative (results/...); baseline/oracle absolute
        print(f"  [attacking] {key} ({ckpt.relative_to(ROOT)}) ...")
        t0 = time.time()
        rec = adversarial_recovery(
            model, clf, cfg["attack_source_loader"], DEVICE, to_img=cfg["to_img"],
            max_steps=max_steps, step_size=step_size, success_threshold=threshold,
            max_batches=ATTACK_MAX_BATCHES,
        )
        rec.pop("z_success")  # tensors, not JSON-serialisable and not needed for the summary
        stats = distance_stats(rec["distances"])

        # The number to actually report. The unconstrained walk above is kept
        # for continuity with the first run and as the step-size-sensitivity
        # evidence, but its "distance" is really n_steps * step_size; only the
        # ball-constrained search below supports a distance claim.
        rad = adversarial_recovery_within_radius(
            model, clf, cfg["attack_source_loader"], DEVICE, to_img=cfg["to_img"],
            radii=RADII, steps_per_radius=STEPS_PER_RADIUS,
            success_threshold=threshold, max_batches=ATTACK_MAX_BATCHES,
        )
        dt = time.time() - t0
        curve = "  ".join(f"r{r}:{rad['success_at_r'][str(float(r))]:.2f}" for r in RADII[:5])
        print(f"    walk_success={stats['success_rate']:.3f} | ball {curve}  ({dt:.1f}s)")
        results[key] = {**rec, "stats": stats, "within_radius": rad,
                        "ckpt": str(ckpt.relative_to(ROOT))}

    cfg["results_path"].parent.mkdir(parents=True, exist_ok=True)
    cfg["results_path"].write_text(json.dumps(results, indent=2))
    print(f"  -> {cfg['results_path'].relative_to(ROOT)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsprites", "celeba", "mpi3d", "all", "both"],
                    default="all")
    args = ap.parse_args()
    if args.dataset in ("all", "both"):
        datasets = ["dsprites", "celeba"] + (["mpi3d"] if args.dataset == "all" else [])
    else:
        datasets = [args.dataset]

    all_results = {}
    for name in datasets:
        all_results[name] = run_dataset(name)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, results in all_results.items():
        print(f"\n{name}  (threshold={results['_threshold']:.3f}, "
              f"gate attack_is_meaningful={results['_gate']['attack_is_meaningful']})")
        a = results["_attack"]
        print(f"  latent_dim={a['latent_dim']}  {a['l2_per_step']} L2/step  reach {a['max_reach']}")
        print(f"  {'model':<18} {'succ':>6} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} "
              f"{'p90':>7} {'at0':>6}")
        for key, r in results.items():
            if key.startswith("_"):
                continue
            s = r["stats"]
            cells = "".join(f"{s.get(f'p{p}', float('nan')):>7.2f}" for p in (10, 25, 50, 75, 90))
            print(f"  {key:<18} {s['success_rate']:>6.3f}{cells} {s['at_zero_frac']:>6.3f}")

        # THE table. Ball-constrained PGD, so the radius rather than the
        # optimiser's stride is the binding constraint.
        radii = [str(float(r)) for r in a["radii"]]
        print(f"\n  BALL-CONSTRAINED success within radius r  <- the trustworthy metric")
        print(f"  {'model':<18} " + " ".join(f"{float(r):>6.1f}" for r in radii))
        for key, r in results.items():
            if key.startswith("_"):
                continue
            sr = r["within_radius"]["success_at_r"]
            print(f"  {key:<18} " + " ".join(f"{sr[rr]:>6.3f}" for rr in radii))

        med = []
        for key, r in results.items():
            if key.startswith("_"):
                continue
            ms = [m for m in r["within_radius"]["min_radius"] if m is not None]
            ms.sort()
            med.append((key, ms[len(ms) // 2] if ms else None,
                        len(ms) / max(r["within_radius"]["n"], 1)))
        print(f"\n  {'model':<18} {'median min_radius':>18} {'reached_any':>12}")
        for key, m, frac in med:
            print(f"  {key:<18} {str(m):>18} {frac:>12.3f}")


if __name__ == "__main__":
    main()
