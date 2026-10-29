"""
scripts/run_attack_robustness.py
=================================
Stress-tests the adversarial-recovery result (ledger E4) against the ways it
could be an artifact rather than a finding.

E4 says latent metrics do not predict active-attack resistance: `feature_unlearn`
and `baseline` have identical Linear AUROC / MLP AUROC / kNN cohesion / dir_norm
(an identity -- its encoder is frozen by construction) yet differ 3.9x in
attack success at r=4. That is a controlled comparison, but it rests on ONE
classifier, one training seed and one threshold. Three ways it could still be
wrong, and the mode that tests each:

  --mode judges     The classifier is the judge of "looks like the concept".
                    If the ranking of models depends on which judge was trained,
                    the metric measures the judge rather than the models.
                    Retrains the judge at several seeds and re-runs the attack.

  --mode threshold  success_threshold=0.5 is arbitrary. Decoded forget images
                    score p(concept)=0.513 under the baseline but 0.0017 under
                    feature_unlearn, so the models sit on opposite sides of that
                    cut. If the ranking flips with the threshold, the number is
                    a cut artifact.

  --mode dose       The linear-probe invariance to partial erasure is PROVEN
                    (P_a is invertible for a<1, so separability cannot change).
                    The attack goes through the DECODER, which carries no such
                    guarantee. If attack success moves with dose while linear
                    AUROC stays flat at 0.6307, the geometry/decodability split
                    is demonstrated on a third, independent axis -- and the
                    "it's a probe artifact" objection to E1 dies.

Already ruled out separately (no mode needed): the blur hypothesis. Matched
reconstruction quality across feature_unlearn / leace / inlp (MSE 0.0309-0.0326,
gradient energy 0.0498-0.0502) coexists with attack@4 spanning 0.172-0.900, so
image degradation does not explain resistance.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.models.vae import VAEConfig  # noqa: E402
from src.evaluation.adversarial import (  # noqa: E402
    train_concept_classifier, eval_concept_classifier,
    adversarial_recovery_within_radius, JUDGE_ARCHS, ConsensusJudge,
)
from src.unlearning.spec import val_budget, METHODS_FORGET_BETTER  # noqa: E402
from src.utils.seed import seed_all  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint, collect_method_champions  # noqa: E402

torch.serialization.add_safe_globals([VAEConfig])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
ATTACK_MAX_BATCHES = 8
DEFAULT_RADII = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
# Flow models carry a full-dimensional latent (C*H*W; 4096 for 64x64x1 dSprites) with
# z ~ N(0,I), so typical ||z|| ~ sqrt(dim) ~ 64 and a VAE-scale radius reads ~0 success
# everywhere. The flow grid is calibrated empirically from the trained baseline's latent
# (scripts/calibrate_flow_radii.py) -- the values below are PROVISIONAL until then.
FLOW_RADII = (16.0, 20.0, 24.0, 28.0, 32.0, 40.0)  # calibrated: ||z||~11, attack floor
#                                                   <=16, steep 18-28, saturates ~32
RADII_BY_DATASET = {"dsprites_flow": FLOW_RADII}


def radii_for(dataset: str):
    """The attack radius grid for a dataset. VAE datasets use DEFAULT_RADII; the flow
    uses its full-dimensional-latent grid."""
    return RADII_BY_DATASET.get(dataset, DEFAULT_RADII)


# Resolved per-invocation in main() (and per-dataset by run_dense_recovery); the module
# default keeps the importable name valid for callers that read it before main().
RADII = DEFAULT_RADII
STEPS_PER_RADIUS = 60

# Import the dataset wiring from the main runner rather than restating it, so
# the two cannot drift on loaders, channels or the to_img convention.
sys.path.insert(0, str(ROOT / "scripts"))
from run_adversarial_recovery import dsprites_setup, celeba_setup, mpi3d_setup, dsprites_b4_setup, mpi3d_b4_setup, celeba_b4_setup, dsprites_btc_setup, celeba_btc_setup, mpi3d_btc_setup, dsprites_flow_setup  # noqa: E402

SETUPS = {"dsprites": dsprites_setup, "celeba": celeba_setup, "mpi3d": mpi3d_setup,
          "dsprites_b4": dsprites_b4_setup, "mpi3d_b4": mpi3d_b4_setup,
          "celeba_b4": celeba_b4_setup, "dsprites_btc": dsprites_btc_setup,
          "celeba_btc": celeba_btc_setup, "mpi3d_btc": mpi3d_btc_setup,
          "dsprites_flow": dsprites_flow_setup}


#: Set by --budget. None means "derive from the canonical baseline checkpoint",
#: which is what every headline number uses.
#:
#: The override exists because the retain budget is defined as
#: canonical_baseline_val * (1 + VAL_TOLERANCE), and queue28 showed the dSprites
#: baseline itself has training-seed sd 0.79 -- so a different baseline draw
#: would have set a different budget. Measured: under the lowest of four
#: baselines (val 35.15 -> budget 38.66 instead of 40.59), three dSprites
#: methods select a DIFFERENT champion configuration (feature_unlearn, fisher,
#: rmu). CelebA is unaffected (baseline sd 0.085; identical champions under all
#: four budgets). This flag makes that sensitivity measurable rather than
#: assumed away.
BUDGET_OVERRIDE = None


def _budget(dataset: str) -> float:
    return val_budget(dataset) if BUDGET_OVERRIDE is None else BUDGET_OVERRIDE


def get_judge(cfg: dict, seed: int, epochs: int = 5, arch: str = "cnn"):
    """Train a judge at a given seed, cached on disk.

    seed 0 reuses the original `concept_classifier.pt` so results stay
    comparable with the main run rather than silently rebasing on a new judge.
    """
    if seed == 0 and arch == "cnn":
        path = cfg["classifier_path"]          # the original, for comparability
    else:
        suffix = f"_{arch}" if arch != "cnn" else ""
        path = cfg["classifier_path"].with_name(
            cfg["classifier_path"].stem + f"{suffix}_s{seed}.pt")
    clf = JUDGE_ARCHS[arch](in_channels=cfg["in_channels"]).to(DEVICE)
    if path.exists():
        clf.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        clf.eval()
        print(f"  [cached]   judge arch={arch} seed={seed}")
        return clf
    print(f"  [training] judge arch={arch} seed={seed} ...")
    t0 = time.time()
    seed_all(seed)
    trained = train_concept_classifier(
        cfg["clf_train_loader"], DEVICE, in_channels=cfg["in_channels"],
        forget_col=cfg["forget_col"], forget_val=cfg["forget_val"], epochs=epochs,
        arch=arch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), path)
    print(f"  [trained]  judge arch={arch} seed={seed} in {time.time() - t0:.0f}s")

    # ROUND-TRIP THROUGH DISK -- do not "optimise" this away by returning
    # `trained` directly. That was the original code, and it made the FIRST run
    # of any analysis disagree with every later run of the same analysis.
    #
    # Measured on dSprites: the run that trained the gap/mlp judges reported
    # fisher 0.2568, rmu 0.2666, neg_grad 0.1670; every subsequent run, reading
    # the same judges back from these files against the same checkpoints,
    # reports 0.2510 / 0.2715 / 0.1699 -- and those later values reproduce
    # exactly, three times in one process and again across processes. So the
    # cached path is the correct one and the freshly-trained object is the
    # outlier; returning it silently made attack numbers depend on whether the
    # judge cache happened to be cold.
    #
    # Deltas were 0.003-0.006, which is larger than real between-method gaps at
    # the resistant end (neg_grad vs feature_unlearn is 0.0019), so this was
    # able to reorder methods. Loading the saved weights into a fresh module
    # makes cold-cache and warm-cache runs the same computation by construction,
    # which is the property that matters -- it holds whatever the residual
    # difference in the trained object turns out to be.
    clf = JUDGE_ARCHS[arch](in_channels=cfg["in_channels"]).to(DEVICE)
    clf.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    clf.eval()
    return clf


def champion_models(dataset: str, cfg: dict) -> dict:
    budget = _budget(dataset)
    out = {"baseline": Path(cfg["baseline_ckpt"]), "oracle": Path(cfg["oracle_ckpt"])}
    for method, champ in collect_method_champions(
            dataset, METHODS_FORGET_BETTER, budget, require_exact=True).items():
        out[method] = Path(champ["ckpt"]).resolve()
    return out


def attack(model_path: Path, clf, cfg, threshold: float) -> dict:
    model, _ = load_vae_checkpoint(model_path, DEVICE)
    return adversarial_recovery_within_radius(
        model, clf, cfg["attack_source_loader"], DEVICE, to_img=cfg["to_img"],
        radii=RADII, steps_per_radius=STEPS_PER_RADIUS,
        success_threshold=threshold, max_batches=ATTACK_MAX_BATCHES,
    )


def spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def mode_judges(dataset: str, seeds: list[int], report_r: float) -> dict:
    cfg = SETUPS[dataset]()
    models = champion_models(dataset, cfg)
    results: dict[str, dict] = {}
    for s in seeds:
        clf = get_judge(cfg, s)
        auroc = eval_concept_classifier(clf, cfg["gate_loader"], DEVICE,
                                        forget_col=cfg["forget_col"],
                                        forget_val=cfg["forget_val"])["auroc"]
        print(f"  judge seed={s}: held-out AUROC {auroc:.4f}")
        row = {}
        for name, p in models.items():
            r = attack(p, clf, cfg, 0.5)
            row[name] = r["success_at_r"][str(report_r)]
            print(f"    {name:<18} attack@{report_r} = {row[name]:.3f}")
        results[f"seed{s}"] = {"auroc": auroc, "attack": row}
    return results


def mode_archs(dataset: str, archs: list[str], report_r: float) -> dict:
    """Judges with genuinely different inductive biases.

    Reseeding does not test architecture -- every reseeded judge is the same
    CNN, so a shared bias survives reseeding and still looks like agreement.
    If cnn / gap / mlp agree on the ORDER of models, the ranking is a property
    of the models rather than of how the judge happens to see images.
    """
    cfg = SETUPS[dataset]()
    models = champion_models(dataset, cfg)
    results: dict[str, dict] = {}
    for arch in archs:
        clf = get_judge(cfg, 0, arch=arch)
        auroc = eval_concept_classifier(clf, cfg["gate_loader"], DEVICE,
                                        forget_col=cfg["forget_col"],
                                        forget_val=cfg["forget_val"])["auroc"]
        print(f"  judge arch={arch}: held-out AUROC {auroc:.4f}")
        row = {}
        for name, p in models.items():
            r = attack(p, clf, cfg, 0.5)
            row[name] = r["success_at_r"][str(report_r)]
            print(f"    {name:<18} attack@{report_r} = {row[name]:.3f}")
        results[arch] = {"auroc": auroc, "attack": row}
    return results


def mode_replicates(dataset: str, archs: list[str], report_r: float) -> dict:
    """Model-level error bars on the attack metric -- the one thing in this
    project's headline result that was still n=1.

    Judge variance is thoroughly replicated (mode judges: 4 seeds; mode archs:
    3 architectures) but that measures the JUDGE. The attack number itself was
    computed on ONE checkpoint per method, so a reader could reasonably ask
    whether feature_unlearn's 0.053 vs LEACE's 0.727 is a property of those
    methods or of the two particular training runs that happened to be
    selected as champions.

    queue24/25 already trained 4 --train-seed replicates of every champion
    config, and they have been sitting unused by this metric. Each replicate is
    re-selected independently (its own history, its own trade-off epoch --
    same discipline build_master_table applies), so this measures how much the
    attack result moves when the model is retrained, not when the judge is.

    Uses the ConsensusJudge, so these are error bars on the metric actually
    being reported (see mode_consensus for why single-judge success is not).
    """
    from src.evaluation.selection import load_history, tradeoff_epoch, resolve_checkpoint, is_replicate

    cfg = SETUPS[dataset]()
    budget = _budget(dataset)
    clf = ConsensusJudge([get_judge(cfg, 0, arch=a) for a in archs]).to(DEVICE).eval()

    champs = collect_method_champions(dataset, METHODS_FORGET_BETTER, budget,
                                      require_exact=True)
    results: dict[str, dict] = {"_archs": archs, "_radius": report_r}

    # baseline and oracle FIRST, in this same process, even though neither has
    # _ts replicates. They are the reference points the method rows are read
    # against, and if they come from a different invocation the comparison
    # spans two runs. Repeated attacks on one checkpoint were measured to shift
    # by up to ~0.006 between processes while being stable within one (cuDNN
    # picks conv algorithms per process; grad.sign() and the 0.5 decision
    # threshold turn tiny gradient differences into flipped borderline
    # samples). That offset is comparable to the between-method gaps at the
    # resistant end, so a cross-run comparison can invent or erase a
    # difference. Emitting every row from one process makes the column
    # internally comparable, which is what the table actually claims.
    for ref in ("baseline", "oracle"):
        ck = Path(cfg[f"{ref}_ckpt"])
        if not ck.exists():
            continue
        v = attack(ck.resolve(), clf, cfg, 0.5)["success_at_r"][str(report_r)]
        results[ref] = {"n": 1, "mean": v, "sd": 0.0, "values": [v]}
        print(f"  {ref:<18} n=1  {v:.4f}  (reference, no training-seed replicates)")

    for method, champ in champs.items():
        champ_dir = Path(champ["run_dir"])
        base = champ_dir.name
        fb = METHODS_FORGET_BETTER[method]

        runs = [champ_dir]
        parent = champ_dir.parent
        if parent.is_dir():
            for d in sorted(parent.iterdir()):
                if not d.is_dir():
                    continue
                # `_ts<N>`: the champion retrained under another TRAINING seed.
                if is_replicate(d.name) and d.name.rsplit("_ts", 1)[0] == base:
                    runs.append(d)
                # `_bl<tag>`: the same closed-form method REFIT on a replicate
                # baseline (run.py --baseline-ckpt). LEACE and INLP are post-hoc
                # transforms of a trained encoder, so they have no training seed
                # to vary and were the only rows in the master table stuck at
                # n=1 -- while sitting at the extreme of the headline attack
                # claim. Refitting them per replicate baseline is the only axis
                # along which they can vary at all, and it is the RIGHT axis:
                # it asks whether the erasure result depends on which particular
                # baseline it was fitted to.
                elif "_bl" in d.name and d.name.rsplit("_bl", 1)[0] == base:
                    runs.append(d)

        vals = []
        for run in runs:
            h = load_history(run)
            if h is None:
                continue
            ep = tradeoff_epoch(h, fb, budget)
            if ep is None:
                continue
            ck, exact = resolve_checkpoint(run, ep, len(h["val_loss"]))
            if not exact or not Path(ck).exists():
                continue
            r = attack(Path(ck).resolve(), clf, cfg, 0.5)
            v = r["success_at_r"][str(report_r)]
            vals.append(v)
            print(f"    {run.name:<34} attack@{report_r} = {v:.4f}")

        if vals:
            import statistics
            mean = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            results[method] = {"n": len(vals), "mean": mean, "sd": sd, "values": vals}
            print(f"  {method:<18} n={len(vals)}  mean={mean:.4f} +- {sd:.4f}")
    return results


def mode_consensus(dataset: str, archs: list[str], report_r: float) -> dict:
    """The transfer-validated attack: success requires a majority of judges.

    Single-judge success overstates reachability, and it does so unequally --
    feature_unlearn's apparent successes almost never transfer while the
    baseline's mostly do. Requiring consensus removes that component, so this is
    the number to report when the claim is about the CONCEPT rather than about
    one classifier.
    """
    cfg = SETUPS[dataset]()
    models = champion_models(dataset, cfg)
    clf = ConsensusJudge([get_judge(cfg, 0, arch=a) for a in archs]).to(DEVICE).eval()
    auroc = eval_concept_classifier(clf, cfg["gate_loader"], DEVICE,
                                    forget_col=cfg["forget_col"],
                                    forget_val=cfg["forget_val"])["auroc"]
    print(f"  consensus of {archs}: held-out AUROC {auroc:.4f}")
    out = {"_archs": archs, "_auroc": auroc}
    for name, path in models.items():
        r = attack(path, clf, cfg, 0.5)
        out[name] = r["success_at_r"]
        print(f"  {name:<18} " +
              "  ".join(f"r{k}={v:.3f}" for k, v in r["success_at_r"].items()))
    return out


def mode_transfer(dataset: str, attack_arch: str, judge_archs: list[str],
                  radius: float) -> dict:
    """Do successful attacks find the CONCEPT, or just adversarial examples?

    THE PROBLEM THIS TESTS
    ----------------------
    The attack is PGD against a classifier, so by construction it can succeed by
    finding an adversarial example -- a latent point whose decoding fools THAT
    classifier without looking like the concept to anything else. If that is
    what is happening, the metric measures classifier fragility, not concept
    reachability, and every ranking built on it is measuring the wrong thing.

    Cross-judge agreement on RANKINGS (mode archs) does not address this: there
    each judge attacked with itself, so each could have been fooled by its own
    adversarial examples and still produce the same ordering.

    THE TEST
    --------
    Attack with judge A. Take the latents where A declared success, decode them,
    and ask judges B and C -- which were never used in the search. High transfer
    means the attack found genuine concept content. Low transfer means
    adversarial examples.

    Reference point: judges B and C do not agree perfectly even on REAL concept
    images, so their agreement rate there is the ceiling, printed alongside.
    """
    cfg = SETUPS[dataset]()
    models = champion_models(dataset, cfg)
    attacker = get_judge(cfg, 0, arch=attack_arch)
    others = {a: get_judge(cfg, 0, arch=a) for a in judge_archs if a != attack_arch}

    # Ceiling: how often do the held-out judges call a REAL concept image the
    # concept? Transfer cannot exceed this and must be read against it.
    ceiling = {}
    fl = cfg.get("forget_eval_loader") or cfg["gate_loader"]
    with torch.no_grad():
        imgs, ys = [], []
        for bi, batch in enumerate(fl):
            if bi >= 4:
                break
            im, y = batch[0], batch[1]
            imgs.append(im)
            ys.append(y)
        im = torch.cat(imgs).to(DEVICE)
        y = torch.cat(ys).reshape(-1)
        pos = im[(y == cfg["forget_val"])]
        for a, clf in others.items():
            ceiling[a] = float((torch.sigmoid(clf(pos)) > 0.5).float().mean()) if len(pos) else float("nan")
    print(f"  ceiling on REAL concept images: " +
          "  ".join(f"{a}={v:.3f}" for a, v in ceiling.items()))

    results = {"_ceiling": ceiling, "_radius": radius, "_attack_arch": attack_arch}
    for name, path in models.items():
        model, _ = load_vae_checkpoint(path, DEVICE)
        model.eval()
        step = 2.5 * radius / STEPS_PER_RADIUS
        hits, agree = 0, {a: 0 for a in others}
        for bi, batch in enumerate(cfg["attack_source_loader"]):
            if bi >= ATTACK_MAX_BATCHES:
                break
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(DEVICE)
            with torch.no_grad():
                mu0, _ = model.encode(x)
            z = mu0.clone().detach().requires_grad_(True)
            for _ in range(STEPS_PER_RADIUS):
                logits = attacker(cfg["to_img"](model.decode(z)))
                g = torch.autograd.grad(logits.sum(), z, allow_unused=True)[0]
                if g is None:
                    break
                with torch.no_grad():
                    z = z + step * g.sign()
                    d = z - mu0
                    nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    z = mu0 + d * torch.clamp(radius / nrm, max=1.0)
                z = z.detach().requires_grad_(True)
            with torch.no_grad():
                img = cfg["to_img"](model.decode(z))
                won = torch.sigmoid(attacker(img)) > 0.5
                hits += int(won.sum())
                for a, clf in others.items():
                    agree[a] += int(((torch.sigmoid(clf(img)) > 0.5) & won).sum())
        row = {"n_success": hits,
               "transfer": {a: (agree[a] / hits if hits else float("nan")) for a in others}}
        results[name] = row
        tr = "  ".join(f"{a}={v:.3f}" for a, v in row["transfer"].items())
        print(f"  {name:<18} A-success={hits:>5}   transfer: {tr}")
    return results


def mode_threshold(dataset: str, thresholds: list[float], report_r: float) -> dict:
    cfg = SETUPS[dataset]()
    models = champion_models(dataset, cfg)
    clf = get_judge(cfg, 0)
    results: dict[str, dict] = {}
    for t in thresholds:
        row = {}
        for name, p in models.items():
            r = attack(p, clf, cfg, t)
            row[name] = r["success_at_r"][str(report_r)]
        results[f"t{t}"] = {"attack": row}
        print(f"  threshold {t}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
    return results


def mode_dose(dataset: str, report_r: float) -> dict:
    """Attack the LEACE dose sweep. The question is whether the DECODER-mediated
    attack tracks dose, given the linear probe provably cannot."""
    cfg = SETUPS[dataset]()
    clf = get_judge(cfg, 0)
    sweep = ROOT / "results" / dataset / f"{dataset}_leace"
    runs = []
    for d in sorted(sweep.iterdir()):
        m = re.fullmatch(r"leace_ep1(?:_a([\d.]+))?", d.name)
        if not (d.is_dir() and m):
            continue
        a = float(m.group(1)) if m.group(1) else 1.0
        for cand in ("epoch_0001.pt", "best_tradeoff.pt", "final.pt"):
            if (d / cand).exists():
                runs.append((a, d / cand))
                break
    runs.sort()
    out = {}
    print(f"  {'a':>6}{'attack@'+str(report_r):>14}")
    for a, p in runs:
        r = attack(p, clf, cfg, 0.5)
        out[f"{a:g}"] = r["success_at_r"][str(report_r)]
        print(f"  {a:>6.2f}{out[f'{a:g}']:>14.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsprites", "celeba", "mpi3d", "dsprites_b4", "mpi3d_b4", "celeba_b4", "dsprites_btc", "celeba_btc", "mpi3d_btc", "dsprites_flow"], default="celeba")
    ap.add_argument("--mode",
                    choices=["judges", "threshold", "dose", "archs", "transfer",
                             "consensus", "replicates"],
                    required=True)
    ap.add_argument("--attack-arch", default="cnn")
    ap.add_argument("--archs", nargs="+", default=["cnn", "gap", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--report-r", type=float, default=None,
                    help="Report radius. Default: the dataset's graded radius from "
                         "src.unlearning.spec.REPORT_RADIUS, so the champion table "
                         "matches the dense study (run_dense_recovery). Override only "
                         "for a deliberate radius-sensitivity sweep.")
    ap.add_argument("--budget", type=float, default=None,
                    help="Override the retain budget. Writes to a _budget<val> "
                         "suffixed file so it never overwrites the headline result.")
    args = ap.parse_args()

    global BUDGET_OVERRIDE
    BUDGET_OVERRIDE = args.budget
    print(f"budget in use: {_budget(args.dataset):.2f}"
          + (" (OVERRIDE)" if args.budget is not None else " (canonical)"))

    global RADII
    RADII = radii_for(args.dataset)      # dataset-aware grid (flow latent is full-dim)
    if args.report_r is None:
        from src.unlearning.spec import REPORT_RADIUS
        args.report_r = REPORT_RADIUS[args.dataset]
    # attack curves are keyed by str(float(r)); fail loud if the radius is off-grid
    assert float(args.report_r) in {float(r) for r in RADII}, (
        f"--report-r {args.report_r} not in RADII={RADII}")

    print(f"{'=' * 70}\n{args.dataset} | mode={args.mode} | r={args.report_r}\n{'=' * 70}")
    if args.mode == "judges":
        res = mode_judges(args.dataset, args.seeds, args.report_r)
    elif args.mode == "replicates":
        res = mode_replicates(args.dataset, args.archs, args.report_r)
    elif args.mode == "consensus":
        res = mode_consensus(args.dataset, args.archs, args.report_r)
    elif args.mode == "transfer":
        res = mode_transfer(args.dataset, args.attack_arch, args.archs, args.report_r)
    elif args.mode == "archs":
        res = mode_archs(args.dataset, args.archs, args.report_r)
    elif args.mode == "threshold":
        res = mode_threshold(args.dataset, args.thresholds, args.report_r)
    else:
        res = mode_dose(args.dataset, args.report_r)

    # Ranking stability is the actual question for judges/threshold: absolute
    # numbers may move, but if the ORDER of models is stable the metric is
    # measuring the models rather than the judge or the cut.
    if args.mode in ("judges", "threshold", "archs"):
        keys = sorted(res)
        names = sorted(res[keys[0]]["attack"])
        base = [res[keys[0]]["attack"][n] for n in names]
        print(f"\n  ranking stability vs {keys[0]} (Spearman over {len(names)} models)")
        for k in keys[1:]:
            v = [res[k]["attack"][n] for n in names]
            print(f"    {k:<10} rho = {spearman(base, v):+.3f}")

    suffix = "" if args.budget is None else f"_budget{args.budget:g}"
    out = ROOT / f"results/{args.dataset}/attack_robustness_{args.mode}{suffix}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
