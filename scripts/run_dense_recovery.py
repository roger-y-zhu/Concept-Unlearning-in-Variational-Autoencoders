"""
scripts/run_dense_recovery.py
=============================
Dense operating-point recovery for the evaluation-strengthening study
(ledger D1 + the PLANNED block: "does the proposed metric add information beyond
standard metrics?").

The champion attack (run_attack_robustness.py --mode consensus) measures the
CRITERION -- ball-constrained, consensus-judged adversarial recovery -- at ONE
operating point per method. That is ~7 points/dataset, too thin to say whether
the proposed oracle-geometry metric is redundant with the standard probe/geometry
battery. This script measures the SAME criterion, with the SAME consensus judge
and identical attack settings (imported, not restated), at a STRATIFIED set of
operating points spanning weak->strong unlearning, and pairs each with the full
metric vector computed on the SAME checkpoint.

Per operating point (each run's final.pt) it records, all on final.pt so metrics
and attack are never on different checkpoints (the cache_checkpoint_mismatch bug):
  - standard proxies : linear/MLP probe AUROC, kNN cohesion, dir_norm, lift_norm
  - proposed metric  : Mahalanobis oracle-gap, CKA-to-oracle, raw Mahalanobis sep
  - criterion        : consensus attack success_at_r curve (primary = attack@adv_r)
  - covariates       : retain val_loss, forget_loss, in_budget

Caches one JSON per run (`<run>/cache/dense_recovery.json`, stamped with the
checkpoint name, skip-if-fresh) and an aggregate `results/<ds>/dense_recovery.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from torch.utils.data import DataLoader, Subset  # noqa: E402

# Reuse the EXACT attack machinery the champion table uses -- importing rather
# than restating guarantees identical judge, RADII, steps, threshold and gate.
from run_attack_robustness import (  # noqa: E402
    SETUPS, get_judge, attack, RADII, STEPS_PER_RADIUS, _budget,
)
from src.evaluation.adversarial import (  # noqa: E402
    ConsensusJudge, eval_concept_classifier, reconstruction_concept_scores,
)
from src.evaluation.separation import concept_separation, oracle_gap  # noqa: E402
from src.evaluation.cka import linear_cka  # noqa: E402
from src.models.vae import VAEConfig  # noqa: E402
from notebooks.nb_utils import (  # noqa: E402
    load_vae_checkpoint, bespoke_metrics_cached, encode_dataset_cached,
)

torch.serialization.add_safe_globals([VAEConfig])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHS = ("cnn", "gap", "mlp")            # the consensus judge, same as the master table
ENCODE_CAP = 20000                        # cap encode/probe cost; uniform across points
MAX_VAL = 1e5                             # above this a run has numerically diverged, not unlearned
# Report radius per dataset -- imported from the single source of truth in
# src.unlearning.spec (REPORT_RADIUS) so this dense study and the champion table
# (run_attack_robustness) cannot silently report attack@r at different radii.
from src.unlearning.spec import REPORT_RADIUS as ADV_R  # noqa: E402
from src.unlearning.spec import method_run_filter  # noqa: E402  finetune=lambda0 only

# Bump when the ATTACK computation changes in a way that alters results (judge
# aggregation, gate/threshold logic, attack optimiser, radii, steps). Stamped
# into every per-run attack cache so a code change invalidates them AUTOMATICALLY.
# The saturated-consensus bug (2026-09) hid precisely because the attack cache had
# no such stamp: a judge fix served stale floored numbers until caches were
# hand-deleted. Never let attack results be cached without this stamp again.
ATTACK_CACHE_VERSION = 2   # v2 = consensus judge aggregates median-of-logits (gradient-safe)


def _attack_cfg_stamp(dataset: str, threshold: float) -> str:
    """Short hash of everything that affects the attack CURVE for a checkpoint.
    Deliberately excludes adv_r: the attack computes the full success-vs-radius
    curve, so the report radius is a cheap post-hoc lookup (re-selected on cache
    hit) and must not force a re-attack."""
    import hashlib
    payload = json.dumps({
        "v": ATTACK_CACHE_VERSION, "archs": list(ARCHS), "radii": list(RADII),
        "steps": STEPS_PER_RADIUS, "threshold": round(float(threshold), 4),
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]

METHODS = ("neg_grad", "finetune", "feature_unlearn", "rmu", "fisher", "leace", "inlp")
# Exclude pure replicates (train-seed _tsN, rmu steering-seed _sN, REPRO_ reruns, and
# `_bltsN` = the same method+config re-run on replicate baseline N, i.e. the U-LiRA
# shadow / closed-form-refit runs): they are the SAME operating point re-measured,
# which would double-count in the correlation. Keep one representative (the
# unsuffixed primary run).  `_blts` was missing until 2026-10-16: two INLP refits
# (celeba k8_ep1_blts1/3) leaked into the Sept data, and the v4 U-LiRA shadows
# (~140 dirs) would have swamped the stratified selection.
REPLICATE_RE = re.compile(r"(_ts\d+|_blts\d+|_s\d+)$|^REPRO_")


def capped_loader(cfg) -> DataLoader:
    """A deterministic, size-capped view of the labelled test set for encoding."""
    ds = cfg["gate_loader"].dataset
    n = min(ENCODE_CAP, len(ds))
    return DataLoader(Subset(ds, range(n)), batch_size=256, shuffle=False)


def encode(model, loader, ckpt) -> tuple[np.ndarray, np.ndarray]:
    """Encode to (mu, concept_label) via the tested cached helper, which unpacks
    the per-dataset label correctly (dSprites returns the shape column from the
    factor vector; CelebA/MPI3D the binary is-concept). Caches next to the ckpt,
    stamped with the checkpoint name (so it can never serve another ckpt's mu)."""
    return encode_dataset_cached(model, loader, DEVICE, Path(ckpt), split="dense")


def stratified_runs(dataset: str, method: str, n_target: int) -> list[dict]:
    """Pick ~n_target operating points spanning the method's retain-damage range.

    Strength axis = final retain val_loss (comparable across methods; monotone-ish
    with how hard the method pushed). Rank-even spacing includes the weakest and
    strongest primary runs and is robust to val outliers (e.g. full LEACE)."""
    base = ROOT / f"results/{dataset}/{dataset}_{method}"
    if not base.is_dir():
        return []
    runs = []
    keep = method_run_filter(method)   # finetune -> lambda==0 only (ledger E22)
    for d in sorted(base.glob("*/")):
        if REPLICATE_RE.search(d.name):
            continue
        if keep and not keep(d.name):
            continue
        fp, hp = d / "final.pt", d / "history.json"
        if not fp.exists() or not hp.exists():
            continue
        try:
            h = json.loads(hp.read_text())
        except Exception:
            continue
        vl = h.get("val_loss") or []
        fl = h.get("forget_loss") or []
        if not vl:
            continue
        v = float(vl[-1])
        # Drop numerical divergences (val -> 1e6+, inf, nan): these are blown-up
        # training runs, not unlearning operating points, and their metrics are
        # degenerate. Legitimate expensive erasure (e.g. INLP on d=8 ~4000) is kept.
        if not np.isfinite(v) or v > MAX_VAL:
            continue
        runs.append({"name": d.name, "ckpt": fp,
                     "val": v, "forget": float(fl[-1]) if fl else None})
    if not runs:
        return []
    runs.sort(key=lambda r: r["val"])
    # Collapse near-identical operating points (e.g. INLP saturates: k=4..7 are
    # bit-identical, val within rounding) so they are not multiply counted.
    dedup = [runs[0]]
    for r in runs[1:]:
        if abs(r["val"] - dedup[-1]["val"]) > 0.01 * max(abs(dedup[-1]["val"]), 1.0):
            dedup.append(r)
    runs = dedup
    if len(runs) <= n_target:
        return runs
    idx = sorted(set(np.linspace(0, len(runs) - 1, n_target).round().astype(int)))
    return [runs[i] for i in idx]


def attack_at(success_at_r: dict, r: float) -> float:
    for k in (str(r), str(float(r)), str(int(r))):
        if k in success_at_r:
            return float(success_at_r[k])
    return float("nan")


def process_point(ckpt: Path, cfg, clf, loader, oracle_sep, mu_o, budget, adv_r,
                  dataset, method, run_name, val, forget, threshold, cfg_stamp) -> dict:
    cache = ckpt.parent / "cache" / "dense_recovery.json"
    if cache.exists():
        try:
            old = json.loads(cache.read_text())
            # Require BOTH the checkpoint identity and the attack-config stamp to
            # match: a stale cache from a different judge/threshold/radii is a
            # cache MISS, not a hit, so code changes can never serve old numbers.
            if old.get("_ckpt") == ckpt.name and old.get("_cfg") == cfg_stamp and "attack" in old:
                # Report radius is a cheap re-selection on the cached full curve.
                old["attack_at_r"] = attack_at(old["attack"], adv_r)
                old["adv_r"] = adv_r
                return old
        except Exception:
            pass
    model, _ = load_vae_checkpoint(ckpt, DEVICE)
    mu, sv = encode(model, loader, ckpt)
    bm = bespoke_metrics_cached(mu, sv, ckpt, forget_val=cfg["forget_val"])
    sep = concept_separation(mu, sv, forget_val=cfg["forget_val"])
    gap = oracle_gap(sep, oracle_sep)["mahalanobis"]
    cka = linear_cka(mu, mu_o)
    atk = attack(ckpt, clf, cfg, threshold)["success_at_r"]
    row = {
        "_ckpt": ckpt.name, "_cfg": cfg_stamp, "dataset": dataset, "method": method, "run": run_name,
        "val": val, "forget": forget,
        "in_budget": (val is not None and val <= budget),
        "linear_auroc": bm.get("linear_auroc"), "mlp_auroc": bm.get("mlp_auroc"),
        "knn_cohesion": bm.get("knn_cohesion"), "dir_norm": bm.get("dir_norm"),
        "lift_norm": bm.get("lift_norm"),
        "mahalanobis": sep.get("mahalanobis"), "oracle_gap": gap, "cka_oracle": cka,
        "attack": atk, "adv_r": adv_r, "attack_at_r": attack_at(atk, adv_r),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(row, indent=2))
    return row


def run_dataset(dataset: str, n_target: int, dry: bool) -> list[dict]:
    # dataset-aware attack radii: the flow latent is full-dimensional, so it needs a
    # different grid. Set the attack module global (attack() reads it) AND ours (the
    # cache stamp + off-grid assert below read this module's RADII) so they agree.
    import run_attack_robustness as _rar
    _rar.RADII = _rar.radii_for(dataset)
    global RADII
    RADII = _rar.RADII
    print(f"\n{'=' * 72}\n{dataset}  (adv_r={ADV_R[dataset]}, n_target={n_target}/method)\n{'=' * 72}")
    selection = {m: stratified_runs(dataset, m, n_target) for m in METHODS}
    total = sum(len(v) for v in selection.values())
    for m in METHODS:
        vals = [f"{r['val']:.1f}" for r in selection[m]]
        print(f"  {m:16s} {len(selection[m]):2d} pts  val range [{', '.join(vals)}]")
    print(f"  -> {total} operating points (+ baseline, oracle)")
    if dry:
        return []

    cfg = SETUPS[dataset]()
    budget = _budget(dataset)
    adv_r = ADV_R[dataset]
    # Fail fast: if the report radius is not one the attack actually evaluates,
    # attack_at() returns nan for EVERY point and those rows vanish from the
    # correlations silently. Catch the config error instead.
    assert float(adv_r) in {float(r) for r in RADII}, (
        f"ADV_R[{dataset}]={adv_r} is not in RADII={RADII}; attack_at would be nan")
    loader = capped_loader(cfg)

    clf = ConsensusJudge([get_judge(cfg, 0, arch=a) for a in ARCHS]).to(DEVICE).eval()
    auroc = eval_concept_classifier(clf, cfg["gate_loader"], DEVICE,
                                    forget_col=cfg["forget_col"],
                                    forget_val=cfg["forget_val"])["auroc"]
    print(f"  consensus judge {ARCHS}: held-out AUROC {auroc:.4f}")

    oracle_model, _ = load_vae_checkpoint(cfg["oracle_ckpt"], DEVICE)
    mu_o, sv_o = encode(oracle_model, loader, cfg["oracle_ckpt"])
    oracle_sep = concept_separation(mu_o, sv_o, forget_val=cfg["forget_val"])

    # Gate + threshold: align with the champion path (which derives the threshold
    # from the baseline's decoded-concept scores) instead of assuming 0.5 is
    # reachable, and FAIL LOUD if the attack is vacuous even on the baseline --
    # the MPI3D-red failure mode where attack@r floors near 0 for every model.
    baseline_model, _ = load_vae_checkpoint(cfg["baseline_ckpt"], DEVICE)
    gate = reconstruction_concept_scores(
        baseline_model, clf, cfg["gate_loader"], DEVICE, to_img=cfg["to_img"],
        forget_col=cfg["forget_col"], forget_val=cfg["forget_val"], max_batches=8)
    threshold = 0.5 if gate["attack_is_meaningful"] else gate["suggested_threshold"]
    if not gate["attack_is_meaningful"]:
        print(f"  *** WARNING: default threshold 0.5 UNREACHABLE on the baseline "
              f"(decoded p_forget_mean={gate['p_forget_mean']:.3f}); attack@r would be "
              f"vacuous for every model. Falling back to Youden cut {threshold:.3f}. "
              f"Interpret these attack numbers with care. ***")
    else:
        print(f"  gate ok: baseline decoded p_forget_mean={gate['p_forget_mean']:.3f}  threshold={threshold:.3f}")
    cfg_stamp = _attack_cfg_stamp(dataset, threshold)

    rows: list[dict] = []
    # Reference anchors: baseline and oracle, same methodology.
    for name, ck in (("baseline", Path(cfg["baseline_ckpt"])), ("oracle", Path(cfg["oracle_ckpt"]))):
        h = json.loads((ck.parent / "history.json").read_text()) if (ck.parent / "history.json").exists() else {}
        vl = (h.get("val_loss") or [None])[-1]
        t0 = time.time()
        rows.append(process_point(ck, cfg, clf, loader, oracle_sep, mu_o, budget, adv_r,
                                  dataset, name, name, vl, None, threshold, cfg_stamp))
        print(f"  [{name:<10}] attack@{adv_r}={rows[-1]['attack_at_r']:.3f}  ({time.time()-t0:.0f}s)")

    for m in METHODS:
        for r in selection[m]:
            t0 = time.time()
            row = process_point(r["ckpt"], cfg, clf, loader, oracle_sep, mu_o, budget, adv_r,
                                dataset, m, r["name"], r["val"], r["forget"], threshold, cfg_stamp)
            rows.append(row)
            print(f"  [{m:<14} {r['name']:<28}] val={r['val']:.1f} "
                  f"lin={row['linear_auroc']:.3f} cka_o={row['cka_oracle']:.3f} "
                  f"attack@{adv_r}={row['attack_at_r']:.3f}  ({time.time()-t0:.0f}s)")

    out = ROOT / f"results/{dataset}/dense_recovery.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"  -> {out.relative_to(ROOT)}  ({len(rows)} points)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dsprites", "celeba", "mpi3d", "dsprites_b4", "mpi3d_b4", "celeba_b4", "dsprites_btc", "celeba_btc", "mpi3d_btc", "dsprites_flow", "all"], default="all")
    ap.add_argument("--n-target", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="print the stratified selection only")
    args = ap.parse_args()
    datasets = ["dsprites", "celeba", "mpi3d"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, args.n_target, args.dry_run)


if __name__ == "__main__":
    main()
