"""
src/evaluation/recovery.py
==========================
Adversarial probe recovery: how many labeled examples does an attacker need
to re-learn the concept from a frozen "unlearned" encoder?

Three evaluations in this project ask related but distinct questions -- keep
them straight when reporting:

    probing.linear_probe / mlp_probe      passive, one-shot: is the concept
        decodable right now, given abundant labeled data?
    adversarial.adversarial_recovery      active, decoder-side: can a nearby
        latent point be found that DECODES to the concept?
    THIS MODULE                           active, encoder-side: an attacker
        with n labeled images retrains probes on the frozen latents.  The
        curve AUROC(n) is the attack's sample-efficiency; its asymptote is
        the passive probe's answer, and the interesting part is how fast it
        gets there.

Why this is the right test for the erasure methods specifically: LEACE's
guarantee (Belrose et al. 2023) is that NO linear classifier -- retrained,
with any amount of data -- beats chance.  So the linear curve must stay flat
at 0.5 for every n; anything else falsifies the implementation.  The MLP
curve then quantifies exactly what linear guardedness does not buy: the
nonlinear information that remains, priced in attacker-samples.  INLP's own
evaluation (Ravfogel et al. 2020, section 6) is precisely this retrain-probes
protocol, and the sample-efficiency framing follows the control-task logic of
Hewitt & Liang 2019: a probe that needs 10x more data to recover the concept
is evidence the representation carries less accessible information, even when
both asymptotes match.

Protocol details that matter:
  * Training sets are class-BALANCED at every n (n/2 positive, n/2 negative),
    mirroring probing.mlp_probe's balancing -- a rational attacker balances,
    and it keeps small-n AUROCs from collapsing onto the majority class.
  * The held-out test set keeps its NATURAL class distribution; AUROC is
    prevalence-insensitive, so curves are comparable across datasets.
  * Every n reuses the same per-seed test split, so the curve's shape is not
    confounded by test-set resampling noise.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.probing import linear_probe, mlp_probe

DEFAULT_N_GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def probe_recovery_curve(
    mu,
    labels,
    *,
    forget_val: int = 1,
    n_grid=DEFAULT_N_GRID,
    probes=("linear", "mlp"),
    n_seeds: int = 5,
    test_size: int = 8192,
    mlp_hidden=(64,),
    thresholds=DEFAULT_THRESHOLDS,
    seed0: int = 0,
) -> dict:
    """AUROC-vs-n recovery curves for an attacker retraining probes on mu.

    Parameters
    ----------
    mu       : (N, D) latent codes of a frozen model.
    labels   : (N,) concept labels; binarised against ``forget_val`` (pass raw
               dSprites shape classes with forget_val=2, CelebA 0/1 as-is).
    n_grid   : attacker labeled-set sizes.  Each n trains on n/2 per class;
               grid entries the data cannot support are skipped, not padded.
    probes   : subset of {"linear", "mlp"}.
    n_seeds  : independent splits; curves report mean and sd across them.
    test_size: held-out evaluation pool per seed (natural class distribution).
    thresholds: AUROC levels for the n_to_recover summary.

    Returns
    -------
    dict:
        n_grid        the sizes actually evaluated (per-probe lists align)
        curves        {probe: {"auroc_mean": [...], "auroc_sd": [...],
                               "per_seed": [[...], ...]}}
        n_to_recover  {probe: {str(t): smallest n with mean AUROC >= t,
                               or None if never reached}}
        params        the protocol knobs, recorded for cache honesty
    """
    mu = np.asarray(mu)
    y_all = (np.asarray(labels).reshape(-1) == forget_val).astype(int)
    if mu.shape[0] != y_all.shape[0]:
        raise ValueError(f"mu has {mu.shape[0]} rows but labels has {y_all.shape[0]}")
    n_total = mu.shape[0]
    if y_all.sum() == 0 or y_all.sum() == n_total:
        raise ValueError("need both classes present to measure recovery")

    test_size = int(min(test_size, max(2, n_total // 4)))

    # Sizes every seed can support: n/2 positives must fit in the train pool
    # even in the worst seed (pool positives >= total positives - test_size).
    worst_pool_pos = int(y_all.sum()) - test_size
    worst_pool_neg = int((1 - y_all).sum()) - test_size
    usable = [int(n) for n in n_grid
              if n >= 2 and n // 2 <= worst_pool_pos and n // 2 <= worst_pool_neg]
    if not usable:
        raise ValueError(
            f"no n in {tuple(n_grid)} fits: pool has at most {worst_pool_pos} "
            f"positives / {worst_pool_neg} negatives after the test split")

    per_seed: dict[str, list[list[float]]] = {p: [] for p in probes}

    for s in range(n_seeds):
        rng = np.random.RandomState(seed0 + s)
        order = rng.permutation(n_total)
        test_idx, pool_idx = order[:test_size], order[test_size:]
        # Guarantee both classes appear in the test half (AUROC needs both).
        while len(np.unique(y_all[test_idx])) < 2:  # pragma: no cover - tiny-N guard
            order = rng.permutation(n_total)
            test_idx, pool_idx = order[:test_size], order[test_size:]

        mu_te, y_te = mu[test_idx], y_all[test_idx]
        pool_pos = pool_idx[y_all[pool_idx] == 1]
        pool_neg = pool_idx[y_all[pool_idx] == 0]

        aurocs: dict[str, list[float]] = {p: [] for p in probes}
        for n in usable:
            half = n // 2
            tr = np.concatenate([pool_pos[:half], pool_neg[:half]])
            rng.shuffle(tr)
            mu_tr, y_tr = mu[tr], y_all[tr]

            if "linear" in probes:
                res = linear_probe(mu_tr, y_tr, mu_te, y_te, seed=seed0 + s)
                aurocs["linear"].append(float(res["auroc"]))
            if "mlp" in probes:
                # balance=False: the training set is already exactly balanced.
                res = mlp_probe(mu_tr, y_tr, mu_te, y_te, seed=seed0 + s,
                                hidden=tuple(mlp_hidden), balance=False)
                aurocs["mlp"].append(float(res["auroc"]))

        for p in probes:
            per_seed[p].append(aurocs[p])

    curves = {}
    n_to_recover: dict[str, dict[str, int | None]] = {}
    for p in probes:
        arr = np.asarray(per_seed[p])                     # (n_seeds, len(usable))
        mean, sd = arr.mean(axis=0), arr.std(axis=0)
        curves[p] = {"auroc_mean": [float(v) for v in mean],
                     "auroc_sd":   [float(v) for v in sd],
                     "per_seed":   [[float(v) for v in row] for row in arr]}
        n_to_recover[p] = {}
        for t in thresholds:
            hit = next((usable[i] for i in range(len(usable)) if mean[i] >= t), None)
            n_to_recover[p][str(t)] = hit

    return {
        "n_grid": usable,
        "curves": curves,
        "n_to_recover": n_to_recover,
        "params": {"forget_val": int(forget_val), "n_seeds": int(n_seeds),
                   "test_size": int(test_size), "mlp_hidden": list(mlp_hidden),
                   "thresholds": [float(t) for t in thresholds],
                   "seed0": int(seed0), "probes": list(probes)},
    }
