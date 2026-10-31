"""
notebooks/nb_utils.py
---------------------
Shared helpers for all project notebooks.  Import after setting ROOT:

    from notebooks.nb_utils import load_vae_checkpoint, ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Encoding cache
# ──────────────────────────────────────────────────────────────────────────────

def encode_dataset_cached(
    model,
    loader,
    device,
    ckpt_path: Path,
    *,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Encode *loader* through *model*, caching the result next to the checkpoint.

    Cache file: ``<ckpt_path.parent>/cache/encodings_<split>.npz``

    Invalidation: if the checkpoint is **newer** than the cache file, the cache
    is stale (model was retrained) and encoding runs again automatically.

    Parameters
    ----------
    model : ConvVAE  — already moved to *device*, in eval mode.
    loader           — DataLoader (any format accepted by ``encode_dataset``).
    device           — torch.device.
    ckpt_path : Path — path to the ``.pt`` file the model was loaded from.
    split : str      — label embedded in the cache filename ("train"/"val"/"test").

    Returns
    -------
    mu     : (N, latent_dim) float32 array
    labels : (N,) array, or None if the loader yields plain images.
    """
    from src.evaluation.probing import encode_dataset   # local import — avoids circular deps

    cache_dir  = ckpt_path.parent / "cache"
    cache_file = cache_dir / f"encodings_{split}.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Cache hit? ────────────────────────────────────────────────────────────
    if cache_file.exists():
        ckpt_mtime  = ckpt_path.stat().st_mtime
        cache_mtime = cache_file.stat().st_mtime
        if cache_mtime >= ckpt_mtime:
            data   = np.load(cache_file, allow_pickle=False)
            # WHICH checkpoint these encodings came from.  `split` alone does not
            # identify a model: one run directory holds best.pt, best_tradeoff.pt,
            # final.pt and epoch_NNNN.pt, and at the default split they all map to
            # the single file encodings_test.npz.  Whichever was encoded first
            # wins, and every later reader silently gets that model's latents.
            #
            # Measured on dsprites_rmu/a200.0_c1.0_ep20_s5: encodings_test.npz
            # held final.pt (epoch 20, val 46.26 -- 14% over the retain budget)
            # while the analysis resolved best_tradeoff.pt (epoch 11, val 40.21).
            # Every metric downstream inherited the wrong model, which is the
            # entirety of the "RMU makes hearts more linearly decodable" result
            # (0.6812/0.5650 from final.pt vs 0.6224/0.3762 for the real
            # champion, against a baseline of 0.6307/0.3625).
            #
            # cross_method_* already worked around this by passing
            # split=f"test_{ckpt.stem}" (see scripts/warm/README) -- correct, but
            # opt-in, so every other notebook stayed exposed. Stamping the name
            # makes the check automatic for all callers.
            cached_ckpt = (str(data["ckpt_name"]) if "ckpt_name" in data.files
                           else None)
            if cached_ckpt in (None, ckpt_path.name):
                mu     = data["mu"]
                labels = data["labels"] if "labels" in data.files else None
                print(f"  [cache hit]  {cache_file.relative_to(cache_dir.parent.parent)}"
                      f"  ({mu.shape[0]:,} samples)")
                return mu, labels
            print(f"  [cache stale] encodings were made from {cached_ckpt}, "
                  f"asked for {ckpt_path.name} — re-encoding…")
        else:
            print(f"  [cache stale] checkpoint newer than cache — re-encoding…")

    # ── Cache miss: encode and save ───────────────────────────────────────────
    print(f"  [encoding]   {ckpt_path.name}  split={split} …")
    mu, labels = encode_dataset(model, loader, device)

    save_kwargs: dict = {"mu": mu, "ckpt_name": np.array(ckpt_path.name)}
    if labels is not None:
        save_kwargs["labels"] = labels
    np.savez(cache_file, **save_kwargs)
    print(f"  [cached]     → {cache_file.name}  ({mu.shape[0]:,} samples)")

    return mu, labels


# ──────────────────────────────────────────────────────────────────────────────
# Tensor utilities
# ──────────────────────────────────────────────────────────────────────────────

def to_np(t: torch.Tensor) -> np.ndarray:
    """
    Convert a decoded image tensor to a display-ready NumPy array.

    * Grayscale (C=1 or 2-D): returns (H, W) float32.
    * RGB (C=3):               returns (H, W, 3) float32.

    The tensor should already be in [0, 1] before calling this.
    """
    t = t.detach().cpu().squeeze()
    return t.numpy() if t.ndim == 2 else t.permute(1, 2, 0).numpy()


def decode_to_img(
    model,
    z: torch.Tensor,
    *,
    mode: str = "tanh",
) -> np.ndarray:
    """
    Decode a single latent vector and return a display-ready HWC numpy array.

    Parameters
    ----------
    model : ConvVAE
        The VAE model (must be on the same device as ``z``).
    z : torch.Tensor
        Shape (1, latent_dim).
    mode : {"tanh", "sigmoid"}
        * ``"tanh"``    — decoder output is in [-1, 1]; denorm with (x+1)/2.
          Use for CelebA (MSE loss, Tanh activation).
        * ``"sigmoid"`` — decoder output is logits; apply sigmoid then clip.
          Use for dSprites (BCE loss).
    """
    with torch.no_grad():
        x_hat = model.decode(z)          # (1, C, H, W)

    if mode == "tanh":
        x_hat = (x_hat + 1.0) / 2.0
    else:
        x_hat = torch.sigmoid(x_hat)

    return to_np(x_hat.clamp(0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Axes helpers
# ──────────────────────────────────────────────────────────────────────────────

def show_ax(ax, img: np.ndarray, title: str = "", *, cmap: str | None = None) -> None:
    """Display *img* on *ax* with no tick marks."""
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ──────────────────────────────────────────────────────────────────────────────
# dSprites visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────


def plot_factor_scatter(
    ax,
    emb: np.ndarray,
    vals: np.ndarray,
    name: str,
    kind: str,
    label_map: dict | None = None,
) -> None:
    """
    Scatter plot of a 2-D embedding coloured by one generative factor.

    Parameters
    ----------
    kind : {"categorical", "continuous"}
    label_map : dict mapping int → str, used when kind=="categorical".
    """
    if kind == "categorical":
        for v in np.unique(vals):
            lbl = label_map.get(int(v), str(int(v))) if label_map else str(int(v))
            ax.scatter(emb[vals == v, 0], emb[vals == v, 1], s=1, label=lbl)
        ax.legend(title=name, fontsize=6, markerscale=4)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, s=1, cmap="viridis")
        plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)


# ──────────────────────────────────────────────────────────────────────────────
# CelebA visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_attr_scatter(
    ax,
    emb: np.ndarray,
    attr_vals: np.ndarray,
    name: str,
) -> None:
    """Scatter coloured by a binary CelebA attribute (0 = absent, 1 = present)."""
    colors = ["steelblue", "tomato"]
    labels_txt = [f"No {name}", name]
    for v in (0, 1):
        idx = attr_vals == v
        ax.scatter(
            emb[idx, 0], emb[idx, 1],
            s=1, alpha=0.5, color=colors[v], label=labels_txt[v],
        )
    ax.legend(fontsize=6, markerscale=4, loc="best")


# ──────────────────────────────────────────────────────────────────────────────
# VAE loading
# ──────────────────────────────────────────────────────────────────────────────

def load_vae_checkpoint(path: Path, device) -> tuple:
    """Load a generative model (ConvVAE or RealNVP flow) from a .pt checkpoint, dispatching
    on the saved model_config type. Returns (model, ckpt_dict)."""
    from src.models.vae import ConvVAE, VAEConfig
    from src.models.flow import RealNVP, FlowConfig
    torch.serialization.add_safe_globals([VAEConfig, FlowConfig])
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    cfg   = ckpt["model_config"]
    model = (RealNVP(cfg) if type(cfg).__name__ == "FlowConfig" else ConvVAE(cfg)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ──────────────────────────────────────────────────────────────────────────────
# Sweep discovery & style
# ──────────────────────────────────────────────────────────────────────────────

def discover_sweep_keys(sweep_dir: Path, *, lam_max: float | None = None,
                        prefix: str = "lam", strict: bool = True) -> list:
    """
    Scan *sweep_dir* for completed {prefix}{x}_ep{y} folders and return sorted
    (param, ep) tuples.  Pass lam_max to exclude high-value collapsed runs.

    prefix="lam"   → lam{x}_ep{y}          (NegGrad+, Finetune)
    prefix="alpha" → alpha{x}_ep{y}        (Feature Unlearning)
    prefix="a"     → a{x}_c{c}_ep{y}       (RMU — secondary components tolerated)

    The LEADING slot (between the prefix and `_ep`) is always tolerant of one
    extra `_<letter><number>` component -- this is what lets `prefix="a"`
    parse RMU's `a{alpha}_c{c}_ep{epoch}` naming down to an (alpha, epoch)
    key at a fixed `c` (see TestSweepKeyParsingRegression in
    tests/test_regressions.py, which pins this down explicitly). It was never
    the source of the bug below and stays tolerant regardless of `strict`.

    strict=True (default) additionally requires nothing follow `_ep{y}` --
    this is the "one homogeneous default-hyperparameter grid" a sweep
    notebook expects, and the only way a caller reconstructing a checkpoint
    path from (param, ep) alone (e.g. `SWEEP_DIR / f"alpha{a}_ep{e}" /
    "final.pt"`) is guaranteed to hit a real directory.

    strict=False restores the old permissive matching, which also tolerates
    a TRAILING `_<letter><number>` suffix after `_ep{y}` -- `_b80` (batch
    budget), `_u1` (uniform steering vector), `_n1` (L1 pixel norm), `_ms0.5`
    (MS-SSIM weight), `_pt1` (translate to threshold), `_te1` (trained
    encoder), `_ea<W>` (ELBO anchor), `_s<seed>` (steering-vector seed). Those
    markers are DROPPED from the returned key, so a directory listing that
    mixes variants collapses onto the same (param, ep) key -- e.g.
    `alpha0.02_ep10_n1_ms0.5_pt1` (a paper-faithful Feature Unlearning run,
    see run_queue9.ps1) yields the same key as a hypothetical plain
    `alpha0.02_ep10`, even when only the suffixed directory actually exists on
    disk. A caller that path-joins the collapsed key back together gets a
    FileNotFoundError pointing at a directory that was never created. Only
    pass strict=False when scanning ONE named variant at a time (filter
    folder names first), never across a mixed listing.
    """
    import re
    keys = []
    if not sweep_dir.exists():
        return keys
    leading = rf"{re.escape(prefix)}([0-9.]+?)(?:_[a-z]+[0-9.]+)*_ep([0-9]+)"
    trailing = "" if strict else r"(?:_[a-z]+[0-9.]+)*"
    pattern = re.compile(leading + trailing)
    for folder in sorted(sweep_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = pattern.fullmatch(folder.name)
        if not m or not (folder / "final.pt").exists():
            continue
        param, ep = float(m.group(1)), int(m.group(2))
        if lam_max is not None and param > lam_max:
            continue
        keys.append((param, ep))
    return keys


# ──────────────────────────────────────────────────────────────────────────────
# Metrics post-processing
# ──────────────────────────────────────────────────────────────────────────────

def compute_progress(
    results: dict,
    sweep_keys: list,
    metric_keys: list,
    *,
    min_denom: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Normalised progress toward oracle for every sweep run.

        progress_m = (run - baseline) / (oracle - baseline)

    Per-metric progress is always reported.  A metric is EXCLUDED from the
    aggregate "mean" when |oracle - baseline| < ``min_denom``, because a tiny
    denominator turns metric noise into enormous apparent progress.

    Why this guard exists
    ---------------------
    On dSprites the oracle -- which has never seen a heart -- scores like this:

        metric          baseline   oracle   denom
        linear_auroc      0.6307   0.5583  -0.0723   usable
        mlp_auroc         0.9868   0.9935  +0.0067   DEGENERATE, and inverted
        knn_cohesion      0.9538   0.9681  +0.0143   DEGENERATE, and inverted
        lift_norm         0.9308   0.9522  +0.0214   DEGENERATE, and inverted
        dir_norm          0.3625   0.1958  -0.1667   usable

    Hearts are out-of-distribution for the oracle, so its encoder collapses them
    into a tight OOD sink (within-cluster spread 0.68 vs 0.98 for retain classes,
    heart ELBO 421 vs ~33).  They therefore become MORE cohesive and MORE
    MLP-separable after genuine erasure, not less.  With a denominator of 0.0067,
    a 0.01 wobble in mlp_auroc registers as 1.5 "progress" and swamps the mean.

    Metrics excluded here are still returned individually -- inspect them
    directly rather than through the aggregate.
    """
    def _prog(rv, bv, ov=None):
        if ov is None:
            return rv - bv          # raw delta when no oracle exists
        d = ov - bv
        return 0.0 if abs(d) < 1e-10 else (rv - bv) / d

    has_oracle = "oracle" in results

    # Decide once which metrics can carry a meaningful normalised progress.
    if has_oracle:
        denoms   = {m: results["oracle"][m] - results["baseline"][m] for m in metric_keys}
        usable   = [m for m in metric_keys if abs(denoms[m]) >= min_denom]
        degenerate = [m for m in metric_keys if m not in usable]
    else:
        denoms, usable, degenerate = {}, list(metric_keys), []

    if verbose and degenerate:
        print(f"  [compute_progress] excluded from mean (|oracle-baseline| < {min_denom}): "
              + ", ".join(f"{m} (denom={denoms[m]:+.4f})" for m in degenerate))
        print(f"  [compute_progress] mean computed over: {', '.join(usable) or '(none)'}")

    progress = {}
    for key in sweep_keys:
        progress[key] = {
            m: _prog(
                results[key][m],
                results["baseline"][m],
                results["oracle"][m] if has_oracle else None,
            )
            for m in metric_keys
        }
        progress[key]["mean"] = (
            float(np.mean([progress[key][m] for m in usable])) if usable else float("nan")
        )
        progress[key]["_mean_over"] = usable
    return progress


# ──────────────────────────────────────────────────────────────────────────────
# Feature-direction translation helpers  (used by feature_unlearn notebooks)
# ──────────────────────────────────────────────────────────────────────────────

def translate_latents_np(mu: np.ndarray, z_e: np.ndarray) -> np.ndarray:
    """
    Remove the component of mu along z_e (numpy, batched).
    mu_translated = mu − proj(mu, z_e)
    where proj(mu, z_e) = (mu @ z_e / ‖z_e‖²) · z_e
    mu : (N, D)  z_e : (D,)  → returns (N, D)
    """
    z_e_norm_sq = float(np.dot(z_e, z_e))
    proj = (mu @ z_e) / z_e_norm_sq          # (N,)
    return mu - np.outer(proj, z_e)


# ──────────────────────────────────────────────────────────────────────────────
# Metric caches  (same mtime-invalidation pattern as encode_dataset_cached)
# ──────────────────────────────────────────────────────────────────────────────

# Bump when the metric definitions change, so stale caches are recomputed rather
# than silently served.  mtime alone cannot catch a change in *how* a metric is
# computed, only a change in the checkpoint.
#   v2 — mlp_probe now class-balances its training half (src/evaluation/probing.py)
BESPOKE_METRICS_VERSION = 2

# Bump when the per-*-ELBO computation changes. Same contract as above: the mtime
# guard only catches a NEW checkpoint, not a change in how the metric is computed.
ELBO_CACHE_VERSION = 1


def bespoke_metrics_cached(
    mu: np.ndarray,
    sv: np.ndarray,
    ckpt_path: Path,
    *,
    forget_val: int = 2,
) -> dict:
    """
    Compute linear probe, MLP probe, kNN cohesion, and direction norm,
    caching the result next to the checkpoint.

    Cache file : <ckpt_path.parent>/cache/bespoke_metrics.json
    Invalidated: when the checkpoint is newer than the cache file, OR when
                 BESPOKE_METRICS_VERSION no longer matches the cached value.
    """
    from sklearn.model_selection import train_test_split
    from src.evaluation.probing import linear_probe, mlp_probe
    from src.evaluation.knn import knn_concept_cohesion

    cache_dir  = ckpt_path.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # KEY ON THE CHECKPOINT, not just the directory.
    #
    # A run directory holds best.pt / best_tradeoff.pt / final.pt /
    # epoch_NNNN.pt, and different analyses legitimately want different ones:
    # the sweep notebooks read final.pt (see rmu_sweep_dsprites), while the
    # master table and cross_method notebooks resolve best_tradeoff.pt. With a
    # single `bespoke_metrics.json` per directory those two consumers overwrite
    # each other's work and silently read each other's numbers -- which is how
    # the dSprites RMU champion came to be reported with final.pt's metrics
    # (epoch 20, 14% over the retain budget) while every table labelled them
    # epoch 11.
    #
    # This is NOT one consumer being wrong. Both are correct for their own
    # question; the cache simply could not represent both answers. Stem-keying
    # lets them coexist, the same way cross_method already keys encodings by
    # stem (scripts/warm/README.md).
    cache_file = cache_dir / f"bespoke_metrics_{ckpt_path.stem}.json"

    # Legacy unsuffixed files predate stem-keying. Accept one only if it says
    # which checkpoint it describes AND that matches -- an unstamped legacy file
    # cannot be attributed to a checkpoint, so trusting it is exactly the
    # mistake this change exists to prevent. Migrate on hit so the cost is paid
    # once.
    legacy = cache_dir / "bespoke_metrics.json"
    if not cache_file.exists() and legacy.exists():
        try:
            old = json.load(open(legacy))
        except Exception:
            old = {}
        if old.get("_ckpt") == ckpt_path.name:
            json.dump(old, open(cache_file, "w"), indent=2)
            print(f"  [migrated]   bespoke_metrics  {ckpt_path.parent.name} "
                  f"-> {cache_file.name}")

    if cache_file.exists():
        data = json.load(open(cache_file))
        fresh   = cache_file.stat().st_mtime >= ckpt_path.stat().st_mtime
        current = data.get("_version") == BESPOKE_METRICS_VERSION
        # WHICH checkpoint produced this file.  A run directory holds several
        # (best.pt / best_tradeoff.pt / final.pt / epoch_NNNN.pt) but there is
        # only ONE cache file, so a cache warmed from one checkpoint was being
        # served for another -- silently, since both sides "work".
        #
        # Measured on dsprites_rmu/a200.0_c1.0_ep20_s5: the cache held final.pt
        # (epoch 20, val 46.26 -- 14% OVER the retain budget) while every
        # analysis resolved best_tradeoff.pt (epoch 11, val 40.21) and reported
        # those numbers as the champion's.  That mismatch is the entirety of the
        # "RMU makes hearts more linearly decodable" result: linear_auroc 0.6812
        # / dir_norm 0.5650 belong to final.pt, against 0.6224 / 0.3762 for the
        # checkpoint actually being claimed.
        #
        # This is the failure require_exact guards at the checkpoint layer,
        # reappearing at the cache layer.  encodings_<split>.npz already keys on
        # the checkpoint stem for exactly this reason (scripts/warm/README);
        # bespoke_metrics.json never did.
        #
        # Legacy files carry no _ckpt.  They are accepted rather than mass
        # invalidated -- most are unambiguous and recomputing every CelebA entry
        # costs days -- and the genuinely at-risk ones are deleted out of band
        # by scripts/audit_metric_caches.py, which knows each run's trade-off
        # epoch and so can tell which are actually suspect.
        same_ckpt = data.get("_ckpt") in (None, ckpt_path.name)
        if fresh and current and same_ckpt:
            print(f"  [cache hit]  bespoke_metrics  {ckpt_path.parent.name}")
            return data
        reason = ("checkpoint newer" if not fresh
                  else f"cached from {data.get('_ckpt')}, asked for {ckpt_path.name}"
                  if not same_ckpt
                  else f"metric version {data.get('_version')} != {BESPOKE_METRICS_VERSION}")
        print(f"  [cache stale] bespoke_metrics  {ckpt_path.parent.name}  ({reason})")

    print(f"  [computing]  bespoke_metrics  {ckpt_path.parent.name} ...")
    y = (sv == forget_val).astype(int)
    mu_tr, mu_te, y_tr, y_te = train_test_split(
        mu, y, test_size=0.5, stratify=y, random_state=0
    )
    lin  = linear_probe(mu_tr, y_tr, mu_te, y_te)
    mlp_ = mlp_probe(mu_tr, y_tr, mu_te, y_te)
    knn  = knn_concept_cohesion(mu, y)
    d    = mu[sv == forget_val].mean(0) - mu[sv != forget_val].mean(0)
    metrics = {
        "linear_auroc": lin["auroc"],
        "mlp_auroc":    mlp_["auroc"],
        "knn_cohesion": knn["cohesion"],
        "lift_norm":    knn["lift_norm"],
        "dir_norm":     float(np.linalg.norm(d)),
        "_version":     BESPOKE_METRICS_VERSION,
        "_ckpt":        ckpt_path.name,   # which checkpoint these describe
    }
    json.dump(metrics, open(cache_file, "w"), indent=2)
    return metrics


def per_shape_elbo_cached(
    model,
    loader,
    device,
    ckpt_path: Path,
    *,
    forget_col: int = 1,
) -> dict:
    """
    Compute per-shape ELBO for squares (0), ellipses (1), and hearts (2),
    caching the result next to the checkpoint.

    Cache file : <ckpt_path.parent>/cache/shape_elbo_<stem>.json
    Invalidated: checkpoint newer than cache, ELBO_CACHE_VERSION change, or a
                 cache written from a different checkpoint (stem-keyed + _ckpt
                 stamped, so a multi-checkpoint dir cannot serve the wrong one --
                 the cache_checkpoint_mismatch failure class).
    Returns    : {0: float, 1: float, 2: float}
    """
    cache_dir  = ckpt_path.parent / "cache"
    cache_file = cache_dir / f"shape_elbo_{ckpt_path.stem}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        raw = json.load(open(cache_file))
        fresh = cache_file.stat().st_mtime >= ckpt_path.stat().st_mtime
        current = raw.get("_version") == ELBO_CACHE_VERSION
        same_ckpt = raw.get("_ckpt") in (None, ckpt_path.name)
        if fresh and current and same_ckpt:
            data = {int(k): v for k, v in raw.items() if not k.startswith("_")}
            print(f"  [cache hit]  shape_elbo       {ckpt_path.parent.name}")
            return data
        print(f"  [cache stale] shape_elbo       {ckpt_path.parent.name}")

    print(f"  [computing]  shape_elbo       {ckpt_path.parent.name} ...")
    model.eval()
    sums   = {s: 0.0 for s in [0, 1, 2]}
    counts = {s: 0   for s in [0, 1, 2]}
    with torch.no_grad():
        for imgs, _, lbl_cls in loader:
            for s in [0, 1, 2]:
                mask = lbl_cls[:, forget_col] == s
                n = mask.sum().item()
                if n > 0:
                    out = model(imgs[mask].to(device), kl_weight=1.0)
                    sums[s]   += out.loss.item() * n
                    counts[s] += n
    data = {s: sums[s] / counts[s] if counts[s] else float("nan") for s in [0, 1, 2]}
    payload = {str(k): v for k, v in data.items()}
    payload["_version"], payload["_ckpt"] = ELBO_CACHE_VERSION, ckpt_path.name
    json.dump(payload, open(cache_file, "w"), indent=2)
    return data


def per_concept_elbo_cached(
    model,
    forget_loader,
    retain_loader,
    device,
    ckpt_path: Path,
) -> dict:
    """
    Compute per-concept ELBO for a binary forget/retain split (e.g. CelebA Eyeglasses).

    Averages the ELBO over all batches in each loader separately.

    Cache file : <ckpt_path.parent>/cache/concept_elbo_<stem>.json
    Invalidated: checkpoint newer than cache, ELBO_CACHE_VERSION change, or a
                 cache written from a different checkpoint (stem-keyed + _ckpt
                 stamped).
    Returns    : {0: retain_elbo, 1: forget_elbo}
    """
    cache_dir  = ckpt_path.parent / "cache"
    cache_file = cache_dir / f"concept_elbo_{ckpt_path.stem}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        raw = json.load(open(cache_file))
        fresh = cache_file.stat().st_mtime >= ckpt_path.stat().st_mtime
        current = raw.get("_version") == ELBO_CACHE_VERSION
        same_ckpt = raw.get("_ckpt") in (None, ckpt_path.name)
        if fresh and current and same_ckpt:
            data = {int(k): v for k, v in raw.items() if not k.startswith("_")}
            print(f"  [cache hit]  concept_elbo     {ckpt_path.parent.name}")
            return data
        print(f"  [cache stale] concept_elbo     {ckpt_path.parent.name}")

    print(f"  [computing]  concept_elbo     {ckpt_path.parent.name} ...")
    model.eval()

    def _avg_elbo(loader):
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
                out = model(x, kl_weight=1.0)
                total += out.loss.item() * x.size(0)
                n     += x.size(0)
        return total / n if n else float("nan")

    data = {0: _avg_elbo(retain_loader), 1: _avg_elbo(forget_loader)}
    payload = {str(k): v for k, v in data.items()}
    payload["_version"], payload["_ckpt"] = ELBO_CACHE_VERSION, ckpt_path.name
    json.dump(payload, open(cache_file, "w"), indent=2)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# RMU run discovery (seed-aware)
# ──────────────────────────────────────────────────────────────────────────────
#
# RMU directories are  a{alpha}_c{c}_ep{ep}[_s{seed}].  discover_sweep_keys
# deliberately DROPS trailing variant markers, so with prefix="a" it collapses
# every seed of a config onto the same (alpha, ep) key.  That is fine for a
# single-seed view and wrong for anything else, hence these.

def rmu_run_name(alpha, ep, *, c: float = 1.0, u_seed: int = 0) -> str:
    """Directory name for one RMU run -- mirrors methods.RMU.run_name()."""
    name = f"a{float(alpha)}_c{float(c)}_ep{int(ep)}"
    if u_seed:
        name += f"_s{int(u_seed)}"
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Fisher scrubbing sweep helpers
#
# Structurally simpler than RMU: FisherScrub.one_shot = True, so every run has
# exactly one epoch and final.pt IS the trade-off checkpoint whenever one
# exists -- there is no "best.pt vs trade-off epoch differ" question to ask,
# because there is no second epoch for them to differ over. The sweep axis is
# noise_scale, with fisher_seed as the repeat axis (mirrors RMU's steering-
# vector seed: the perturbation is a random draw, so one run is one sample).
# ──────────────────────────────────────────────────────────────────────────────

def fisher_run_name(noise_scale, *, fisher_seed: int = 0) -> str:
    """Directory name for one Fisher run -- mirrors methods.FisherScrub.run_name().

    Omits the _fs{fisher_samples} segment: every run in this project used the
    default sample count, so it never appears in a real directory name. If a
    future sweep varies --fisher-samples, this helper needs that segment added
    or it will stop matching.
    """
    name = f"sig{np.format_float_positional(float(noise_scale), trim='-')}_ep1"
    if fisher_seed:
        name += f"_s{int(fisher_seed)}"
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Cross-method comparison
# ──────────────────────────────────────────────────────────────────────────────

def collect_method_champions(dataset: str, methods: dict, budget: float,
                             *, results_root: Path = Path("results"),
                             require_exact: bool = True) -> dict:
    """Best-trading-off run per method, using the shared selection rule.

    `methods` maps method name -> forget_better ("higher"/"lower").  Returns
    {method: champion_dict}, omitting methods with no run inside budget --
    which is itself reportable, not an error.

    require_exact defaults to TRUE here, unlike champion_run itself.  Every
    caller of this function goes on to LOAD the weights and compute metrics
    from them, and without it resolve_checkpoint silently substitutes final.pt
    when the trade-off epoch was never saved -- so the analysis runs on a
    different model than the one the table claims.  Measured consequence on
    celeba_finetune: the loose rule selects lam0.02_ep20 at epoch 15
    (val 407.14, inside a 417.18 budget) and loads epoch 20 at val 8022,
    reporting a concept direction norm of 50.0 against the baseline's 2.1 and
    CKA-to-baseline of 0.06.  That reads as a spectacular unlearning result and
    is entirely an artefact.  With require_exact the champion is lam0.01_ep30
    at val 370.03, and direction norm 1.55.

    Pass require_exact=False only to reproduce an older table, and report
    ckpt_is_exact alongside it if you do.
    """
    from src.evaluation.selection import champion_run
    from src.unlearning.spec import method_run_filter
    out = {}
    for name, forget_better in methods.items():
        sweep_dir = Path(results_root) / dataset / f"{dataset}_{name}"
        champ = champion_run(sweep_dir, forget_better, budget,
                             name_filter=method_run_filter(name),
                             require_exact=require_exact)
        if champ:
            out[name] = champ
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Linear concept erasure — LEACE / INLP
# ──────────────────────────────────────────────────────────────────────────────
# Two shapes, deliberately different from every other method's sweep:
#
#   LEACE has NO hyperparameter, so it is one deterministic run per dataset --
#   there is nothing to sweep and no seed-variance study to run (contrast
#   Fisher, where only 2 of 3 seeds stayed in budget at the same noise_scale).
#   Its single point answers "what does perfect linear guardedness cost", and
#   that cost is not a tuning artefact.
#
#   INLP's --n-iter IS a sweep axis: each round projects out one more rank of
#   mu-space, so k traces a damage-vs-erasure curve that LEACE's point sits on.
#   k is bounded by latent_dim (8 on dSprites, 256 on CelebA).


# ── Probe-recovery curves ─────────────────────────────────────────────────────

RECOVERY_VERSION = 1


def probe_recovery_cached(mu, sv, ckpt_path: Path, *, forget_val: int = 2,
                          **kwargs) -> dict:
    """probe_recovery_curve, cached next to the checkpoint.

    Cache file : <ckpt_path.parent>/cache/probe_recovery_<stem>.json
    Invalidated: checkpoint newer than cache, RECOVERY_VERSION changed, the
                 PROTOCOL changed, or the cache describes a DIFFERENT
                 checkpoint in the same directory.

    THE SAME BUG bespoke_metrics_cached WAS FIXED FOR, found here too and only
    caught because a later reader (scripts/build_master_table.py) recomputed
    the RMU dSprites champion directly and got a different answer than the
    cache: `probe_recovery.json` was one file per DIRECTORY with a freshness
    check that only compares mtime against WHATEVER checkpoint is currently in
    hand. Once any checkpoint's curve is cached, the file's mtime is newer
    than every checkpoint's mtime forever, so the freshness check passes for
    ANY of them -- final.pt's curve was served for best_tradeoff.pt requests.

    Measured impact on dsprites_rmu/a200.0_c1.0_ep20_s5: the cache held
    final.pt's curve (n_to_recover mlp@0.7 = 64) while the champion is
    best_tradeoff.pt (n_to_recover mlp@0.7 = 128 -- TIED with the oracle, not
    below it). A conclusion already written up ("RMU is the most
    passively-recoverable trained method on dSprites, below even the oracle")
    was retracted because of this.

    Fixed the same way: stem-key the cache file and stamp `_ckpt`.
    """
    from src.evaluation.recovery import probe_recovery_curve

    cache_dir = ckpt_path.parent / "cache"
    cache_file = cache_dir / f"probe_recovery_{ckpt_path.stem}.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    want_grid = list(kwargs.get("n_grid", ())) or None

    # Legacy unsuffixed file: migrate on a stamp match, never trust blindly.
    legacy = cache_dir / "probe_recovery.json"
    if not cache_file.exists() and legacy.exists():
        try:
            old = json.load(open(legacy))
        except Exception:
            old = {}
        if old.get("_ckpt") == ckpt_path.name:
            json.dump(old, open(cache_file, "w"), indent=2)
            print(f"  [migrated]   probe_recovery  {ckpt_path.parent.name} "
                  f"-> {cache_file.name}")

    if cache_file.exists():
        data = json.load(open(cache_file))
        params = data.get("params", {})
        fresh   = cache_file.stat().st_mtime >= ckpt_path.stat().st_mtime
        current = data.get("_version") == RECOVERY_VERSION
        same_ckpt = data.get("_ckpt") in (None, ckpt_path.name)
        same_protocol = (
            params.get("n_seeds") == kwargs.get("n_seeds", 5)
            and params.get("forget_val") == int(forget_val)
            and (want_grid is None or data.get("_requested_grid") == want_grid)
        )
        if fresh and current and same_ckpt and same_protocol:
            print(f"  [cache hit]  probe_recovery  {ckpt_path.parent.name}")
            return data
        why = ("checkpoint newer" if not fresh else
               "version changed" if not current else
               f"cached from {data.get('_ckpt')}, asked for {ckpt_path.name}" if not same_ckpt
               else "protocol changed")
        print(f"  [cache stale] probe_recovery  {ckpt_path.parent.name}  ({why})")

    print(f"  [computing]  probe_recovery  {ckpt_path.parent.name} ...")
    out = probe_recovery_curve(mu, sv, forget_val=forget_val, **kwargs)
    out["_version"] = RECOVERY_VERSION
    out["_requested_grid"] = want_grid
    out["_ckpt"] = ckpt_path.name
    json.dump(out, open(cache_file, "w"), indent=2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Qualitative sanity checks
#
# Everything above this line reduces a model to numbers. These functions show
# what the model actually DOES, which catches a category of problem no scalar
# does: a metric can look healthy while the decoder emits grey mush, and a
# checkpoint loaded from the wrong epoch produces plausible-looking numbers but
# obviously broken images. Both failure modes have occurred in this project
# (RESEARCH_LEDGER R1, E14), and both would have been visible instantly here.
#
# These are also the plots that make a latent claim legible to a reader who
# does not already believe the metrics.
# ──────────────────────────────────────────────────────────────────────────────
