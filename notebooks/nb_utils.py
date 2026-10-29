"""
notebooks/nb_utils.py
---------------------
Shared helpers for all project notebooks.  Import after setting ROOT:

    from notebooks.nb_utils import find_project_root, ...

Or, if CWD is already the project root:

    from notebooks.nb_utils import ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Project-root detection
# ──────────────────────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    """
    Return the project root and set it as CWD.

    Resolution order (first match wins):
      1. VS Code exposes the notebook's absolute path as ``__vsc_ipynb_file__``
         in the kernel globals — go up two levels (notebooks/ → project root).
      2. Walk *up* from CWD until a directory that contains both ``src/`` and
         ``configs/`` is found.
      3. Fall back to one level above CWD (classic Jupyter: kernel starts in
         the ``notebooks/`` directory).

    The function is **idempotent**: calling it multiple times always returns
    the same directory and is safe even after ``os.chdir`` has already run.
    """
    # Strategy 1 — VS Code (walk up from the notebook's directory)
    nb_file = sys._getframe(1).f_globals.get("__vsc_ipynb_file__")
    if nb_file:
        candidate = Path(nb_file).resolve().parent
        while candidate != candidate.parent:
            if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
                os.chdir(candidate)
                return candidate
            candidate = candidate.parent

    # Strategy 2 — walk up from CWD
    candidate = Path(".").resolve()
    while candidate != candidate.parent:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            os.chdir(candidate)
            return candidate
        candidate = candidate.parent

    # Strategy 3 — assume kernel CWD is notebooks/
    root = Path(".").resolve().parent
    os.chdir(root)
    return root


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


def plot_latent_variance(
    mu: np.ndarray,
    *,
    title: str = "Active latent dimensions",
    out_path: Path | str | None = None,
) -> None:
    """Bar chart of per-dimension variance of the encoder mean μ."""
    dim_var = mu.var(axis=0)
    x = np.arange(mu.shape[1])

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, dim_var, color="steelblue", label="mu variance")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="prior variance = 1")
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Variance of μ across dataset")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()

    for bar, v in zip(bars, dim_var):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.02 * dim_var.max(),
            f"{v:.3f}", ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# dSprites visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_3d_plotly_grid(
    emb3d: np.ndarray,
    labels: np.ndarray,
    factors: list[tuple],
    title: str,
    axis_labels: tuple[str, str, str] = ("Dim 1", "Dim 2", "Dim 3"),
):
    """
    Build a 2×3 Plotly 3-D scatter grid, one panel per generative factor.

    Parameters
    ----------
    emb3d : (N, 3) array
        3-D embedding (PCA / UMAP / etc.).
    labels : (N, F) array
        Generative-factor labels; one column per factor.
    factors : list of (name, col, kind, label_map)
        ``kind`` is ``"categorical"`` or ``"continuous"``.
        ``label_map`` maps int value → display string (or None).
    title : str
        Figure title.
    axis_labels : (x, y, z) strings
        Scene axis labels.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    rows, cols = 2, 3
    cb_x = [0.30, 0.65, 1.00]
    cb_y = [0.78, 0.22]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"}] * cols] * rows,
        subplot_titles=[name for name, *_ in factors],
    )

    for i, (name, col, kind, label_map) in enumerate(factors):
        r, c = i // cols + 1, i % cols + 1
        color_vals = labels[:, col].reshape(-1)

        if kind == "categorical":
            colorscale = [[0.0, "#4C72B0"], [0.5, "#DD8452"], [1.0, "#55A868"]]
            uv = sorted(np.unique(color_vals))
            cb_extra = (
                dict(tickvals=uv, ticktext=[label_map.get(int(v), str(int(v))) for v in uv])
                if label_map else {}
            )
        else:
            colorscale, cb_extra = "Viridis", {}

        fig.add_trace(
            go.Scatter3d(
                x=emb3d[:, 0], y=emb3d[:, 1], z=emb3d[:, 2],
                mode="markers", name=name,
                marker=dict(
                    size=1.5, color=color_vals,
                    colorscale=colorscale, opacity=0.6, showscale=True,
                    colorbar=dict(
                        thickness=10, len=0.35,
                        x=cb_x[c - 1], y=cb_y[r - 1],
                        title=dict(text=name, font=dict(size=9)),
                        tickfont=dict(size=8), **cb_extra,
                    ),
                ),
            ),
            row=r, col=c,
        )
        fig.update_scenes(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            row=r, col=c,
        )

    fig.update_layout(height=900, width=1300, title=title, showlegend=False)
    return fig


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


def plot_embedding_grid_dsprites(
    emb: np.ndarray,
    labels: np.ndarray,
    factors: list[tuple],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    out_path: Path | str | None = None,
) -> None:
    """
    2×3 matplotlib grid of 2-D scatter plots, one per dSprites factor.

    Parameters
    ----------
    emb : (N, 2) array — 2-D embedding.
    labels : (N, F) array — generative-factor labels.
    factors : list of (name, col, kind, label_map).
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    fig.suptitle(title, fontsize=11)

    for ax, (name, col, kind, label_map) in zip(axes.flat, factors):
        plot_factor_scatter(ax, emb, labels[:, col], name, kind, label_map)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)

    for ax in axes.flat[len(factors):]:
        ax.set_visible(False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


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


def plot_embedding_grid_celeba(
    emb: np.ndarray,
    attrs: np.ndarray,
    plot_attrs: list[tuple[str, int]],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    out_path: Path | str | None = None,
) -> None:
    """
    2×3 matplotlib grid of 2-D scatter plots, one per CelebA attribute.

    Parameters
    ----------
    emb : (N, 2) array — 2-D embedding.
    attrs : (N, 40) array — binary attribute matrix (0/1).
    plot_attrs : list of (display_name, 0-based column index) — exactly 6 entries.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle(title, fontsize=12)

    for ax, (name, col) in zip(axes.flat, plot_attrs):
        plot_attr_scatter(ax, emb, attrs[:, col], name)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(plot_attrs):]:
        ax.set_visible(False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


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


def make_sweep_style(lambdas: list, epochs: list) -> tuple:
    """
    Build colour / marker style maps for a lambda × epochs sweep grid.

    Returns
    -------
    lam_norm   : {lam: float in [0,1]}  position in plasma colour scale
    ep_markers : {ep: marker_str}       matplotlib marker per epoch value
    lam_color  : {lam: rgba_str}        Plotly colour per lambda value
    """
    import plotly.colors as pc
    lam_norm  = {l: i / max(len(lambdas) - 1, 1) for i, l in enumerate(lambdas)}
    ep_markers = {ep: ["o", "s", "^", "D", "v", "P", "X", "*"][i % 8]
                  for i, ep in enumerate(epochs)}
    palette   = pc.sample_colorscale("plasma", [lam_norm[l] for l in lambdas])
    lam_color = {l: palette[i] for i, l in enumerate(lambdas)}
    return lam_norm, ep_markers, lam_color


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


def compute_pareto_front(
    sweep_keys: list,
    shape_elbo: dict,
    val_losses: dict,
    progress: dict,
    retain_thresh: float,
    ht_base: float,
) -> tuple:
    """
    Pareto front of unlearning quality (↑ heart ELBO) vs retain quality
    (↓ retain avg ELBO) subject to retain_avg ≤ retain_thresh.

    Sweeping retain low→high, a point is on the front only if it improves
    heart ELBO over all prior points.  BEST = Pareto point with highest heart.

    Returns: (all_cands, passed_cands, pareto, BEST)
    """
    all_cands = [
        {
            "key":      k,
            "label":    f"lam={k[0]} ep={k[1]}",
            "sq":       shape_elbo[k][0],
            "el":       shape_elbo[k][1],
            "ht":       shape_elbo[k][2],
            "ret":      (shape_elbo[k][0] + shape_elbo[k][1]) / 2,
            "ht_rise":  shape_elbo[k][2] - ht_base,
            "val_loss": val_losses.get(k, float("nan")),
            "progress": progress.get(k, {}).get("mean", float("nan")),
        }
        for k in sweep_keys
    ]
    passed = [c for c in all_cands if c["ret"] <= retain_thresh]
    passed.sort(key=lambda x: x["ret"])

    pareto, best_ht = [], float("-inf")
    for c in passed:
        if c["ht"] > best_ht:
            pareto.append(c)
            best_ht = c["ht"]

    BEST = pareto[-1] if pareto else None
    return all_cands, passed, pareto, BEST


def print_sweep_tables(
    results: dict,
    shape_elbo: dict,
    val_losses: dict,
    progress: dict,
    all_keys: list,
    metric_names: list,
    metric_keys: list,
    ht_base: float,
) -> None:
    """Print bespoke-metrics and per-shape ELBO summary tables to stdout."""
    def _lab(key):
        return f"lam={key[0]} ep={key[1]}" if isinstance(key, tuple) else key

    print()
    print("Bespoke metrics")
    hdr = (f"{'Run':<18}" +
           "".join(f"{n:>14}" for n in metric_names) +
           f"{'ValLoss':>10}  {'Progress':>9}")
    print(hdr)
    print("-" * len(hdr))
    for key in all_keys:
        lab = _lab(key)
        row = f"{lab:<18}" + "".join(f"{results[key][m]:>14.5f}" for m in metric_keys)
        vl  = f"{val_losses[key]:>10.4f}" if key in val_losses else f"{'—':>10}"
        pg  = f"{progress[key]['mean']:>+9.3f}" if key in progress else f"{'—':>9}"
        print(row + vl + "  " + pg)

    print()
    print("Per-shape ELBO")
    hdr2 = (f"{'Run':<18}{'Square':>12}{'Ellipse':>12}"
            f"{'Heart':>12}{'RetainAvg':>12}{'HeartRise':>12}")
    print(hdr2)
    print("-" * len(hdr2))
    for key in all_keys:
        lab = _lab(key)
        e   = shape_elbo[key]
        ret = (e[0] + e[1]) / 2
        print(f"{lab:<18}{e[0]:>12.4f}{e[1]:>12.4f}"
              f"{e[2]:>12.4f}{ret:>12.4f}{e[2]-ht_base:>+12.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Sweep plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_pareto_front(
    all_cands: list,
    passed_cands: list,
    pareto: list,
    BEST: dict | None,
    shape_elbo: dict,
    retain_thresh: float,
    ret_base: float,
    ht_base: float,
    lam_norm: dict,
    ep_markers: dict,
    lam_color: dict,
    lambdas: list,
    epochs: list,
    *,
    out_path: Path | str | None = None,
) -> None:
    """Scatter of retain avg ELBO vs heart ELBO with Pareto front highlighted."""
    import matplotlib.lines as mlines

    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(9, 6))

    for c in passed_cands:
        lam, ep = c["key"]
        ax.scatter(c["ret"], c["ht"], color=cmap(lam_norm[lam]),
                   marker=ep_markers[ep], s=80, alpha=0.7, zorder=3,
                   edgecolors="black", linewidths=0.4)
        ax.annotate(c["label"], (c["ret"], c["ht"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6)

    for c in all_cands:
        if c["ret"] > retain_thresh:
            ax.scatter(c["ret"], c["ht"], color=cmap(lam_norm[c["key"][0]]),
                       marker="x", s=50, alpha=0.25, zorder=2)

    if pareto:
        ax.plot([c["ret"] for c in pareto], [c["ht"] for c in pareto],
                color="black", linewidth=2, zorder=4, label="Pareto front")

    has_oracle = "oracle" in shape_elbo
    if has_oracle:
        sq_or = shape_elbo["oracle"][0]
        el_or = shape_elbo["oracle"][1]
        ht_or = shape_elbo["oracle"][2]

    ax.scatter([ret_base], [ht_base], s=130, marker="D", color="#2196F3", zorder=5,
               label=f"Baseline (ret={ret_base:.2f}, ht={ht_base:.2f})")
    if has_oracle:
        ax.scatter([(sq_or + el_or) / 2], [ht_or], s=130, marker="D", color="#4CAF50", zorder=5,
                   label=f"Oracle (ret={(sq_or+el_or)/2:.2f}, ht={ht_or:.2f})")
    if BEST:
        ax.scatter([BEST["ret"]], [BEST["ht"]], s=180, marker="*",
                   color="#FF5722", zorder=6, label=f"Best: {BEST['label']}")

    ax.axvline(retain_thresh, color="red",     ls="--", lw=1.2, alpha=0.6,
               label="Retain threshold")
    ax.axhline(ht_base,       color="#2196F3", ls=":",  lw=1.0, alpha=0.5)
    if has_oracle:
        ax.axhline(ht_or,     color="#4CAF50", ls=":",  lw=1.0, alpha=0.5)

    lam_h = [mlines.Line2D([], [], marker="o", color="w",
              markerfacecolor=cmap(lam_norm[l]), markersize=8, label=f"λ={l}")
             for l in lambdas]
    ep_h  = [mlines.Line2D([], [], marker=ep_markers[e], color="grey",
              markersize=8, label=f"{e}ep")
             for e in epochs]
    named_h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=named_h + lam_h + ep_h, fontsize=7, ncol=2)

    # ── Zoom tightly around Pareto front points only ──────────────────────────
    if pareto:
        px   = [c["ret"] for c in pareto]
        py   = [c["ht"]  for c in pareto]
        xpad = max((max(px) - min(px)) * 0.15, 0.02)
        ypad = max((max(py) - min(py)) * 0.15, 0.02)
        # ax.set_xlim(min(px) - xpad, max(px) + xpad)
        # ax.set_ylim(min(py) - ypad, max(py) + ypad)
        ax.set_xlim(0 - xpad, max(px) + xpad)
        ax.set_ylim(0 - ypad, max(py) + ypad)

    ax.set_xlabel("Retain avg ELBO  (lower = less damage to squares+ellipses)", fontsize=10)
    ax.set_ylabel("Heart ELBO  (higher = more forgotten)", fontsize=10)
    ax.set_title("Pareto front — Retain quality vs Forgetting quality", fontsize=11)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_progress_heatmaps(
    progress: dict,
    lambdas: list,
    epochs: list,
    metric_keys: list,
    metric_names: list,
    *,
    out_path: Path | str | None = None,
) -> None:
    """Grid of annotated heatmaps — one per bespoke metric plus mean."""
    n_cols = len(metric_keys) + 1
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4 * n_cols, max(3, len(lambdas))),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle("Progress toward oracle  (0=baseline  1=oracle)", fontsize=11)

    for ax, mkey, mname in zip(axes, metric_keys + ["mean"], metric_names + ["Mean"]):
        grid = np.full((len(lambdas), len(epochs)), np.nan)
        for i, lam in enumerate(lambdas):
            for j, ep in enumerate(epochs):
                if (lam, ep) in progress:
                    grid[i, j] = progress[(lam, ep)].get(mkey, float("nan"))
        vext = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1.0)
        im = ax.imshow(grid, cmap="RdYlGn", vmin=-vext, vmax=vext, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(lambdas)):
            for j in range(len(epochs)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, fontweight="bold")
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([f"{e}ep" for e in epochs], fontsize=7)
        ax.set_yticks(range(len(lambdas)))
        ax.set_yticklabels([f"λ={l}" for l in lambdas], fontsize=7)
        ax.set_title(mname, fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_val_loss_heatmap(
    val_losses: dict,
    lambdas: list,
    epochs: list,
    baseline_val_loss: float,
    *,
    out_path: Path | str | None = None,
) -> None:
    """Annotated heatmap of retain val loss across the lambda × epochs grid."""
    grid = np.full((len(lambdas), len(epochs)), np.nan)
    for i, lam in enumerate(lambdas):
        for j, ep in enumerate(epochs):
            if (lam, ep) in val_losses:
                grid[i, j] = val_losses[(lam, ep)]

    fig, ax = plt.subplots(
        figsize=(max(4, len(epochs) * 1.2), max(3, len(lambdas) * 0.8)))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Val loss")
    for i in range(len(lambdas)):
        for j in range(len(epochs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([f"{e}ep" for e in epochs])
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels([f"λ={l}" for l in lambdas])
    ax.set_title(f"Retain val loss  (baseline={baseline_val_loss:.1f})", fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tradeoff_scatter(
    progress: dict,
    val_losses: dict,
    sweep_keys: list,
    lam_norm: dict,
    ep_markers: dict,
    lam_color: dict,
    lambdas: list,
    epochs: list,
    baseline_val_loss: float,
    retain_val_thresh: float,
    *,
    out_path: Path | str | None = None,
) -> None:
    """Scatter: mean oracle progress vs retain val loss, annotated per run."""
    import matplotlib.lines as mlines

    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(9, 6))

    for lam, ep in sweep_keys:
        key = (lam, ep)
        if key not in progress or key not in val_losses:
            continue
        ax.scatter(progress[key]["mean"], val_losses[key],
                   color=cmap(lam_norm[lam]), marker=ep_markers[ep],
                   s=100, zorder=5, edgecolors="black", linewidths=0.4)
        ax.annotate(f"λ={lam} {ep}ep", (progress[key]["mean"], val_losses[key]),
                    textcoords="offset points", xytext=(5, 3), fontsize=6)

    ax.axhline(baseline_val_loss, color="#2196F3", ls="--", lw=1.5,
               label="Baseline val loss")
    ax.axhline(retain_val_thresh, color="#FF5722", ls=":",  lw=1.2,
               label="1.5x threshold")
    ax.axvline(1.0, color="#4CAF50", ls="--", lw=1.5, label="Oracle progress")
    ax.axvline(0.0, color="grey",    ls=":",  lw=1.0)

    lam_h = [mlines.Line2D([], [], marker="o", color="w",
              markerfacecolor=cmap(lam_norm[l]), markersize=8, label=f"λ={l}")
             for l in lambdas]
    ep_h  = [mlines.Line2D([], [], marker=ep_markers[e], color="grey",
              markersize=8, label=f"{e}ep")
             for e in epochs]
    ax.legend(handles=lam_h + ep_h, fontsize=7, ncol=2)

    ax.set_xlabel("Mean progress toward oracle")
    ax.set_ylabel("Retain val loss")
    ax.set_title("Trade-off: progress vs retain quality")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def build_loss_space_3d(
    sweep_keys: list,
    shape_elbo: dict,
    pareto: list,
    BEST: dict | None,
    retain_thresh: float,
    lam_color: dict,
    lambdas: list,
):
    """
    3-D Plotly scatter in (Square ELBO, Ellipse ELBO, Heart ELBO) space.
    Gold diamonds = Pareto front.  Orange star = BEST.  Green line = oracle dir.
    Returns a go.Figure — caller calls .show() and .write_html().
    """
    import plotly.graph_objects as go

    sq_b = shape_elbo["baseline"][0]
    el_b = shape_elbo["baseline"][1]
    ht_b = shape_elbo["baseline"][2]
    has_oracle = "oracle" in shape_elbo
    if has_oracle:
        sq_o = shape_elbo["oracle"][0]
        el_o = shape_elbo["oracle"][1]
        ht_o = shape_elbo["oracle"][2]

    fig = go.Figure()
    if has_oracle:
        fig.add_trace(go.Scatter3d(
            x=[sq_b, sq_o], y=[el_b, el_o], z=[ht_b, ht_o],
            mode="lines", line=dict(color="#4CAF50", width=6), name="Baseline->Oracle"))

    for key in sweep_keys:
        lam, ep = key
        e      = shape_elbo[key]
        passed = (e[0] + e[1]) / 2 <= retain_thresh
        fig.add_trace(go.Scatter3d(
            x=[sq_b, e[0]], y=[el_b, e[1]], z=[ht_b, e[2]],
            mode="lines", line=dict(color=lam_color[lam], width=1.2),
            opacity=0.3, showlegend=False, hoverinfo="skip"))
        hover = (f"<b>lam={lam} {ep}ep</b><br>sq={e[0]:.4f}<br>el={e[1]:.4f}<br>"
                 f"heart={e[2]:.4f}<br>retain_avg={(e[0]+e[1])/2:.4f}<br>"
                 f"heart_rise={e[2]-ht_b:+.4f}<br>"
                 f"retain_ok={'yes' if passed else 'NO'}")
        fig.add_trace(go.Scatter3d(
            x=[e[0]], y=[e[1]], z=[e[2]], mode="markers",
            marker=dict(size=6 if passed else 4, color=lam_color[lam],
                        symbol="circle" if passed else "x",
                        opacity=1.0 if passed else 0.3,
                        line=dict(color="black", width=0.5)),
            showlegend=False, hovertemplate=hover + "<extra></extra>"))

    fig.add_trace(go.Scatter3d(
        x=[sq_b], y=[el_b], z=[ht_b], mode="markers+text",
        marker=dict(size=12, color="#2196F3", symbol="diamond"),
        text=["Baseline"], textposition="top center", name="Baseline"))
    # Guarded on has_oracle: sq_o/el_o/ht_o are only assigned above when an
    # "oracle" key is present in shape_elbo. Using them unconditionally here
    # raised UnboundLocalError for any caller without one -- which both CelebA
    # sweep notebooks were, until the oracle checkpoint was wired into them.
    if has_oracle:
        fig.add_trace(go.Scatter3d(
            x=[sq_o], y=[el_o], z=[ht_o], mode="markers+text",
            marker=dict(size=12, color="#4CAF50", symbol="diamond"),
            text=["Oracle"], textposition="top center", name="Oracle"))

    if BEST:
        fig.add_trace(go.Scatter3d(
            x=[BEST["sq"]], y=[BEST["el"]], z=[BEST["ht"]], mode="markers+text",
            marker=dict(size=14, color="#FF5722", symbol="diamond",
                        line=dict(color="black", width=2)),
            text=[f"BEST: {BEST['label']}"],
            textposition="top center", name="Best (Pareto)"))

    for pt in pareto:
        fig.add_trace(go.Scatter3d(
            x=[pt["sq"]], y=[pt["el"]], z=[pt["ht"]], mode="markers",
            marker=dict(size=10, color="gold", symbol="diamond",
                        line=dict(color="black", width=1)),
            showlegend=False,
            hovertemplate=(f"<b>Pareto: {pt['label']}</b><br>"
                           f"ret={pt['ret']:.4f}<br>"
                           f"heart={pt['ht']:.4f}<extra></extra>")))

    for l in lambdas:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=8, color=lam_color[l]), name=f"lambda={l}"))

    fig.update_layout(
        title=dict(
            text=("Loss-space 3D — Square / Ellipse / Heart ELBO<br>"
                  "<sup>Gold diamonds = Pareto front | Orange = best | "
                  "Green line = oracle dir</sup>"),
            x=0.5),
        scene=dict(xaxis_title="Square ELBO",
                   yaxis_title="Ellipse ELBO",
                   zaxis_title="Heart ELBO"),
        width=950, height=720,
        margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(font=dict(size=9)))
    return fig


def build_metric_pca_3d(
    results: dict,
    sweep_keys: list,
    metric_keys: list,
    metric_names: list,
    lam_color: dict,
    progress: dict | None = None,
):
    """
    PCA on the bespoke metric vectors for all models, shown as a 3-D Plotly figure.
    Returns a go.Figure — caller calls .show() and .write_html().
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    all_keys   = ["baseline", "oracle"] + sweep_keys
    all_labels = ["Baseline", "Oracle"] + [f"lambda={l} {e}ep" for l, e in sweep_keys]
    X   = np.array([[results[k][m] for m in metric_keys] for k in all_keys])
    pca = PCA(n_components=3)
    X3  = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_
    print(f"PCA var: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  PC3={ev[2]:.1%}")

    bl_pt, or_pt, sw_pts = X3[0], X3[1], X3[2:]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=[bl_pt[0], or_pt[0]], y=[bl_pt[1], or_pt[1]], z=[bl_pt[2], or_pt[2]],
        mode="lines", line=dict(color="#4CAF50", width=6), name="->Oracle"))

    for (lam, ep), pt in zip(sweep_keys, sw_pts):
        fig.add_trace(go.Scatter3d(
            x=[bl_pt[0], pt[0]], y=[bl_pt[1], pt[1]], z=[bl_pt[2], pt[2]],
            mode="lines", line=dict(color=lam_color[lam], width=2),
            opacity=0.5, showlegend=False, hoverinfo="skip"))

    for (lam, ep), pt, lab in zip(sweep_keys, sw_pts, all_labels[2:]):
        mp    = (progress or {}).get((lam, ep), {}).get("mean", float("nan"))
        hover = (f"<b>{lab}</b><br>Mean progress: {mp:.3f}<br>" +
                 "<br>".join(f"{n}: {results[(lam,ep)][m]:.4f}"
                             for n, m in zip(metric_names, metric_keys)))
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
            marker=dict(size=6, color=lam_color[lam],
                        line=dict(color="black", width=1)),
            showlegend=False, hovertemplate=hover + "<extra></extra>"))

    fig.add_trace(go.Scatter3d(
        x=[bl_pt[0]], y=[bl_pt[1]], z=[bl_pt[2]], mode="markers+text",
        marker=dict(size=12, color="#2196F3", symbol="diamond"),
        text=["Baseline"], textposition="top center", name="Baseline"))
    fig.add_trace(go.Scatter3d(
        x=[or_pt[0]], y=[or_pt[1]], z=[or_pt[2]], mode="markers+text",
        marker=dict(size=12, color="#4CAF50", symbol="diamond"),
        text=["Oracle"], textposition="top center", name="Oracle"))

    loading_txt = "PC loadings:<br>" + "<br>".join(
        f"  {n}: PC1={pca.components_[0][i]:+.3f} "
        f"PC2={pca.components_[1][i]:+.3f} "
        f"PC3={pca.components_[2][i]:+.3f}"
        for i, n in enumerate(metric_names))

    fig.update_layout(
        title=dict(
            text=("Metric-space PCA — movement from baseline<br>"
                  "<sup>Green=oracle dir | colour=lambda | hover for metrics</sup>"),
            x=0.5),
        scene=dict(xaxis_title=f"PC1 ({ev[0]:.1%})",
                   yaxis_title=f"PC2 ({ev[1]:.1%})",
                   zaxis_title=f"PC3 ({ev[2]:.1%})"),
        annotations=[dict(text=loading_txt, align="left", showarrow=False,
                          xref="paper", yref="paper", x=0.01, y=0.01,
                          bordercolor="grey", borderwidth=1, bgcolor="lightyellow",
                          font=dict(size=9, family="monospace"))],
        width=900, height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(font=dict(size=9)))
    return fig


def collect_shape_batch(
    test_loader,
    forget_col: int,
    shape_names: dict,
    device,
    *,
    target_counts: dict | None = None,
    ordered: list | None = None,
) -> tuple:
    """
    Pull fixed counts of each shape from *test_loader* and stack into a batch.

    target_counts : {shape_val: n}  default {2:4, 0:2, 1:2}  (hearts, squares, ellipses)
    ordered       : column ordering of shape vals in the returned batch.
                    default [2,2,2,2,0,0,1,1]

    Returns (batch_tensor_on_device, col_labels_list).
    """
    if target_counts is None:
        target_counts = {2: 4, 0: 2, 1: 2}
    if ordered is None:
        ordered = [2, 2, 2, 2, 0, 0, 1, 1]

    collected = {s: [] for s in target_counts}
    for imgs, _, lbl_cls in test_loader:
        for s, n in target_counts.items():
            if sum(b.shape[0] for b in collected[s]) < n:
                mask = lbl_cls[:, forget_col] == s
                if mask.any():
                    collected[s].append(imgs[mask])
        if all(sum(b.shape[0] for b in collected[s]) >= n
               for s, n in target_counts.items()):
            break

    bufs = {s: torch.cat(collected[s]) for s in target_counts}
    ptrs = {s: 0 for s in target_counts}
    imgs_list = []
    for s in ordered:
        imgs_list.append(bufs[s][ptrs[s]])
        ptrs[s] += 1
    batch      = torch.stack(imgs_list).to(device)
    col_labels = [shape_names[s] for s in ordered]
    return batch, col_labels


def plot_sweep_reconstructions(
    batch,
    col_labels: list,
    baseline_model,
    oracle_model,
    sweep_models: dict,
    sweep_keys: list,
    device,
    *,
    decode_mode: str = "sigmoid",  # "sigmoid" (dSprites) | "tanh" (CelebA)
    out_path: Path | str | None = None,
) -> None:
    """
    Grid figure: rows = baseline / oracle / sweep runs; columns = input images.
    Row 0 shows originals; subsequent rows show reconstructions.

    decode_mode:
        "sigmoid" — decoder logits go through sigmoid (dSprites BCE decoder).
        "tanh"    — decoder output is in [-1, 1]; display as (x+1)/2 (CelebA MSE/Tanh decoder).
    """
    def _decode(model, x):
        # Return in the SAME range as the input `batch`, because `_to_display`
        # (below) applies the denorm. A tanh decoder outputs [-1,1] just like the
        # normalised input, so return it raw -- denorming here as well would apply
        # (x+1)/2 TWICE (once here, once in _to_display), lifting every dark pixel
        # to >=0.5 and washing every reconstruction toward white (the "white
        # sheen" bug, tanh/CelebA+MPI3D only; the sigmoid/dSprites path was fine).
        # A sigmoid decoder outputs logits, so map them to the [0,1] input range.
        with torch.no_grad():
            mu_enc, _ = model.encode(x)
            raw = model.decode(mu_enc)
        return raw if decode_mode == "tanh" else torch.sigmoid(raw)

    def _to_display(t):
        """Convert a CHW tensor to a displayable numpy array."""
        img = t.cpu()
        if decode_mode == "tanh":
            img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
        arr = img.permute(1, 2, 0).numpy() if img.shape[0] == 3 else img.squeeze(0).numpy()
        return arr

    cmap = None if (batch.shape[1] == 3) else "gray"

    all_rows = (
        [("Baseline", baseline_model), ("Oracle", oracle_model)] +
        [(f"λ={lam} {ep}ep", sweep_models[(lam, ep)]) for lam, ep in sweep_keys]
    )
    N = batch.shape[0]
    fig, axes = plt.subplots(len(all_rows) + 1, N,
                             figsize=(N * 1.5, (len(all_rows) + 1) * 1.4))
    fig.suptitle("Reconstructions — 4 forget | 2 retain-A | 2 retain-B", fontsize=10)
    fig.subplots_adjust(left=0.14)

    for col in range(N):
        show_ax(axes[0, col], _to_display(batch[col]),
                col_labels[col] if col in [0, 4, 6] else "", cmap=cmap)
    axes[0, 0].text(-0.35, 0.5, "Original", transform=axes[0, 0].transAxes,
                    ha="right", va="center", fontsize=8, fontweight="bold")

    for row, (label, model) in enumerate(all_rows, start=1):
        recon = _decode(model, batch)
        for col in range(N):
            show_ax(axes[row, col], _to_display(recon[col]), "", cmap=cmap)
        axes[row, 0].text(-0.35, 0.5, label, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=6)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_umap_panels(
    encodings: dict,
    show_keys: list,
    show_labels: list,
    shape_colors: dict,
    shape_names: dict,
    seed: int,
    *,
    n_neighbors: int = 7,
    min_dist: float = 0.1,
    out_path: Path | str | None = None,
) -> None:
    """
    Row of independent UMAP scatter panels, one per key model, coloured by shape.
    Fits 3 components and displays the first two.
    """
    from umap import UMAP

    def _fit(mu):
        return UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=seed, n_jobs=1).fit_transform(mu)[:, :2]

    n = len(show_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("UMAP — key models  (n_components=3, independently fitted)", fontsize=11)

    for ax, key, lab in zip(axes[0], show_keys, show_labels):
        mu, sv = encodings[key]
        print(f"Fitting UMAP: {lab} ...")
        emb = _fit(mu)
        for s in sorted(np.unique(sv)):
            mask = sv == s
            ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.4,
                       color=shape_colors[int(s)], label=shape_names[int(s)])
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pca_panels(
    encodings: dict,
    show_keys: list,
    show_labels: list,
    shape_colors: dict,
    shape_names: dict,
    *,
    out_path: Path | str | None = None,
) -> None:
    """
    Row of PCA scatter panels, one per key model, coloured by class — the linear
    counterpart to plot_umap_panels. Each model's mu is projected onto ITS OWN
    top-2 principal components (fitted independently, like the UMAP panels), so a
    concept that is *linearly* separable shows up as separation here even when the
    nonlinear UMAP would fold it away, and vice versa. The fraction of latent
    variance the two axes capture is printed in each panel title.
    """
    from sklearn.decomposition import PCA

    n = len(show_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("PCA — key models  (top-2 principal components, independently fitted)",
                 fontsize=11)

    for ax, key, lab in zip(axes[0], show_keys, show_labels):
        mu, sv = encodings[key]
        pca = PCA(n_components=2)
        emb = pca.fit_transform(mu)
        var = pca.explained_variance_ratio_
        for s in sorted(np.unique(sv)):
            mask = sv == s
            ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.4,
                       color=shape_colors[int(s)], label=shape_names[int(s)])
        ax.set_title(f"{lab}\n(PC1+PC2 = {var.sum():.0%} of variance)", fontsize=10)
        ax.set_xlabel(f"PC 1 ({var[0]:.0%})")
        ax.set_ylabel(f"PC 2 ({var[1]:.0%})")
        ax.legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


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


@torch.no_grad()
def compute_feature_direction_np(model, forget_loader, retain_loader, device) -> np.ndarray:
    """
    Compute z_e = mean_mu(forget) − mean_mu(retain) using *model*'s encoder.
    Returns a numpy array of shape (latent_dim,).
    """
    model.eval()

    def _mean_mu(loader):
        acc, n = None, 0
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            mu, _ = model.encode(x)
            mu = mu.cpu().numpy()
            acc = mu.sum(0) if acc is None else acc + mu.sum(0)
            n += x.size(0)
        return acc / n

    return _mean_mu(forget_loader) - _mean_mu(retain_loader)


def plot_translation_umap(
    mu: np.ndarray,
    sv: np.ndarray,
    z_e: np.ndarray,
    forget_val: int,
    shape_names: dict,
    shape_colors: dict,
    seed: int,
    *,
    n_neighbors: int = 7,
    min_dist: float = 0.1,
    n_arrows: int = 150,
    title: str = "Feature-direction translation",
    out_path=None,
) -> None:
    """
    Single UMAP panel showing the feature-direction translation effect.

    Retain points    — full colour, normal alpha.
    Forget points    — pale dot at original position, full-colour dot at
                       translated position, thin arrow connecting each pair
                       (subsampled to n_arrows for readability).

    The UMAP reducer is fitted on original latents only; translated positions
    are embedded with .transform() so arrows are meaningful in the same space.
    """
    from umap import UMAP
    import matplotlib.colors as mcolors

    def _lighten(hex_color: str, amount: float = 0.55) -> str:
        """Blend hex_color toward white by *amount* (0=original, 1=white)."""
        rgb = np.array(mcolors.to_rgb(hex_color))
        rgb_light = rgb + (1.0 - rgb) * amount
        return mcolors.to_hex(rgb_light)

    forget_mask = sv == forget_val
    retain_mask = ~forget_mask

    # ── Fit UMAP on all original latents ────────────────────────────────────
    print("Fitting UMAP for translation visualisation ...")
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                   random_state=seed, n_jobs=1)
    emb_orig = reducer.fit_transform(mu)                    # (N, 2)

    # ── Translate forget latents and embed in same space ────────────────────
    mu_translated = translate_latents_np(mu[forget_mask], z_e)
    emb_trans = reducer.transform(mu_translated)            # (n_forget, 2)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title, fontsize=11)

    # Retain points
    for val in sorted(shape_names):
        if val == forget_val:
            continue
        mask = sv == val
        if mask.any():
            ax.scatter(emb_orig[mask, 0], emb_orig[mask, 1],
                       s=2, alpha=0.4, color=shape_colors[val],
                       label=shape_names[val])

    # Forget: pale original positions
    fgt_color      = shape_colors[forget_val]
    fgt_color_pale = _lighten(fgt_color)
    ax.scatter(emb_orig[forget_mask, 0], emb_orig[forget_mask, 1],
               s=3, alpha=0.35, color=fgt_color_pale,
               label=f"{shape_names[forget_val]} (before)")

    # Forget: translated positions (full colour)
    ax.scatter(emb_trans[:, 0], emb_trans[:, 1],
               s=3, alpha=0.6, color=fgt_color,
               label=f"{shape_names[forget_val]} (after)")

    # Arrows (random subset)
    n_fgt = int(forget_mask.sum())
    if n_fgt > 0 and n_arrows > 0:
        rng   = np.random.default_rng(seed)
        idxs  = rng.choice(n_fgt, size=min(n_arrows, n_fgt), replace=False)
        ox    = emb_orig[forget_mask][idxs, 0]
        oy    = emb_orig[forget_mask][idxs, 1]
        tx    = emb_trans[idxs, 0]
        ty    = emb_trans[idxs, 1]
        for xi, yi, dxi, dyi in zip(ox, oy, tx - ox, ty - oy):
            ax.annotate("", xy=(xi + dxi, yi + dyi), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=0.4, alpha=0.35))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5, fontsize=9, loc="best")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


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


def discover_rmu_keys(sweep_dir: Path, *, u_seed: int = 0, c: float = 1.0) -> list:
    """Completed (alpha, ep) pairs for ONE steering-vector seed.

    Filtering by seed matters: the same (alpha, ep) exists at several seeds, and
    mixing them silently averages over a variable the sweep is meant to hold
    fixed.
    """
    import re
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        return []
    pat = re.compile(r"a([0-9.]+)_c([0-9.]+)_ep([0-9]+)(?:_s([0-9]+))?")
    keys = []
    for folder in sorted(sweep_dir.iterdir()):
        if not folder.is_dir() or not (folder / "final.pt").exists():
            continue
        m = pat.fullmatch(folder.name)
        if not m:
            continue
        a = float(m.group(1))
        cc = float(m.group(2))
        ep = int(m.group(3))
        s = int(m.group(4) or 0)
        if s == u_seed and abs(cc - c) < 1e-9:
            keys.append((a, ep))
    return sorted(keys)


def rmu_seed_variance(sweep_dir: Path, alphas, ep, seeds, *, budget: float,
                      c: float = 1.0) -> list:
    """Per-alpha spread of final val / forget across steering-vector seeds.

    This is the analysis that overturned the single-seed dSprites "optimum":
    alpha=100/ep5 sat inside the retain budget on some random directions and
    outside it on others, so a one-draw point estimate was not a real optimum.
    `in_budget` is the count that matters -- a config that only sometimes fits
    is not a usable operating point.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    for a in alphas:
        vals, forgets, missing = [], [], 0
        for s in seeds:
            hp = sweep_dir / rmu_run_name(a, ep, c=c, u_seed=s) / "history.json"
            if not hp.exists():
                missing += 1
                continue
            h = json.loads(hp.read_text())
            vals.append(h["val_loss"][-1])
            forgets.append(h["forget_loss"][-1])
        if not vals:
            continue
        rows.append({
            "alpha":       a,
            "n":           len(vals),
            "missing":     missing,
            "val_mean":    float(np.mean(vals)),
            "val_sd":      float(np.std(vals)),
            "forget_mean": float(np.mean(forgets)),
            "forget_sd":   float(np.std(forgets)),
            "in_budget":   int(sum(1 for v in vals if v <= budget)),
            "vals":        vals,
            "forgets":     forgets,
        })
    return rows


def print_seed_variance(rows: list, budget: float) -> None:
    print(f"retain val budget = {budget:.2f}")
    print(f"{'alpha':>7} {'n':>3} {'val mean':>9} {'val sd':>7} "
          f"{'in budget':>10} {'forget mean':>12} {'forget sd':>10}")
    print("-" * 66)
    for r in rows:
        if r["in_budget"] == r["n"]:
            flag = ""
        elif r["in_budget"]:
            flag = "  <-- UNRELIABLE"
        else:
            flag = "  <-- never fits"
        print(f"{r['alpha']:>7.0f} {r['n']:>3d} {r['val_mean']:>9.2f} {r['val_sd']:>7.3f} "
              f"{r['in_budget']:>6d}/{r['n']:<3d} {r['forget_mean']:>12.3f} "
              f"{r['forget_sd']:>10.3f}{flag}")


def plot_seed_variance(rows: list, budget: float, *, title: str = "",
                       out_path: Path | None = None):
    """Val and forget vs alpha, with per-seed scatter and the budget line.

    Individual draws are drawn as well as the mean, so a config that straddles
    the budget is visibly straddling it rather than hidden inside an error bar.
    """
    if not rows:
        print("  (no runs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = [r["alpha"] for r in rows]

    ax = axes[0]
    for r in rows:
        ax.scatter([r["alpha"]] * len(r["vals"]), r["vals"], s=16,
                   color="#888888", alpha=0.6, zorder=2)
    ax.errorbar(xs, [r["val_mean"] for r in rows], yerr=[r["val_sd"] for r in rows],
                marker="o", color="#2196F3", capsize=4, zorder=3, label="mean +/- sd")
    ax.axhline(budget, color="#F44336", ls="--", lw=1.5, label=f"budget {budget:.2f}")
    ax.set_xlabel("alpha (retain anchor)")
    ax.set_ylabel("retain val loss")
    ax.set_title("Retain val vs alpha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        ax.scatter([r["alpha"]] * len(r["forgets"]), r["forgets"], s=16,
                   color="#888888", alpha=0.6, zorder=2)
    ax.errorbar(xs, [r["forget_mean"] for r in rows], yerr=[r["forget_sd"] for r in rows],
                marker="o", color="#4CAF50", capsize=4, zorder=3)
    ax.set_xlabel("alpha (retain anchor)")
    ax.set_ylabel("forget loss (lower = more forgetting)")
    ax.set_title("Forgetting vs alpha")
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


def plot_rmu_trajectories(sweep_dir: Path, runs: list, budget: float, *,
                          labels: list | None = None, title: str = "",
                          out_path: Path | None = None):
    """Per-epoch val and forget trajectories for named runs.

    RMU's whole story is in the trajectory shape, not the endpoint: dSprites
    rises monotonically past budget within a few epochs, while CelebA peaks
    then self-corrects back under budget while still forgetting.
    """
    sweep_dir = Path(sweep_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap("plasma")
    for i, run in enumerate(runs):
        hp = sweep_dir / run / "history.json"
        if not hp.exists():
            print(f"  missing: {run}")
            continue
        h = json.loads(hp.read_text())
        color = cmap(i / max(len(runs) - 1, 1) * 0.85)
        lab = labels[i] if labels else run
        eps = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(eps, h["val_loss"], marker="", lw=1.6, color=color, label=lab)
        axes[1].plot(eps, h["forget_loss"], marker="", lw=1.6, color=color, label=lab)
    axes[0].axhline(budget, color="#F44336", ls="--", lw=1.5, label=f"budget {budget:.2f}")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("retain val loss")
    axes[0].set_title("Retain val trajectory"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("forget loss (lower = more forgetting)")
    axes[1].set_title("Forget trajectory"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


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


def discover_fisher_keys(sweep_dir: Path, *, fisher_seed: int = 0) -> list:
    """Completed noise_scale values for ONE fisher seed, sorted ascending.

    Filtering by seed matters for the same reason it does for RMU: the same
    noise_scale exists at several seeds, and mixing them would silently average
    over the one variable a seed-variance study needs to hold fixed.
    """
    import re
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        return []
    pat = re.compile(r"sig([0-9.]+)_ep1(?:_fs([0-9]+))?(?:_s([0-9]+))?")
    scales = []
    for folder in sorted(sweep_dir.iterdir()):
        if not folder.is_dir() or not (folder / "final.pt").exists():
            continue
        m = pat.fullmatch(folder.name)
        if not m:
            continue
        scale = float(m.group(1))
        s = int(m.group(3) or 0)
        if s == fisher_seed:
            scales.append(scale)
    return sorted(scales)


def fisher_seed_variance(sweep_dir: Path, noise_scales, seeds, *, budget: float) -> list:
    """Per-noise_scale spread of retain val / forget ELBO across seeds.

    The count that matters is `in_budget`: a noise_scale that only stays under
    the retain budget on some draws of the perturbation is not a usable
    operating point, however good its mean looks. Measured on dSprites at
    noise_scale=0.01: 3 of 5 seeds stayed in budget (val 39.64/38.99 in,
    43.10/42.56 out) -- a single-seed run at that scale could have reported
    either "usable" or "destroyed" depending on the draw.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    for scale in noise_scales:
        vals, forgets, missing = [], [], 0
        for s in seeds:
            hp = sweep_dir / fisher_run_name(scale, fisher_seed=s) / "history.json"
            if not hp.exists():
                missing += 1
                continue
            h = json.loads(hp.read_text())
            vals.append(h["val_loss"][-1])
            forgets.append(h["forget_loss"][-1])
        if not vals:
            continue
        rows.append({
            "noise_scale": scale,
            "n":           len(vals),
            "missing":     missing,
            "val_mean":    float(np.mean(vals)),
            "val_sd":      float(np.std(vals)),
            "forget_mean": float(np.mean(forgets)),
            "forget_sd":   float(np.std(forgets)),
            "in_budget":   int(sum(1 for v in vals if v <= budget)),
            "vals":        vals,
            "forgets":     forgets,
        })
    return rows


def print_fisher_seed_variance(rows: list, budget: float) -> None:
    print(f"retain val budget = {budget:.2f}")
    print(f"{'noise_scale':>11} {'n':>3} {'val mean':>9} {'val sd':>7} "
          f"{'in budget':>10} {'forget mean':>12} {'forget sd':>10}")
    print("-" * 70)
    for r in rows:
        if r["in_budget"] == r["n"]:
            flag = ""
        elif r["in_budget"]:
            flag = "  <-- UNRELIABLE"
        else:
            flag = "  <-- never fits"
        print(f"{r['noise_scale']:>11.4f} {r['n']:>3d} {r['val_mean']:>9.2f} "
              f"{r['val_sd']:>7.3f} {r['in_budget']:>6d}/{r['n']:<3d} "
              f"{r['forget_mean']:>12.3f} {r['forget_sd']:>10.3f}{flag}")


def plot_fisher_seed_variance(rows: list, budget: float, *, title: str = "",
                              out_path: Path | None = None):
    """Val and forget vs noise_scale, with per-seed scatter and the budget line.

    Individual draws are plotted alongside the mean, same rationale as
    plot_seed_variance: a config straddling the budget should look like it is
    straddling it, not be hidden inside an error bar.
    """
    if not rows:
        print("  (no runs)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = [r["noise_scale"] for r in rows]

    ax = axes[0]
    for r in rows:
        ax.scatter([r["noise_scale"]] * len(r["vals"]), r["vals"], s=16,
                   color="#888888", alpha=0.6, zorder=2)
    ax.errorbar(xs, [r["val_mean"] for r in rows], yerr=[r["val_sd"] for r in rows],
                marker="o", color="#2196F3", capsize=4, zorder=3, label="mean +/- sd")
    ax.axhline(budget, color="#F44336", ls="--", lw=1.5, label=f"budget {budget:.2f}")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("retain val loss")
    ax.set_title("Retain val vs noise_scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        ax.scatter([r["noise_scale"]] * len(r["forgets"]), r["forgets"], s=16,
                   color="#888888", alpha=0.6, zorder=2)
    ax.errorbar(xs, [r["forget_mean"] for r in rows], yerr=[r["forget_sd"] for r in rows],
                marker="o", color="#4CAF50", capsize=4, zorder=3)
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("forget ELBO (higher = more forgetting)")
    ax.set_title("Forgetting vs noise_scale")
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


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


def print_method_champions(champions: dict, budget: float,
                           all_methods: list | None = None) -> None:
    print(f"retain val budget = {budget:.2f}\n")
    print(f"{'method':<18} {'run':<26} {'ep':>4} {'val':>9} {'forget':>11}  checkpoint")
    print("-" * 94)
    for name, c in champions.items():
        note = "" if c["ckpt_is_exact"] else "  (final.pt -- exact epoch not saved)"
        print(f"{name:<18} {c['name']:<26} {c['epoch']:>4d} {c['val']:>9.2f} "
              f"{c['forget']:>11.3f}  {c['ckpt'].name}{note}")

    if all_methods:
        missing = [m for m in all_methods if m not in champions]
        if missing:
            print(f"\n  NO run inside budget for: {', '.join(missing)}")
            print( "  (every configuration of these damages retain past tolerance)")

    inexact = [n for n, c in champions.items() if not c["ckpt_is_exact"]]
    if inexact:
        print(f"\n  NOTE: {len(inexact)} method(s) analysed at final.pt because the "
              f"trade-off epoch's weights were never saved (--save-every 0):")
        print(f"        {', '.join(inexact)}")
        print( "        Rerun those with --save-every 1 for an exact comparison.")


def plot_method_metric_bars(results: dict, order: list, metric_keys: list,
                            metric_names: list, *, out_path: Path | None = None):
    """Grouped bars: one panel per metric, one bar per model.

    Absolute metric values, not progress -- progress hides that some metrics
    move the wrong way for the oracle (see compute_progress's docstring).
    """
    present = [k for k in order if k in results]
    if not present:
        print("  (nothing to plot)")
        return
    n_m = len(metric_keys)
    fig, axes = plt.subplots(1, n_m, figsize=(3.1 * n_m, 3.9))
    if n_m == 1:
        axes = [axes]
    fixed = {"baseline": "#9E9E9E", "oracle": "#212121"}
    palette = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    bar_colors, ci = {}, 0
    for k in present:
        if k in fixed:
            bar_colors[k] = fixed[k]
        else:
            bar_colors[k] = palette[ci % len(palette)]
            ci += 1

    for ax, mk, mn in zip(axes, metric_keys, metric_names):
        vals = [results[k][mk] for k in present]
        ax.bar(range(len(present)), vals, color=[bar_colors[k] for k in present])
        if "baseline" in results:
            ax.axhline(results["baseline"][mk], color="#9E9E9E", ls=":", lw=1)
        if "oracle" in results:
            ax.axhline(results["oracle"][mk], color="#212121", ls="--", lw=1)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        ax.set_title(mn, fontsize=10)
        ax.set_xlabel("model", fontsize=8)
        ax.set_ylabel(mn, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


def plot_cka_matrix(cka: dict, *, title: str = "Latent-space CKA",
                    out_path: Path | None = None):
    """Heatmap of src.evaluation.cka.cka_matrix output.

    CKA=1 means two representations are identical up to rotation and scale.
    Read the baseline row: a method whose CKA to baseline is ~1 barely moved
    the latent geometry at all, whatever its forget loss reports.
    """
    M, labels = cka["matrix"], cka["labels"]
    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 3, 1.05 * len(labels) + 2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < 0.6 else "black")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


def plot_method_umap_panels(encodings: dict, order: list, *, forget_mask_fn,
                            title: str = "", out_path: Path | None = None,
                            sample_n: int = 6000, seed: int = 0):
    """One UMAP panel per model, forget class highlighted, shared layout.

    Each panel is fitted independently -- UMAP coordinates are not comparable
    across panels, only the *shape* of the forget cluster relative to retain is.
    CKA is the quantitative counterpart; this is the qualitative one.
    """
    from umap import UMAP
    present = [k for k in order if k in encodings]
    if not present:
        print("  (nothing to plot)")
        return
    n = len(present)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    rng = np.random.default_rng(seed)
    for i, key in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        mu, sv = encodings[key]
        if len(mu) > sample_n:
            idx = rng.choice(len(mu), sample_n, replace=False)
            mu_s, sv_s = mu[idx], sv[idx]
        else:
            mu_s, sv_s = mu, sv
        emb = UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(mu_s)
        m = forget_mask_fn(sv_s)
        ax.scatter(emb[~m, 0], emb[~m, 1], s=3, c="#2196F3", alpha=0.35, label="retain")
        ax.scatter(emb[m, 0], emb[m, 1], s=3, c="#F44336", alpha=0.55, label="forget")
        ax.set_title(str(key), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, markerscale=3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


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

def inlp_run_name(n_iter, *, inlp_seed: int = 0, fit_samples: int = 0) -> str:
    """Directory name for one INLP run -- mirrors methods.INLPErase.run_name()."""
    name = f"k{int(n_iter)}_ep1"
    if fit_samples:
        name += f"_fs{int(fit_samples)}"
    if inlp_seed:
        name += f"_s{int(inlp_seed)}"
    return name


def discover_inlp_keys(sweep_dir: Path, *, inlp_seed: int = 0) -> list:
    """Completed n_iter values for ONE inlp seed, sorted ascending."""
    import re
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.exists():
        return []
    pat = re.compile(r"k([0-9]+)_ep1(?:_fs([0-9]+))?(?:_s([0-9]+))?")
    ks = []
    for folder in sorted(sweep_dir.iterdir()):
        if not folder.is_dir() or not (folder / "final.pt").exists():
            continue
        m = pat.fullmatch(folder.name)
        if not m:
            continue
        if int(m.group(3) or 0) == inlp_seed:
            ks.append(int(m.group(1)))
    return sorted(ks)


def erasure_run_rows(sweep_dir: Path, keys, name_fn, *, budget: float) -> list:
    """Per-run retain val / forget ELBO, read straight from history.json.

    One-shot methods have a single entry per history, so [-1] is the only entry
    rather than a choice of epoch -- there is no trade-off-epoch ambiguity to
    resolve, which is why this and the Fisher notebook are structurally simpler
    than RMU's.
    """
    sweep_dir = Path(sweep_dir)
    rows = []
    for k in keys:
        hp = sweep_dir / name_fn(k) / "history.json"
        if not hp.exists():
            continue
        h = json.loads(hp.read_text())
        val, forget = h["val_loss"][-1], h["forget_loss"][-1]
        rows.append({"key": k, "val": val, "forget": forget,
                     "in_budget": bool(val <= budget)})
    return rows


def print_erasure_rows(rows: list, budget: float, *, axis_label: str = "k") -> None:
    print(f"retain val budget = {budget:.2f}")
    print(f"{axis_label:>8} {'retain val':>12} {'forget ELBO':>13}  status")
    print("-" * 50)
    for r in rows:
        flag = "" if r["in_budget"] else "  <-- OVER BUDGET"
        print(f"{r['key']:>8} {r['val']:>12.2f} {r['forget']:>13.3f}{flag}")


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


def print_recovery_table(curves: dict, *, thresholds=(0.7, 0.8, 0.9)) -> None:
    """AUROC at the largest n, plus attacker-samples-to-threshold, per model."""
    probes = sorted({p for c in curves.values() for p in c["curves"]})
    head = f"{'model':<22}"
    for p in probes:
        head += f" {p + ' AUROC@max':>17}"
        for t in thresholds:
            head += f" {'n@' + str(t):>8}"
    print(head)
    print("-" * len(head))
    for label, c in curves.items():
        row = f"{label:<22}"
        for p in probes:
            if p not in c["curves"]:
                row += f" {'--':>17}" + "".join(f" {'--':>8}" for _ in thresholds)
                continue
            row += f" {c['curves'][p]['auroc_mean'][-1]:>17.4f}"
            for t in thresholds:
                v = c["n_to_recover"][p].get(str(t))
                row += f" {('never' if v is None else v):>8}"
        print(row)


def plot_recovery_curves(curves: dict, *, title: str = "", out_path=None,
                         probes=("linear", "mlp")):
    """AUROC vs attacker labelled-set size, one panel per probe.

    Log x axis on purpose: the quantity of interest is the ORDER OF MAGNITUDE
    of labels an attacker needs. A method that pushes recovery from 32 to 512
    samples has done something real, and on a linear axis that shift is
    invisible next to the 2048 end of the grid.
    """
    present = [p for p in probes if any(p in c["curves"] for c in curves.values())]
    if not present:
        print("no curves to plot")
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.6),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    labels = list(curves)
    for ax, probe in zip(axes[0], present):
        n = None
        for i, label in enumerate(labels):
            c = curves[label]
            if probe not in c["curves"]:
                continue
            n = c["n_grid"]
            m = np.asarray(c["curves"][probe]["auroc_mean"])
            sd = np.asarray(c["curves"][probe]["auroc_sd"])
            low = label.lower()
            colour = ("#000000" if low == "baseline" else
                      "#F44336" if low == "oracle" else
                      cmap(i / max(len(labels) - 1, 1)))
            style = "--" if low in ("baseline", "oracle") else "-"
            ax.plot(n, m, style, marker="o", ms=4, color=colour, label=label)
            ax.fill_between(n, m - sd, m + sd, color=colour, alpha=0.12)
        ax.axhline(0.5, color="grey", lw=1, ls=":")
        if n:
            ax.text(n[0], 0.507, "chance", fontsize=8, color="grey")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("attacker labelled samples (n)")
        ax.set_ylabel("recovered AUROC")
        ax.set_title(f"{probe} probe")
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"  saved -> {out_path}")
    plt.show()


def latent_rank(mu, *, rtol: float = 1e-3) -> int:
    """Effective rank of the centered latent cloud.

    The honest label for an erasure run: INLP was ASKED for k rounds, but it
    stops adding directions once the concept classifier hits chance, so
    requested-k and delivered-rank diverge. Measured on dSprites, k=4,5,6 all
    erase rank 4 and produce bit-identical checkpoints (val 4178.3788 for all
    three) -- plotting against requested k would draw three distinct points on
    top of one model.

    Reading the rank straight off the encodings also means this reports what
    the ARTEFACT does rather than what the config asked for, which is the same
    discipline champion_run's require_exact enforces for checkpoints.
    """
    mu = np.asarray(mu, dtype=np.float64)
    s = np.linalg.svd(mu - mu.mean(0), compute_uv=False)
    return int((s > s.max() * rtol).sum()) if s.size and s.max() > 0 else 0


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

def plot_reconstruction_pairs(
    model,
    images: torch.Tensor,
    device,
    *,
    mode: str = "tanh",
    n: int = 8,
    title: str = "Reconstruction sanity check",
    cmap: str | None = None,
    out_path: Path | str | None = None,
) -> float:
    """Originals (top) against their reconstructions (bottom). Returns MSE.

    The cheapest possible check that a checkpoint is what it claims to be. A
    model loaded from the wrong epoch, or one whose retain loss has blown past
    budget, produces reconstructions that are visibly wrong even when its
    latent metrics still look ordinary -- which is precisely how the
    `celeba_finetune` epoch-20 artefact (val 8022, direction norm 50.0 against
    the baseline's 2.1) managed to read as a spectacular unlearning result.

    `mode` follows decode_to_img: "tanh" for CelebA (decoder in [-1,1]),
    "sigmoid" for dSprites (decoder emits BCE logits).
    """
    model.eval()
    x = images[:n].to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
        x_hat = model.decode(mu)
    x_hat = (x_hat + 1.0) / 2.0 if mode == "tanh" else torch.sigmoid(x_hat)
    x_show = (x + 1.0) / 2.0 if mode == "tanh" else x
    x_hat, x_show = x_hat.clamp(0, 1), x_show.clamp(0, 1)
    mse = float(((x_hat - x_show) ** 2).mean().item())

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.4))
    for i in range(n):
        show_ax(axes[0, i], to_np(x_show[i:i + 1]), "original" if i == 0 else "", cmap=cmap)
        show_ax(axes[1, i], to_np(x_hat[i:i + 1]), "reconstruction" if i == 0 else "", cmap=cmap)
    fig.suptitle(f"{title}   (pixel MSE = {mse:.4f})", fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return mse


def plot_latent_traversal(
    model,
    z_base: torch.Tensor,
    device,
    *,
    mode: str = "tanh",
    dims: list[int] | None = None,
    span: float = 3.0,
    n_steps: int = 7,
    cmap: str | None = None,
    title: str = "Latent traversal",
    out_path: Path | str | None = None,
) -> None:
    """One row per latent dimension, swept from -span to +span in n_steps.

    The standard VAE diagnostic: walk one coordinate while holding the rest
    fixed and watch what changes in pixel space. Rows that do not change are
    collapsed dimensions carrying no information -- the visual counterpart of
    `plot_latent_variance`'s near-zero bars, and worth cross-checking against
    it, because a dimension can have healthy variance across the dataset while
    still being ignored by the decoder.

    Read this alongside the concept traversal below: this asks "what does the
    latent space encode at all", that one asks "is the forget concept still in
    there". `dims` defaults to every dimension, which is right at d=8 and
    unreadable at d=256 -- pass the dimensions of interest there (e.g. the
    highest-variance ones, or those with the largest concept-direction
    loading).
    """
    model.eval()
    z_base = z_base.to(device).reshape(1, -1)
    dims = list(range(z_base.shape[1])) if dims is None else list(dims)
    offsets = np.linspace(-span, span, n_steps)

    fig, axes = plt.subplots(len(dims), n_steps,
                             figsize=(1.3 * n_steps, 1.35 * len(dims)),
                             squeeze=False)
    for r, d in enumerate(dims):
        for c, off in enumerate(offsets):
            z = z_base.clone()
            z[0, d] = z_base[0, d] + float(off)
            show_ax(axes[r][c], decode_to_img(model, z, mode=mode),
                    f"{off:+.1f}" if r == 0 else "", cmap=cmap)
        axes[r][0].set_ylabel(f"z{d}", rotation=0, labelpad=16,
                              fontsize=9, va="center")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_concept_traversal(
    model,
    z_start: torch.Tensor,
    z_e: np.ndarray,
    device,
    *,
    mode: str = "tanh",
    radii=None,
    cmap: str | None = None,
    title: str = "Traversal along the concept direction",
    out_path: Path | str | None = None,
) -> None:
    """Walk a latent along z_e, stepping in L2 DISTANCE (not multiples of z_e).

    This is the qualitative counterpart of the whole thesis. `z_e` is the
    direction that separates forget from retain in latent space, so stepping
    along it asks the question the attack metric asks numerically: **starting
    from a retain point, can the concept be reached by moving in latent
    space?** If the concept appears as the distance grows, it is still
    decodable and the model has not removed it -- however small its `dir_norm`
    has become.

    That makes this the plot that explains the project's central result to a
    reader who does not want to read a table. LEACE drives dir_norm to ~0.0085
    (CelebA) and Linear AUROC below chance, yet is the most attackable model in
    the study; if this traversal still produces the concept for LEACE while
    staying clean for feature_unlearn, the reader has seen the point directly.

    WHY DISTANCE AND NOT ALPHA
    --------------------------
    The first version of this stepped `z_start + alpha * z_e` over
    alpha in [0, 2], and produced a row of visually IDENTICAL images. The
    reason is a scale mismatch, and it matters: `z_e` is a difference of class
    means, so on the dSprites baseline its norm is only **0.3625** -- alpha=2
    therefore travels 0.725 in latent space, while `plot_latent_traversal`
    needs +-3 per coordinate before anything visibly changes. The plot was not
    wrong, it was measuring in the wrong unit and showed nothing.

    Parameterising by L2 distance instead fixes that AND makes the columns
    directly comparable to `attack@r`, which is also an L2 radius -- so the
    r=2.0 column here is the same budget the dSprites attack number reports,
    and the r=4.0 column matches CelebA's. Defaults follow the attack's own
    radius grid.

    Note this means `dir_norm` no longer sets the scale of the walk, only its
    direction -- which is the point: a method can shrink `dir_norm` to nearly
    zero (LEACE) without making the concept any harder to reach.
    """
    model.eval()
    if radii is None:
        radii = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    radii = np.asarray(list(radii), dtype=np.float64)
    z_start = z_start.to(device).reshape(1, -1)

    z_np = np.asarray(z_e, dtype=np.float32)
    norm = float(np.linalg.norm(z_np))
    if norm <= 0:
        raise ValueError("z_e has zero norm -- no concept direction to traverse")
    unit = torch.as_tensor(z_np / norm, device=device).reshape(1, -1)

    fig, axes = plt.subplots(1, len(radii), figsize=(1.4 * len(radii), 2.0),
                             squeeze=False)
    for c, r in enumerate(radii):
        show_ax(axes[0][c], decode_to_img(model, z_start + float(r) * unit, mode=mode),
                f"r={r:g}", cmap=cmap)
    fig.suptitle(f"{title}   (‖z_e‖ = {norm:.4f}; steps are L2 distance, "
                 f"matching attack@r)", fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_method_concept_traversals(
    models: dict,
    z_starts: dict,
    z_es: dict,
    device,
    *,
    mode: str = "tanh",
    radii=None,
    cmap: str | None = None,
    title: str = "Concept direction traversal, by method",
    out_path: Path | str | None = None,
) -> None:
    """One row per method, each traversing its OWN concept direction.

    Each method gets its own z_e and its own starting point deliberately: the
    question is not "what happens along the baseline's direction" (that
    direction may not mean anything in another model's latent space) but "along
    whatever direction separates forget from retain in THIS model, is the
    concept reachable". Sharing one direction across models would compare
    incomparable coordinate systems.

    Steps are L2 DISTANCE along the unit direction, for the reason spelled out
    in `plot_concept_traversal`: `dir_norm` varies by more than two orders of
    magnitude across these methods (CelebA baseline 2.10 vs LEACE 0.0085), so
    stepping in multiples of z_e would give every row a different scale and the
    comparison would be meaningless -- LEACE's row would barely move, which
    would look like success and is exactly backwards. Fixing the distance makes
    every row the same question, and matches `attack@r`'s units.

    Expects the three dicts to be keyed identically; rows are drawn in the key
    order of `models`, so pass an ordered dict (e.g. sorted by attack
    resistance) to make the visual ordering match the table's.
    """
    if radii is None:
        radii = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    radii = np.asarray(list(radii), dtype=np.float64)
    names = [k for k in models if k in z_starts and k in z_es]

    fig, axes = plt.subplots(len(names), len(radii),
                             figsize=(1.4 * len(radii), 1.5 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        m = models[name]
        m.eval()
        z0 = z_starts[name].to(device).reshape(1, -1)
        z_np = np.asarray(z_es[name], dtype=np.float32)
        nrm = float(np.linalg.norm(z_np))
        if nrm <= 0:
            continue
        zd = torch.as_tensor(z_np / nrm, device=device).reshape(1, -1)
        for c, rad in enumerate(radii):
            show_ax(axes[r][c], decode_to_img(m, z0 + float(rad) * zd, mode=mode),
                    f"r={rad:g}" if r == 0 else "", cmap=cmap)
        axes[r][0].set_ylabel(str(name), rotation=0, labelpad=42, fontsize=9, va="center")
        axes[r][0].axis("on")
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_latent_interpolation(
    model,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    device,
    *,
    mode: str = "tanh",
    n_steps: int = 9,
    cmap: str | None = None,
    title: str = "Latent interpolation",
    out_path: Path | str | None = None,
) -> None:
    """Decode the straight line between two real images' encodings.

    A smoothness check on the latent space: a well-behaved VAE produces a
    gradual morph, while a damaged one jumps or passes through mush. Useful
    after an unlearning intervention, where the question "did this break the
    representation generally, or only for the forget concept" is exactly what
    the retain-budget number is trying to answer numerically.

    Pass a retain image and a forget image to see where along the path the
    concept appears -- a qualitative read on how sharply the two regions are
    separated.
    """
    model.eval()
    with torch.no_grad():
        mu_a, _ = model.encode(x_a.to(device).reshape(1, *x_a.shape[-3:]))
        mu_b, _ = model.encode(x_b.to(device).reshape(1, *x_b.shape[-3:]))
    ts = np.linspace(0.0, 1.0, n_steps)

    fig, axes = plt.subplots(1, n_steps, figsize=(1.4 * n_steps, 2.0), squeeze=False)
    for c, t in enumerate(ts):
        z = (1.0 - float(t)) * mu_a + float(t) * mu_b
        show_ax(axes[0][c], decode_to_img(model, z, mode=mode), f"t={t:.2f}", cmap=cmap)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_reconstruction_grid(
    models: dict,
    batch: torch.Tensor,
    col_labels: list,
    device,
    *,
    mode: str = "tanh",
    title: str = "Reconstructions by model",
    out_path: Path | str | None = None,
) -> dict:
    """Row 0 = inputs, one further row per model. Returns {name: pixel MSE}.

    Distinct from `plot_sweep_reconstructions`, which keys its rows by
    ``(lam, epochs)`` hyperparameter tuples because it serves the sweep
    notebooks. Here rows are keyed by arbitrary labels -- method names -- which
    is what the cross-method and champion comparisons need. Reusing the sweep
    version for that fails with "too many values to unpack", since it tries to
    destructure each key as a 2-tuple.

    Returns per-model reconstruction MSE against the same inputs, so the figure
    comes with the number that summarises it -- useful because "does this look
    worse" is exactly the judgement a reader should not have to make by eye
    alone.
    """
    names = list(models)
    x = batch.to(device)
    x_show = ((x + 1.0) / 2.0 if mode == "tanh" else x).clamp(0, 1)
    cmap = None if x.shape[1] == 3 else "gray"
    n = x.shape[0]

    recons, mses = {}, {}
    for name in names:
        m = models[name]
        m.eval()
        with torch.no_grad():
            mu, _ = m.encode(x)
            raw = m.decode(mu)
        r = ((raw + 1.0) / 2.0 if mode == "tanh" else torch.sigmoid(raw)).clamp(0, 1)
        recons[name] = r
        mses[name] = float(((r - x_show) ** 2).mean().item())

    fig, axes = plt.subplots(len(names) + 1, n,
                             figsize=(n * 1.5, (len(names) + 1) * 1.5),
                             squeeze=False)
    for c in range(n):
        show_ax(axes[0][c], to_np(x_show[c:c + 1]),
                str(col_labels[c]) if c < len(col_labels) else "", cmap=cmap)
    axes[0][0].set_ylabel("input", rotation=0, labelpad=44, fontsize=9, va="center")
    axes[0][0].axis("on"); axes[0][0].set_xticks([]); axes[0][0].set_yticks([])

    for r_i, name in enumerate(names, start=1):
        for c in range(n):
            show_ax(axes[r_i][c], to_np(recons[name][c:c + 1]), "", cmap=cmap)
        axes[r_i][0].set_ylabel(f"{name}\nMSE {mses[name]:.4f}", rotation=0,
                                labelpad=52, fontsize=8, va="center")
        axes[r_i][0].axis("on")
        axes[r_i][0].set_xticks([]); axes[r_i][0].set_yticks([])

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return mses
