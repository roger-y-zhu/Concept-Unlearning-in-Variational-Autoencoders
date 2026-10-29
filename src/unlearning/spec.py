"""
src/unlearning/spec.py
======================
Everything that differs between dSprites and CelebA, in one place.

Before this module the two datasets were expressed by forking every unlearning
script into ``experiments/unlearn/{dataset}/{method}.py``.  That duplicated the
training loop, checkpointing and argument parsing 6-8 times, and a single
BatchNorm bug in feature unlearning ended up living in four separate copies.

A DatasetSpec answers four questions:
    * where is the baseline checkpoint, and where do results go
    * how do I get (forget, retain) training loaders
    * how do I get a retain validation loader
    * how do I get HELD-OUT (forget, retain) loaders for scoring

Add a dataset by writing one subclass and registering it in DATASETS.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SEED = 48025845


class DatasetSpec:
    """Base class -- see DSpritesSpec / CelebASpec below."""

    name: str
    baseline_ckpt: Path
    results_dir: Path      # e.g. results/dsprites
    dir_prefix: str        # e.g. "dsprites"  -> results/dsprites/dsprites_neg_grad
    batch_size: int
    num_workers: int
    concept_desc: str      # human-readable, for logs
    train_beta: float = 1.0  # KL weight the VAE was TRAINED at; 1.0 for the
                             # standard datasets, 4.0 for the beta-VAE variants.
                             # Only affects the retain-budget reference (see
                             # baseline_val); LGA and the attack are beta-independent.
    train_objective: str = "elbo"  # "btcvae" for the beta-TCVAE variants; forces the
                                   # beta=1 ELBO budget yardstick (recorded loss has TC).

    # -- loaders ------------------------------------------------------------

    def concept_loaders(self, num_workers: int | None = None
                        ) -> tuple[DataLoader, DataLoader]:
        """(forget, retain) from the TRAIN split -- what unlearning trains on."""
        raise NotImplementedError

    def val_loader(self, num_workers: int | None = None) -> DataLoader:
        """Retain validation loader -- drives best.pt selection."""
        raise NotImplementedError

    def eval_loaders(self, num_workers: int | None = None
                     ) -> tuple[DataLoader, DataLoader]:
        """(forget, retain) HELD OUT -- what Optuna objectives are scored on.

        Scoring on the train split makes a retain objective reward overfitting,
        so tuning must never use ``concept_loaders`` for measurement.
        """
        raise NotImplementedError

    # -- paths --------------------------------------------------------------

    def method_dir(self, method: str) -> Path:
        """results/<dataset>/<dataset>_<method>  -- unchanged from the old layout."""
        return self.results_dir / f"{self.dir_prefix}_{method}"

    def optuna_dir(self, method: str) -> Path:
        return self.results_dir / f"{self.dir_prefix}_{method}_optuna"

    def _workers(self, num_workers: int | None) -> int:
        return self.num_workers if num_workers is None else num_workers


# ---------------------------------------------------------------------------
# dSprites -- forget concept: hearts (latents_classes[:, 1] == 2)
# ---------------------------------------------------------------------------

class DSpritesSpec(DatasetSpec):
    name          = "dsprites"
    baseline_ckpt = ROOT / "results/dsprites/dsprites_baseline/epoch_0150.pt"
    results_dir   = ROOT / "results/dsprites"
    dir_prefix    = "dsprites"
    batch_size    = 512
    num_workers   = 0
    concept_desc  = "hearts (shape=2)"

    npz_path      = ROOT / "data/dsprites/dsprites.npz"
    concept_col   = 1
    concept_val   = 2

    def concept_loaders(self, num_workers=None):
        from src.data.dsprites import get_concept_loaders
        return get_concept_loaders(
            npz_path=str(self.npz_path),
            concept_col=self.concept_col, concept_val=self.concept_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, split="train",
        )

    def val_loader(self, num_workers=None):
        from src.data.dsprites import get_retain_dataloaders
        _, val_loader, _ = get_retain_dataloaders(
            npz_path=str(self.npz_path),
            concept_col=self.concept_col, concept_val=self.concept_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, include_labels=False,
        )
        return val_loader

    def eval_loaders(self, num_workers=None):
        from src.data.dsprites import get_concept_loaders
        return get_concept_loaders(
            npz_path=str(self.npz_path),
            concept_col=self.concept_col, concept_val=self.concept_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, split="test",
        )


# ---------------------------------------------------------------------------
# CelebA -- forget concept: Eyeglasses (~6.5% of images)
# ---------------------------------------------------------------------------

class CelebASpec(DatasetSpec):
    name          = "celeba"
    baseline_ckpt = ROOT / "results/celeba/celeba_baseline/epoch_0250.pt"
    results_dir   = ROOT / "results/celeba"
    dir_prefix    = "celeba"
    batch_size    = 128
    num_workers   = 4
    concept_desc  = "Eyeglasses"

    data_root     = ROOT / "data/celeba/img_align_celeba"
    forget_attr   = "Eyeglasses"

    def concept_loaders(self, num_workers=None):
        from src.data.celeba import get_concept_loaders
        return get_concept_loaders(
            data_root=str(self.data_root), forget_attr=self.forget_attr,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED,
        )

    def val_loader(self, num_workers=None):
        from src.data.celeba import get_retain_dataloaders
        _, val_loader, _ = get_retain_dataloaders(
            data_root=str(self.data_root), forget_attr=self.forget_attr,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED,
        )
        return val_loader

    def eval_loaders(self, num_workers=None):
        from src.data.celeba import get_concept_eval_loaders
        return get_concept_eval_loaders(
            data_root=str(self.data_root), forget_attr=self.forget_attr,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED,
        )


# ---------------------------------------------------------------------------
# MPI3D-real -- forget concept: object_color == red (factor 0, value 2)
# The "real world, disentangled" corner of the design space; see
# src/data/mpi3d.py. Concept label is BINARY (is-red), so eval uses forget_val=1
# on the encoded labels, exactly like CelebA -- not the raw factor value.
# ---------------------------------------------------------------------------

class MPI3DSpec(DatasetSpec):
    name          = "mpi3d"
    baseline_ckpt = ROOT / "results/mpi3d/mpi3d_baseline/epoch_0120.pt"
    results_dir   = ROOT / "results/mpi3d"
    dir_prefix    = "mpi3d"
    batch_size    = 256
    num_workers   = 4
    concept_desc  = "object_color = red"

    npz_path      = ROOT / "data/mpi3d/mpi3d_real.npz"
    forget_factor = 0     # object_color
    forget_val    = 2     # red

    def concept_loaders(self, num_workers=None):
        from src.data.mpi3d import get_concept_loaders
        return get_concept_loaders(
            npz_path=str(self.npz_path),
            forget_factor=self.forget_factor, forget_val=self.forget_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, split="train",
        )

    def val_loader(self, num_workers=None):
        from src.data.mpi3d import get_retain_dataloaders
        _, val_loader, _ = get_retain_dataloaders(
            npz_path=str(self.npz_path),
            forget_factor=self.forget_factor, forget_val=self.forget_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, include_labels=False,
        )
        return val_loader

    def eval_loaders(self, num_workers=None):
        from src.data.mpi3d import get_concept_loaders
        return get_concept_loaders(
            npz_path=str(self.npz_path),
            forget_factor=self.forget_factor, forget_val=self.forget_val,
            batch_size=self.batch_size, num_workers=self._workers(num_workers),
            seed=SEED, split="test",
        )


# ---------------------------------------------------------------------------
# dSprites beta=4 variant -- SAME data + forget concept as dSprites; only the
# trained VAE differs (KL weight 4.0 instead of 1.0). Results live in a parallel
# tree (results/dsprites_b4/) so the beta=1 results are untouched. Hypothesis: does
# beta=4's stronger latent organisation make LGA/geometry predict recoverability?
# Configs: configs/experiment/dsprites_b4_{baseline,oracle}.yaml (HPO-tuned per the
# hold-latent+beta, retune-everything-else policy). See ledger.
# ---------------------------------------------------------------------------

class DSpritesB4Spec(DSpritesSpec):
    name          = "dsprites_b4"
    baseline_ckpt = ROOT / "results/dsprites_b4/dsprites_b4_baseline/epoch_0150.pt"
    results_dir   = ROOT / "results/dsprites_b4"
    dir_prefix    = "dsprites_b4"
    concept_desc  = "hearts (shape=2), beta=4 VAE"
    train_beta    = 4.0
    # npz_path / concept_col / concept_val / batch_size / num_workers and the three
    # loader methods are inherited unchanged from DSpritesSpec: the DATA is identical,
    # only the model being unlearned differs.


class MPI3DB4Spec(MPI3DSpec):
    """beta=4 MPI3D variant -- same data + forget concept (object_color=red), only the
    trained VAE differs (KL weight 4.0). Parallel tree results/mpi3d_b4/. See ledger."""
    name          = "mpi3d_b4"
    baseline_ckpt = ROOT / "results/mpi3d_b4/mpi3d_b4_baseline/epoch_0120.pt"
    results_dir   = ROOT / "results/mpi3d_b4"
    dir_prefix    = "mpi3d_b4"
    concept_desc  = "object_color = red, beta=4 VAE"
    train_beta    = 4.0


class CelebAB4Spec(CelebASpec):
    """beta=4 CelebA variant -- same data + forget concept (Eyeglasses), only the trained
    VAE differs (KL weight 4.0). Parallel tree results/celeba_b4/. See ledger."""
    name          = "celeba_b4"
    baseline_ckpt = ROOT / "results/celeba_b4/celeba_b4_baseline/epoch_0250.pt"
    results_dir   = ROOT / "results/celeba_b4"
    dir_prefix    = "celeba_b4"
    concept_desc  = "Eyeglasses, beta=4 VAE"
    train_beta    = 4.0


class DSpritesBTCSpec(DSpritesSpec):
    """beta-TCVAE dSprites variant -- SAME data + forget concept as dSprites; only the
    trained VAE differs (total-correlation penalty, Chen 2018; tc_beta=6, KL weight 1.0).
    Parallel tree results/dsprites_btc/. train_objective='btcvae' routes baseline_val to
    the beta=1 ELBO yardstick (the recorded loss includes the TC term). Configs:
    configs/experiment/dsprites_btc_{baseline,oracle}.yaml. See second-model plan."""
    name            = "dsprites_btc"
    baseline_ckpt   = ROOT / "results/dsprites_btc/dsprites_btc_baseline/epoch_0150.pt"
    results_dir     = ROOT / "results/dsprites_btc"
    dir_prefix      = "dsprites_btc"
    concept_desc    = "hearts (shape=2), beta-TCVAE"
    train_beta      = 1.0
    train_objective = "btcvae"

class CelebABTCSpec(CelebASpec):
    """beta-TCVAE CelebA variant (tc_beta=6, KL weight 1.0). Parallel tree results/celeba_btc/."""
    name            = "celeba_btc"
    baseline_ckpt   = ROOT / "results/celeba_btc/celeba_btc_baseline/epoch_0250.pt"
    results_dir     = ROOT / "results/celeba_btc"
    dir_prefix      = "celeba_btc"
    concept_desc    = "Eyeglasses, beta-TCVAE"
    train_beta      = 1.0
    train_objective = "btcvae"


class MPI3DBTCSpec(MPI3DSpec):
    """beta-TCVAE MPI3D variant (tc_beta=6, KL weight 1.0). Parallel tree results/mpi3d_btc/."""
    name            = "mpi3d_btc"
    baseline_ckpt   = ROOT / "results/mpi3d_btc/mpi3d_btc_baseline/epoch_0120.pt"
    results_dir     = ROOT / "results/mpi3d_btc"
    dir_prefix      = "mpi3d_btc"
    concept_desc    = "object_color = red, beta-TCVAE"
    train_beta      = 1.0
    train_objective = "btcvae"

class DSpritesFlowSpec(DSpritesSpec):
    """RealNVP flow variant -- same data + forget concept (hearts); the generative model is
    a normalizing flow (invertible, exact likelihood) instead of a VAE. Parallel tree
    results/dsprites_flow/. train_objective='flow' -> budget uses the flow's own NLL
    retain-val reference (baseline_val -> _beta1_retain_val, which calls model.forward)."""
    name            = "dsprites_flow"
    # best-val checkpoint (flow NLL overfits past ~ep9; epoch_0040 is the overfit model).
    baseline_ckpt   = ROOT / "results/dsprites_flow/flow_dsprites_baseline/best.pt"
    results_dir     = ROOT / "results/dsprites_flow"
    dir_prefix      = "dsprites_flow"
    concept_desc    = "hearts (shape=2), RealNVP flow"
    # The flow holds BOTH retain and forget forward graphs for neg_grad's
    # retain_NLL - lambda*forget_NLL on a 4096-dim / 8-coupling model, so the VAE's
    # batch_size=512 saturates the 24 GB GPU and the CUDA allocator thrashes (epoch
    # ~5x slower, no OOM). 128 keeps all data (just more, smaller steps) with headroom.
    batch_size      = 128
    train_beta      = 1.0
    train_objective = "flow"


DATASETS: dict[str, DatasetSpec] = {
    "dsprites": DSpritesSpec(),
    "celeba":   CelebASpec(),
    "mpi3d":    MPI3DSpec(),
    "dsprites_b4": DSpritesB4Spec(),
    "mpi3d_b4":    MPI3DB4Spec(),
    "celeba_b4":   CelebAB4Spec(),
    "dsprites_btc": DSpritesBTCSpec(),
    "celeba_btc": CelebABTCSpec(),
    "mpi3d_btc": MPI3DBTCSpec(),
    "dsprites_flow": DSpritesFlowSpec(),
}


# ---------------------------------------------------------------------------
# Shared experimental facts -- derive these, never transcribe them
# ---------------------------------------------------------------------------
# The retain budgets (40.59 / 417.18) had been re-typed as literals in at least
# four executable places: the cache audit, both master-table warmers and the
# dose warmer.  They are DERIVED quantities -- baseline val x (1 + tolerance) --
# so each copy silently stops matching the moment a baseline is retrained, which
# has already happened once via the BatchNorm fix.  A stale budget does not
# raise; it quietly changes which runs count as in-budget, which is a change to
# the selection rule itself.
#
# Same argument for the method list, which existed in six files: adding a method
# meant six edits, and forgetting one produced a table quietly missing a row
# rather than an error.

VAL_TOLERANCE = 0.10        # retain val may exceed baseline by this fraction

#: SINGLE SOURCE OF TRUTH for the ball-constrained attack's report radius per
#: dataset -- the radius of MAXIMUM across-operating-point separation (max std of
#: attack success over the dense operating points), so attack@r best discriminates
#: between models. Imported by the dense study (run_dense_recovery.ADV_R), the
#: champion table (run_attack_robustness --report-r default), AND build_master_table,
#: so no path can silently report at a different radius. Re-tuned 2026-09 on the
#: FIXED consensus judge (the old {2,4,8} was calibrated against the starved attack;
#: dSprites/CelebA unchanged, MPI3D moved 8->6). Chosen on the attack distribution
#: only (never on any correlation) -> leakage-free.
REPORT_RADIUS = {"dsprites": 2.0, "celeba": 4.0, "mpi3d": 6.0,
                 "dsprites_b4": 4.0,  # max-separation radius on the b4 dense study (std peaks r=4), like beta=1 per-dataset
                 "mpi3d_b4": 6.0,     # max-separation on mpi3d_b4 dense (std peaks r=6), same as beta=1
                 "celeba_b4": 4.0,    # placeholder = beta=1 celeba; re-tune to max-separation after the celeba_b4 dense study
                 "dsprites_btc": 2.0,
                 "celeba_btc": 4.0, "mpi3d_btc": 6.0,
                 "dsprites_flow": 24.0} # dsprites_flow CALIBRATED 2026-10-09 from the trained flow (scripts/calibrate_flow_radii.py): ||z||~11, attack floor<=r16, transition 18-28; 24.0 is mid-transition (PROVISIONAL -- finalize from per-method spread after dense_recovery)

#: method name -> which direction of its forget_loss means MORE forgetting.
#: Mirrors each UnlearnMethod.forget_better; see src/unlearning/methods.py.
METHODS_FORGET_BETTER: dict[str, str] = {
    "neg_grad":        "higher",   # negative ELBO pushed up
    "finetune":        "higher",
    "fisher":          "higher",
    "leace":           "higher",
    "inlp":            "higher",
    "feature_unlearn": "lower",    # distance-to-target pushed down
    "rmu":             "lower",
}

_BASELINE_VAL_CACHE: dict[str, float] = {}


def baseline_val(dataset: str) -> float:
    """The baseline checkpoint's RECORDED validation loss -- the reference the
    retain budget sits 10% above. Cached per process (the analysis layer asks
    repeatedly).

    R4 (audited 2026-09): the baseline trains on the FULL dataset, so this
    recorded loss is the FULL-set val loss, which is ~1-3% ABOVE the baseline's
    true retain-only val loss (measured: dSprites +3.2%, CelebA +3.0%,
    MPI3D +1.1%). The budget is therefore ~1-3% LOOSER than a pure retain-val
    budget -- a uniform, bounded approximation, well under the 10% tolerance, and
    NOT the drifting-literal hazard this module warns about elsewhere (it is still
    derived from the checkpoint, not transcribed). Left as-is deliberately:
    recomputing a retain-only val here would pull model construction + a forward
    pass into this config module and shift every champion by a sub-tolerance
    amount. Do not silently "correct" it to `metrics['retain_loss']` unless the
    baselines are retrained to record one.
    """
    if dataset not in _BASELINE_VAL_CACHE:
        spec = DATASETS[dataset]
        if getattr(spec, "train_beta", 1.0) == 1.0 and getattr(spec, "train_objective", "elbo") == "elbo":
            import torch
            from src.models.vae import VAEConfig
            torch.serialization.add_safe_globals([VAEConfig])
            ckpt = torch.load(spec.baseline_ckpt, map_location="cpu", weights_only=False)
            _BASELINE_VAL_CACHE[dataset] = float(ckpt["metrics"]["loss"])
        else:
            # beta!=1 variant: the recorded loss is at the TRAINING beta (e.g. recon+4*KL),
            # which is NOT comparable to the beta=1 val the unlearning loop measures. Put the
            # budget on the common beta=1 yardstick by measuring the baseline's beta=1 retain
            # val the same way loop.run_val does (mean recon+1*KL over retain-val).
            _BASELINE_VAL_CACHE[dataset] = _beta1_retain_val(spec)
    return _BASELINE_VAL_CACHE[dataset]


#: Seeds of the beta=1 retain-val passes averaged into the beta!=1 budget reference, and the
#: version of that reference (bump to invalidate every stored beta1_retain_val.json).
BETA1_REF_SEEDS = (0, 1, 2, 3, 4)
BETA1_REF_VERSION = 1


def _beta1_retain_val(spec) -> float:
    """Mean beta=1 ELBO of a spec's baseline over its retain-val split -- the
    budget reference for beta!=1 variants (see baseline_val).

    DETERMINISTIC AND STORED (fix 2026-10-18, ledger S12). The VAE forward pass samples
    z, so the earlier single unseeded pass returned a different reference on every call
    (dSprites beta=4: 38.44 / 38.71 / 38.81 in three calls), which made the beta!=1 retain
    budget -- and every in-budget flag derived from it -- irreproducible. The reference is
    now the mean over BETA1_REF_SEEDS seeded passes, computed once and written next to the
    checkpoint (`beta1_retain_val.json`, keyed by checkpoint name/size/mtime and
    BETA1_REF_VERSION, with the per-seed values and their sd). Every caller -- the
    unlearning loop, the attack scripts and the analysis -- reads the stored value."""
    import json
    import torch
    ckpt_path = Path(spec.baseline_ckpt)
    st = ckpt_path.stat()
    key = {"ckpt": ckpt_path.name, "size": st.st_size, "mtime": int(st.st_mtime),
           "version": BETA1_REF_VERSION, "seeds": list(BETA1_REF_SEEDS)}
    store = ckpt_path.parent / "beta1_retain_val.json"
    if store.exists():
        rec = json.loads(store.read_text())
        if all(rec.get(k) == v for k, v in key.items()):
            return float(rec["mean"])
    from src.models.vae import ConvVAE, VAEConfig
    from src.models.flow import RealNVP, FlowConfig
    torch.serialization.add_safe_globals([VAEConfig, FlowConfig])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    _cfg = ckpt["model_config"]
    model = (RealNVP(_cfg) if type(_cfg).__name__ == "FlowConfig" else ConvVAE(_cfg)).to(dev).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    vals, n = [], 0
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    for s in BETA1_REF_SEEDS:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == "cuda" else []):
            torch.manual_seed(s)
            tot = n = 0
            with torch.no_grad():
                for batch in spec.val_loader(0):
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    tot += float(model(x.to(dev), kl_weight=1.0).loss.item()); n += 1
        vals.append(tot / max(n, 1))
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    store.write_text(json.dumps({**key, "mean": mean, "sd": sd, "values": vals, "n_batches": n,
                                 "device": dev}, indent=1))
    return mean


def val_budget(dataset: str, tolerance: float = VAL_TOLERANCE) -> float:
    """The retain-val budget every method is held to, computed not transcribed."""
    return baseline_val(dataset) * (1.0 + tolerance)


# ---------------------------------------------------------------------------
# finetune vs neg_grad: the same algorithm, distinguished only by lambda
# ---------------------------------------------------------------------------
# NegGrad and Finetune are both subclasses of _ElboAscent (src/unlearning/
# methods.py) with IDENTICAL step() code; they differ only in default lambda.
# A finetune run with lambda>0 is therefore byte-identical to the neg_grad run
# of the same (lambda, epochs, seed): it double-counts in the dense correlation
# and, on MPI3D, collided the two methods' budget champions onto one checkpoint
# (finetune's champion was a lambda=0.01 run identical to neg_grad's). finetune
# is DEFINED as retain-only (Finetune.default_lambda == 0.0; its step() comment:
# "0 = retain-only finetuning"), so it is restricted to lambda==0 runs wherever
# a champion or dense operating point is selected. This predicate is the single
# source of that rule -- both champion selection (nb_utils.collect_method_
# champions) and the dense study (run_dense_recovery.stratified_runs) call it.
# Added 2026-09-12 (finetune/neg_grad duplication audit; ledger E22).
import re as _re

_LAM_RE = _re.compile(r"^lam(?P<lam>[0-9]+(?:\.[0-9]+)?)_ep")


def method_run_filter(method: str):
    """Predicate on a run DIRECTORY NAME: is it a valid champion / dense
    operating-point candidate for `method`? Returns None (accept every run)
    for all methods except finetune, which is restricted to lambda_forget == 0
    (see the module note above)."""
    if method != "finetune":
        return None

    def keep(run_name: str) -> bool:
        m = _LAM_RE.match(run_name)
        return m is not None and float(m.group("lam")) == 0.0

    return keep
