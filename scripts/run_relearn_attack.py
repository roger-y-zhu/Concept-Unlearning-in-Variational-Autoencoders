"""
scripts/run_relearn_attack.py
=============================
WEIGHT-SPACE relearning attack: an alternative recoverability criterion to the decoder-side
input attack `attack@r` (ledger: supervisor action item, Oct 2026).

Threat model (Hu et al. 2024 arXiv:2406.13356 "benign relearning"; Deeb & Roger 2024
arXiv:2410.08827; Lynch et al. 2024 arXiv:2402.16835): the attacker has the unlearned WEIGHTS and a
tiny amount of data, and fine-tunes. If the concept is still "in the weights" the model regains
the capability far faster than a model that never had it.

Two attacker arms (same budget: k examples, `steps` Adam steps, batch 32):
  forget  fine-tune on k concept-POSITIVE (forget) examples drawn from a split DISJOINT from the
          evaluation images (dSprites/MPI3D: train-split vs held-out test split; CelebA has no
          partition, so a disjoint index split of the one forget set).
  retain  fine-tune on k concept-NEGATIVE (retain) examples -- the attacker never sees the
          concept ("benign relearning"; Deeb & Roger's 'facts independent of the eval facts').

Measurement: mean per-sample negative ELBO (kl_weight=1, decode(mu); identical to TOFU/U-LiRA) on
the held-out FORGET evaluation images, recorded at several fine-tuning step counts. Reference
models (baseline = knows the concept, oracle = retrained without it) go through the SAME attack,
so the analysis can normalise:  advantage(s) = (E_oracle'(s) - E_model'(s)) / (E_oracle(0) - E_baseline(0)).
0 = relearns no better than a model that never had the concept; 1 = as good as the baseline.

Writes results/<ds>/relearn_attack.json (separate file; dense_recovery.json is untouched).
Resumable (row key = method|run). Root of the project via env CU_ROOT.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(os.environ.get("CU_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from src.evaluation.membership import per_sample_negative_elbo  # noqa: E402
from notebooks.nb_utils import load_vae_checkpoint  # noqa: E402
from run_tofu_forget_quality import row_ckpt  # noqa: E402  (b4 entries added 2026-10-16)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_CAP = 3000
VERSION = 1


def _x(item):
    return item[0] if isinstance(item, (list, tuple)) else item


def _gather(ds, idx):
    return torch.stack([_x(ds[int(i)]) for i in idx]).float()


def build_sets(dataset: str, k: int, seed: int = 0):
    """-> (A_forget[k], A_retain[k], E_forget[<=3000]) as CPU tensors; deterministic."""
    base = dataset.removesuffix("_b4")
    # The EVALUATION set is fixed (identical for every seed); `seed` only changes which k attacker examples are drawn.
    # seed 0 reproduces the original draw order exactly (results/<ds>/relearn_attack.json).
    rng0 = np.random.default_rng(0)
    rng = rng0 if seed == 0 else np.random.default_rng(seed)
    if base in ("dsprites", "mpi3d"):
        if base == "dsprites":
            from src.data.dsprites import get_concept_loaders
            kw = dict(npz_path=str(ROOT / "data/dsprites/dsprites.npz"), concept_col=1, concept_val=2)
        else:
            from src.data.mpi3d import get_concept_loaders
            kw = dict(npz_path=str(ROOT / "data/mpi3d/mpi3d_real.npz"), forget_factor=0, forget_val=2)
        f_tr, r_tr = get_concept_loaders(batch_size=256, num_workers=0, split="train", **kw)
        f_te, _ = get_concept_loaders(batch_size=256, num_workers=0, split="test", **kw)
        fa = rng.permutation(len(f_tr.dataset))[:k]
        ra = rng.permutation(len(r_tr.dataset))[:k]
        ev = np.arange(min(EVAL_CAP, len(f_te.dataset)))          # same subset as run_tofu_forget_quality
        return _gather(f_tr.dataset, fa), _gather(r_tr.dataset, ra), _gather(f_te.dataset, ev)
    from src.data.celeba import get_concept_loaders, EYEGLASSES_COL
    f, r = get_concept_loaders(data_root=str(ROOT / "data/celeba/img_align_celeba"),
                               forget_attr=EYEGLASSES_COL, batch_size=256, num_workers=0)
    perm = rng0.permutation(len(f.dataset))                        # fixed eval/attacker partition for every seed
    ev = perm[:EVAL_CAP]
    fa = perm[EVAL_CAP:EVAL_CAP + k] if seed == 0 else perm[EVAL_CAP:][rng.permutation(len(perm) - EVAL_CAP)[:k]]
    ra = rng.permutation(len(r.dataset))[:k]
    return _gather(f.dataset, fa), _gather(r.dataset, ra), _gather(f.dataset, ev)


def eval_elbo(model, E):
    ld = DataLoader(TensorDataset(E), batch_size=256, shuffle=False)
    return float(per_sample_negative_elbo(model, ld, DEVICE).mean())


def attack(ckpt: Path, A: torch.Tensor, E: torch.Tensor, beta: float, lr: float, eval_at, seed: int = 0):
    """Fine-tune a copy on A; return {step: mean forget-eval negELBO} incl. step 0."""
    model, _ = load_vae_checkpoint(ckpt, DEVICE)
    curve = {0: eval_elbo(model, E)}
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = A.to(DEVICE)
    bs, n, step = 32, len(A), 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    stop = max(eval_at)
    # BatchNorm stays in EVAL mode: fine-tuning with train-mode BN on a one-class batch normalises with
    # hearts-only batch statistics but is evaluated with running statistics -> a train/eval mismatch that
    # made even the ORACLE's held-out forget loss worse (436 -> 816, calibration 2026-10-16). The attack
    # is meant to change WEIGHTS only; gradients still flow, the model has no dropout.
    model.eval()
    while step < stop:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            if step >= stop:
                break
            x = A[perm[i:i + bs].to(DEVICE)]
            loss = model(x, kl_weight=beta).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step in eval_at:
                curve[step] = eval_elbo(model, E)
    return {str(s): v for s, v in curve.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["dsprites", "celeba", "mpi3d", "dsprites_b4", "celeba_b4", "mpi3d_b4"])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-at", default="10,25,50,100")
    ap.add_argument("--seed", type=int, default=0, help="attacker-sample / fine-tuning seed (eval set is fixed)")
    ap.add_argument("--out", default=None, help="default results/<ds>/relearn_attack[_k<k>_lr<lr>_s<seed>].json")
    ap.add_argument("--max-rows", type=int, default=0, help="debug/calibration: refs + first N unlearned rows")
    a = ap.parse_args()
    ds, eval_at = a.dataset, sorted({int(s) for s in a.eval_at.split(",")})
    beta = 4.0 if ds.endswith("_b4") else 1.0
    default_cfg = (a.k == 64 and a.lr == 1e-4 and a.seed == 0)
    out = Path(a.out) if a.out else ROOT / f"results/{ds}/relearn_attack{'' if default_cfg else f'_k{a.k}_lr{a.lr:g}_s{a.seed}'}.json"
    rows_in = json.loads((ROOT / f"results/{ds}/dense_recovery.json").read_text())
    if a.max_rows:
        rows_in = [r for r in rows_in if r["method"] in ("baseline", "oracle")] + \
                  [r for r in rows_in if r["method"] not in ("baseline", "oracle")][:a.max_rows]
    meta = dict(version=VERSION, dataset=ds, k=a.k, lr=a.lr, eval_at=eval_at, beta=beta,
                steps_batch=32, eval_cap=EVAL_CAP, seed=a.seed)
    done = {}
    if out.exists() and not a.max_rows:
        old = json.loads(out.read_text())
        if old.get("meta") == meta:
            done = {f"{r['method']}|{r['run']}": r for r in old["rows"]}
    A_f, A_r, E = build_sets(ds, a.k, a.seed)
    print(f"{ds}: attacker k={a.k}  eval n={len(E)}  beta={beta}  lr={a.lr}  eval_at={eval_at}  "
          f"({len(done)} rows cached)", flush=True)
    res, t0 = [], time.time()
    for i, r in enumerate(rows_in):
        key = f"{r['method']}|{r['run']}"
        if key in done:
            res.append(done[key]); continue
        ck = row_ckpt(ds, r)
        if not ck.exists():
            continue
        row = dict(method=r["method"], run=r["run"],
                   forget=attack(ck, A_f, E, beta, a.lr, eval_at, a.seed),
                   retain=attack(ck, A_r, E, beta, a.lr, eval_at, a.seed))
        res.append(row)
        f, rt = row["forget"], row["retain"]
        print(f"  [{i+1}/{len(rows_in)}] {r['method']:<15} {r['run'][:26]:<26} "
              f"E0={f['0']:.1f}  forget-ft@{eval_at[-1]}={f[str(eval_at[-1])]:.1f}  "
              f"retain-ft@{eval_at[-1]}={rt[str(eval_at[-1])]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        if not a.max_rows and len(res) % 10 == 0:
            out.write_text(json.dumps(dict(meta=meta, rows=res), indent=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, rows=res), indent=1))
    print(f"-> {out}  ({len(res)} rows)")


if __name__ == "__main__":
    main()
