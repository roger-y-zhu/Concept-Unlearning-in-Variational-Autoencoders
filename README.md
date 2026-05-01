# Concept Unlearning in Variational Autoencoders

A research project exploring **concept unlearning in variational autoencoders**. Can we train a VAE and then selectively "forget" a concept from its latent space without retraining from scratch?

Tested on dSprites (shapes, scales, orientations) and CelebA (faces). Four unlearning strategies are in the works: gradient reversal, fine-tuning, feature suppression, and oracle baselines.

---

## What's in here

```
configs/        experiment, model, dataset, and unlearning configs (YAML)
experiments/    train.py and optuna hyperparameter tuning
scripts/        data subsampling utilities (celeba 75k, dsprites 200k)
src/
  models/       ConvVAE — works for both 1-channel and 3-channel images
  training/     training loop, KL annealing, checkpointing
  utils/        config loading, seeding
results/        training histories, tuning trials, figures
```

---

## Setup

```bash
pip install -r requirements.txt
```

You'll need to grab the datasets yourself:

- **dSprites** — downloads automatically on first run via the URL in `configs/dataset/dsprites.yaml`
- **CelebA** — point `data_root` in `configs/dataset/celeba.yaml` at your local copy of `img_align_celeba/`

If you want the subsampled versions (recommended for faster iteration):

```bash
python scripts/subsample_dsprites.py   # → data/dsprites/dsprites_200k.npz
python scripts/subsample_celeba.py     # → data/celeba/img_align_celeba_75k/
```

---

## Training

Pick an experiment config and go:

```bash
python experiments/train.py --experiment dsprites_vae_baseline
python experiments/train.py --experiment celeba75k_vae
python experiments/train.py --experiment dsprites_beta_vae
```

You can override config values on the fly:

```bash
python experiments/train.py --experiment dsprites_vae_baseline \
    --latent-dim 16 \
    --kl-max-weight 2.0 \
    --epochs 80
```

Checkpoints and training history land in `results/<dataset>/<experiment>/`.

---

## Hyperparameter tuning

Uses Optuna with a TPE sampler and median pruner. Results are persisted to SQLite so you can stop and resume freely:

```bash
python experiments/optuna.py --experiment dsprites200k_vae --n-trials 50
```

Trial configs and metrics are saved to `results/tuning/<experiment>/trial_N/`. A summary CSV is written at the end. You can also run `results/tuning/visualise.py` to plot the trial landscape.

---

## The model

`ConvVAE` is a fairly standard convolutional VAE for 64×64 images:

- **Encoder** — 4× stride-2 conv layers → flatten → linear heads for μ and σ
- **Decoder** — linear → reshape → 4× transposed conv layers
- **Std parameterisation** — softplus + ε (no log-var instability)
- **Loss** — summed reconstruction (BCE for dSprites, MSE for CelebA) + β-weighted KL
- **KL annealing** — linear warmup over configurable number of epochs

The β-VAE variant just cranks `kl_max_weight` above 1.

---

## Experiments at a glance

| Config | Dataset | Latent dim | Notes |
|---|---|---|---|
| `dsprites_vae_baseline` | dSprites | 16 | standard VAE |
| `dsprites200k_vae` | dSprites 200k | 16 | bigger subset, tuned β |
| `dsprites_beta_vae` | dSprites | 16 | β = 4.0 |
| `celeba_vae` | CelebA full | 128 | RGB, MSE loss |
| `celeba75k_vae` | CelebA 75k | 128 | smaller subset, lower lr |

---

## Reproducibility

Everything seeds via `src/utils/seed.py` — Python, NumPy, PyTorch CPU+GPU, and cuDNN determinism. The seed is set in `configs/base/training.yaml` and flows through all experiments and tuning trials.

---

## Project status

Training and tuning pipelines are solid. The unlearning methods (`configs/unlearning/`) are the next thing to implement — that's where the actual research lives.

---

## Requirements

See `requirements.txt`. Main dependencies: `torch`, `torchvision`, `optuna`, `numpy`, `pyyaml`.