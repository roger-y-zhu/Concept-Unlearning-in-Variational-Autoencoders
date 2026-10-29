# Tests

```bash
python -m pytest              # run everything (~26s)
python -m pytest -q            # quiet
python -m pytest -k rmu        # just RMU-related tests
python -m pytest tests/test_regressions.py -v   # today's pinned bugs, verbose
```

No installed dataset or GPU required — everything runs on CPU against tiny
synthetic models and data. Nothing under `results/` or `data/` is read or
written.

## Files

| File | Covers |
|---|---|
| `test_vae.py` | `ConvVAE` basics: shapes, encode/decode, loss decomposition, BCE vs MSE modes |
| `test_evaluation.py` | Do CKA / probing / kNN / clustering measure what they claim, on toy examples with a known correct answer (identical reps → CKA=1, separated clusters → AUROC≈1, random labels → AUROC≈0.5, etc.) |
| `test_unlearning_methods.py` | Does each method (NegGrad+, Finetune, Feature Unlearning, RMU) actually do what it claims — forget ELBO rises under ascent, the encoder stays frozen for Feature Unlearning but moves for RMU, forget latents are pulled toward RMU's steering target, `project_out` actually zeroes the projected component |
| `test_unlearning_loop.py` | End-to-end: every registered method runs through `run_unlearning`, checkpointing, skip-if-done, `history.json`, run-name → output-path convention |
| `test_data.py` | Train/val/test and forget/retain split disjointness and determinism, for both dSprites and CelebA loaders |
| `test_config.py` | Real experiment configs still load; the dead `unlearning:` config path stays rejected |
| `test_regressions.py` | One test per bug found in the 2026-08-04 audit — BatchNorm train/eval mismatch, logit-space vs image-space MSE, train-split Optuna scoring, degenerate metric denominators, sweep-key prefix collisions. Written to fail against the pre-fix code. |

## Fixtures (`conftest.py`)

- `tiny_vae_bce` / `tiny_vae_mse` — small (`base_channels=4, latent_dim=4`) real `ConvVAE` instances, fast enough for CPU
- `synthetic_dsprites_npz` — a tiny `.npz` with the same keys as the real dSprites file (`imgs`, `latents_values`, `latents_classes`)
- `synthetic_celeba_dir` — a tiny folder of generated images + `list_attr_celeba.txt` in the real on-disk layout
- `tiny_spec` — an in-memory `DatasetSpec` for testing `src/unlearning/loop.py` without any file-based dataset

## Adding a test for a new method

If you add a method to `src/unlearning/methods.py` (see `docs/ARCHITECTURE.md`),
give it a `TestYourMethod` class in `test_unlearning_methods.py` that checks
the property specific to its docstring's claim — not just that `step()` runs.
The pattern used for RMU (`test_encoder_changes_unlike_feature_unlearn`,
`test_forget_latents_move_toward_the_steering_target`) is a reasonable template:
run a few real optimiser steps and assert the thing the method exists to do
actually happened.
