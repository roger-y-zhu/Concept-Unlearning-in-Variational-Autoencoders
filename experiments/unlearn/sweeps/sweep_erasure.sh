#!/bin/bash
# Erasure sweeps: LEACE (one deterministic point) + INLP (n_iter curve).
#
# LEACE has no hyperparameter -- that is the point, so it contributes exactly
# one run per dataset.  INLP's --n-iter is the sweep axis: each round removes
# one more rank from mu-space, so the sweep traces erasure against collateral
# damage, which is the curve LEACE's single point sits on.
#
# n_iter is capped by latent_dim (8 on dSprites, 256 on CelebA): projecting
# out every direction leaves the zero map, so dSprites stops at 7.
set -u
cd "$(dirname "$0")/../../.."

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY="python"
RUN="experiments/unlearn/run.py"

echo "=== dSprites: LEACE ==="
"$PY" "$RUN" --dataset dsprites --method leace

echo "=== dSprites: INLP k=1..7 ==="
for k in 1 2 3 4 5 6 7; do
    echo "--- k=$k ---"
    "$PY" "$RUN" --dataset dsprites --method inlp --n-iter "$k"
done

echo "=== CelebA: LEACE ==="
"$PY" "$RUN" --dataset celeba --method leace

echo "=== CelebA: INLP k=1,2,4,8,16,32,64,128 ==="
for k in 1 2 4 8 16 32 64 128; do
    echo "--- k=$k ---"
    "$PY" "$RUN" --dataset celeba --method inlp --n-iter "$k"
done

echo "=== erasure sweeps done ==="
