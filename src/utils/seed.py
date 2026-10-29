# src/utils/seed.py

import os
import random
import numpy as np
import torch


def seed_all(seed: int) -> None:
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU + GPU RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN determinism (GPU reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Avoid nondeterministic CUDA algorithms where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"