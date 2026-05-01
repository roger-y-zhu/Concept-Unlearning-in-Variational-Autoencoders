import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data.dsprites import get_dataloaders
from src.evaluation.probing import encode_dataset, linear_probe, mlp_probe
from src.models.vae import ConvVAE, VAEConfig
from src.training.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/dsprites/dsprites_vae_baseline/epoch_0150.pt"
DATA_PATH  = ROOT / "data/dsprites/dsprites.npz"
OUT_DIR    = ROOT / "results/dsprites/dsprites_vae_baseline"
DEVICE     = "cuda"


def tick(msg):
    print(f"  {msg}...", end=" ", flush=True)
    return time.time()

def tock(t0):
    print(f"({time.time() - t0:.1f}s)")

def print_results(name, r):
    print(f"  {'accuracy':<12}{r['accuracy']:.2%}")
    print(f"  {'auroc':<12}{r['auroc']:.2%}")
    print(f"  {'precision':<12}{r['precision']:.2%}")
    print(f"  {'recall':<12}{r['recall']:.2%}")
    print(f"  {'f1':<12}{r['f1']:.2%}")


# ── Model ────────────────────────────────────────────────────────────────────
print("[ Model ]")
t = tick("Loading checkpoint")
model = ConvVAE(VAEConfig(latent_dim=8, in_channels=1, recon_loss="bce")).to(DEVICE)
load_checkpoint(model, CHECKPOINT, DEVICE)
tock(t)

# ── Data ─────────────────────────────────────────────────────────────────────
print("[ Data ]")
t = tick("Loading dSprites")
train_loader, _, test_loader = get_dataloaders(
    DATA_PATH, include_labels=True, batch_size=512, num_workers=0
)
tock(t)

t = tick("Encoding train set")
mu_train, y_train = encode_dataset(model, train_loader, DEVICE)
tock(t)

t = tick("Encoding test set")
mu_test, y_test = encode_dataset(model, test_loader, DEVICE)
tock(t)

y_train_bin = (y_train == 2).astype(int)
y_test_bin  = (y_test  == 2).astype(int)
print(f"  Heart positives — train: {y_train_bin.sum()} / {len(y_train_bin)}  "
      f"test: {y_test_bin.sum()} / {len(y_test_bin)}")

# ── Probes ───────────────────────────────────────────────────────────────────
print("[ Probes ]")
t = tick("Linear probe")
lin = linear_probe(mu_train, y_train_bin, mu_test, y_test_bin)
tock(t)
print("  Linear probe:")
print_results("linear", lin)

t = tick("MLP probe")
mlp = mlp_probe(mu_train, y_train_bin, mu_test, y_test_bin)
tock(t)
print("  MLP probe:")
print_results("mlp", mlp)

# ── Plot ─────────────────────────────────────────────────────────────────────
print("[ Plot ]")
t = tick("Saving ROC curve")
fpr_lin, tpr_lin, _ = roc_curve(y_test_bin, lin["proba"])
fpr_mlp, tpr_mlp, _ = roc_curve(y_test_bin, mlp["proba"])

plt.figure(figsize=(5, 5))
plt.plot(fpr_lin, tpr_lin, label=f"Linear  (AUROC={lin['auroc']:.3f})")
plt.plot(fpr_mlp, tpr_mlp, label=f"MLP     (AUROC={mlp['auroc']:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Heart probe — dSprites VAE baseline")
plt.legend()
plt.tight_layout()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.grid(True)
plt.show()
tock(t)