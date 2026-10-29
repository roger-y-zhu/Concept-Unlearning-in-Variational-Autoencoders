"""
_probe_all_attrs.py
Probe all 40 CelebA attributes in the VAE latent space.
Reports linear AUROC, MLP AUROC, and concept direction norm for every attribute.
"""
import sys, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.vae import ConvVAE, VAEConfig

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT      = ROOT / "results/celeba/celeba_baseline/epoch_0250.pt"
DATA_ROOT = ROOT / "data/celeba/img_align_celeba"
ATTR_FILE = ROOT / "data/celeba/list_attr_celeba.txt"
BATCH     = 256
SEED      = 48025845

print(f"Device: {DEVICE}")

# 鈹€鈹€ Load model 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
torch.serialization.add_safe_globals([VAEConfig])
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model = ConvVAE(ckpt["model_config"]).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded epoch {ckpt['epoch']} | val_loss={ckpt['metrics']['loss']:.4f} "
      f"| latent_dim={ckpt['model_config'].latent_dim}")

# 鈹€鈹€ Load attribute file 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
with open(ATTR_FILE) as f:
    _          = f.readline()                 # n_images count line
    attr_names = f.readline().strip().split()
    rows = []
    for line in f:
        parts = line.strip().split()
        rows.append((parts[0], [int(x) for x in parts[1:]]))

print(f"Attributes: {len(attr_names)}  |  Images in attr file: {len(rows)}")

# 鈹€鈹€ Dataset: returns image + all 40 labels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
VALID = {".jpg", ".jpeg", ".png"}
tf = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CelebAAllAttrs(Dataset):
    def __init__(self, rows, root, transform):
        existing = {p.name for p in root.iterdir() if p.suffix.lower() in VALID}
        self.samples  = [(root / fn, vals) for fn, vals in rows if fn in existing]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, vals = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, torch.tensor(vals, dtype=torch.int8)

dataset = CelebAAllAttrs(rows, DATA_ROOT, tf)
print(f"Dataset size: {len(dataset):,}")

loader = DataLoader(dataset, batch_size=BATCH, shuffle=False,
                    num_workers=0, pin_memory=torch.cuda.is_available())

# 鈹€鈹€ Encode everything 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print("Encoding...", flush=True)
t0 = time.time()
all_mu, all_labels = [], []
with torch.no_grad():
    for i, (imgs, labels) in enumerate(loader):
        mu, _ = model.encode(imgs.to(DEVICE))
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())
        if (i + 1) % 100 == 0:
            print(f"  batch {i+1}/{len(loader)}", flush=True)

mu     = np.concatenate(all_mu,     axis=0)           # (N, 256)
labels = np.concatenate(all_labels, axis=0)           # (N, 40) in {-1, +1}
labels = ((labels + 1) // 2).astype(np.int32)        # remap to {0, 1}
print(f"Encoded {len(mu):,} images in {time.time()-t0:.1f}s")
print(f"mu shape: {mu.shape}  |  labels shape: {labels.shape}")

# 鈹€鈹€ Probe every attribute 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print("\nProbing all 40 attributes...", flush=True)
scaler = StandardScaler()
mu_s   = scaler.fit_transform(mu)

results = []
for col, name in enumerate(attr_names):
    y        = labels[:, col]
    pos_frac = float(y.mean())

    # Skip near-constant attributes (probe would be trivial / meaningless)
    if pos_frac < 0.01 or pos_frac > 0.99:
        results.append(dict(attr=name, pos_frac=pos_frac,
                            lin_auroc=float("nan"), mlp_auroc=float("nan"),
                            dir_norm=float("nan")))
        continue

    mu_tr, mu_te, y_tr, y_te = train_test_split(
        mu_s, y, test_size=0.3, random_state=SEED, stratify=y
    )

    # Linear probe
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    lr.fit(mu_tr, y_tr)
    lin_auroc = roc_auc_score(y_te, lr.predict_proba(mu_te)[:, 1])

    # MLP probe  (2-layer, early stopping)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                        random_state=SEED, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=15)
    mlp.fit(mu_tr, y_tr)
    mlp_auroc = roc_auc_score(y_te, mlp.predict_proba(mu_te)[:, 1])

    # Concept direction norm: how far apart are the two class centroids?
    dir_norm = float(np.linalg.norm(mu[y == 1].mean(0) - mu[y == 0].mean(0)))

    results.append(dict(attr=name, pos_frac=pos_frac,
                        lin_auroc=lin_auroc, mlp_auroc=mlp_auroc,
                        dir_norm=dir_norm))
    print(f"  {name:<25}  pos={pos_frac:.2f}  "
          f"lin={lin_auroc:.4f}  mlp={mlp_auroc:.4f}  |dir|={dir_norm:.2f}",
          flush=True)

# 鈹€鈹€ Rank by MLP AUROC 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
results.sort(
    key=lambda r: r["mlp_auroc"] if not np.isnan(r["mlp_auroc"]) else 0,
    reverse=True,
)

print("\n" + "=" * 78)
print(f"  {'Rank':<4} {'Attribute':<25} {'Pos%':>6} "
      f"{'Lin AUROC':>10} {'MLP AUROC':>10} {'Dir norm':>9}")
print("-" * 78)
for rank, r in enumerate(results, 1):
    lin = f"{r['lin_auroc']:.4f}" if not np.isnan(r['lin_auroc']) else "   n/a"
    mlp = f"{r['mlp_auroc']:.4f}" if not np.isnan(r['mlp_auroc']) else "   n/a"
    dn  = f"{r['dir_norm']:.2f}"  if not np.isnan(r['dir_norm'])  else "   n/a"
    print(f"  {rank:<4} {r['attr']:<25} {r['pos_frac']*100:>5.1f}%  "
          f"{lin:>10} {mlp:>10} {dn:>9}")

# Gap analysis: how much does MLP improve over linear?
print("\n=== Attributes where MLP >> Linear (non-linearity gap > 0.02) ===")
for r in results:
    if not (np.isnan(r['lin_auroc']) or np.isnan(r['mlp_auroc'])):
        gap = r['mlp_auroc'] - r['lin_auroc']
        if gap > 0.02:
            print(f"  {r['attr']:<25}  lin={r['lin_auroc']:.4f}  "
                  f"mlp={r['mlp_auroc']:.4f}  gap={gap:+.4f}")

