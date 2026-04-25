# Concept Unlearning in VAEs
 
This project investigates **machine unlearning in Variational Autoencoders (VAEs)**, with a focus on whether removing a concept at the output level also removes its internal representation in the latent space.
 
The repository is structured to support:
- multiple **unlearning methods**
- multiple **datasets** (CelebA, dSprites)
- a consistent **evaluation framework** (probing, CKA, clustering, etc.)
- reproducible **experiments**
---
 
## Project Structure
 
The codebase is organised by **responsibility**, not by experiment. This keeps training, unlearning, and evaluation modular and comparable.
 
```
project/
│
├── data/                   # (ignored in git) raw + processed datasets
├── configs/                # experiment configs (YAML)
├── src/                    # core implementation
├── experiments/            # entry-point scripts
├── results/                # outputs (ignored in git)
├── notebooks/              # analysis / visualisation
├── tests/                  # sanity tests
├── requirements.txt
└── README.md
```
 
---
 
## `src/` — Core Code
 
All main logic lives here, split by functionality:
 
```
src/
│
├── data/           # dataset loading + preprocessing
├── models/         # model architectures (VAE, β-VAE)
├── training/       # training loops and losses
├── unlearning/     # unlearning methods
├── evaluation/     # metrics and analysis tools
├── utils/          # shared utilities
```
 
### Design principles
 
- **models/** → architecture only  
- **training/** → optimisation logic  
- **unlearning/** → methods (core contribution)  
- **evaluation/** → measurement framework  
This separation ensures:
- methods are comparable
- evaluation is consistent
- code is reusable
---
 
## `experiments/` — Entry Points
 
Thin scripts that orchestrate experiments:
 
```
experiments/
├── train_baseline.py
├── run_oracle.py
├── run_finetune.py
├── run_feature_unlearning.py
├── evaluate_all.py
```
 
Each script:
1. Loads a config  
2. Calls functions from `src/`  
3. Saves outputs to `results/`  
No heavy logic should live here.
 
---
 
## `configs/` — Reproducibility
 
All hyperparameters and experiment settings are defined here:
 
```
configs/
├── base_vae.yaml
├── beta_vae.yaml
│
├── unlearning/
│   ├── oracle.yaml
│   ├── finetune.yaml
│   ├── neg_grad.yaml
│   └── feature.yaml
│
├── datasets/
│   ├── celeba.yaml
│   └── dsprites.yaml
```
 
This allows:
- clean comparison between methods  
- easy hyperparameter tuning  
- reproducible experiments  
---
 
## `results/` — Outputs
 
Structured by dataset → method:
 
```
results/
├── celeba/
│   ├── baseline/
│   ├── oracle/
│   ├── finetune/
│   └── feature_unlearning/
│
├── dsprites/
│   └── ...
│
├── figures/
│   ├── cka/
│   ├── tsne/
│   └── probes/
```
 
Contains:
- model checkpoints  
- logs  
- evaluation outputs  
- plots  
(Not tracked in git.)
 
---
 
## `notebooks/`
 
Used for:
- latent space visualisation (t-SNE / UMAP)
- exploratory analysis
- debugging
Core logic should **not** live here.
 
---
 
## `data/`
 
Stores:
- raw datasets (CelebA, dSprites)
- processed splits
This directory is **excluded from version control**.
 
---
 
## Key Ideas Behind the Structure
 
- **Decoupling**: training, unlearning, and evaluation are independent  
- **Comparability**: all methods share the same evaluation pipeline  
- **Reproducibility**: configs define all experiments  
- **Scalability**: easy to add new methods or metrics  
---
 
## Setup
 
```bash
pip install -r requirements.txt
```
 
## Running Experiments (Example)
 
```bash
python experiments/train_baseline.py
python experiments/run_feature_unlearning.py
python experiments/evaluate_all.py
```
 
---
 
## Notes
 
- Large files (datasets, checkpoints, results) are not tracked in git
- Configs should be used instead of hardcoding parameters
- All new methods should be added under `src/unlearning/`
- All new metrics should be added under `src/evaluation/`
---
 
## Author
 
Roger Zhu  
University of Queensland