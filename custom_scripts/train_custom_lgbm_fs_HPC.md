# Running `train_custom_lgbm_fs.py` on an HPC Cluster

A short, copy-pasteable guide. Assumes you have already moved your `train.jsonl` and `test.jsonl` to the cluster.

---

## 1. Minimal files to upload

Only these need to be on the cluster — nothing else from the repo is required:

```
EMBER2024/
├── pyproject.toml
├── setup.cfg
├── src/thrember/                       # the whole folder (the importable package)
├── examples/lgbm_config.json           # LightGBM hyperparameters
├── Documentation/feature_index_map.json   # optional, only used for nice feature names
└── custom_scripts/
    └── train_custom_lgbm_fs.py         # the script
```

Plus your data, anywhere on the cluster (any folder name works, e.g. `data/`):

```
data/
├── PE_train.jsonl       # filename must contain "train"
└── PE_test.jsonl        # filename must contain "test"
```

JSONL requirements: one PE record per line, in the standard thrember feature schema, with an integer `label` field (`0` = benign, `1` = malicious). Exactly one `*train*.jsonl` and one `*test*.jsonl` per data folder.

---

## 2. One-time setup on the cluster

From the `EMBER2024/` folder:

```bash
module load python/3.13.3         # whatever your cluster calls Python >= 3.10

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

That installs `lightgbm`, `scikit-learn`, `pefile`, `polars`, `tqdm`, `signify`, `huggingface_hub`, `matplotlib`, and `numpy` automatically.

If your cluster blocks internet on compute nodes, run those `pip` commands on a login node.

---

## 3. Submit script (SLURM)

Save as `run.slurm` next to the `EMBER2024/` folder, edit the two paths, then `sbatch run.slurm`.

```bash
#!/bin/bash
#SBATCH --job-name=ember_fs
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=run_%j.out

module load python/3.11
source EMBER2024/.venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python EMBER2024/custom_scripts/train_custom_lgbm_fs.py \
    /path/to/data \
    /path/to/output/model.txt \
    --config-file EMBER2024/examples/lgbm_config.json
```

Resources rule-of-thumb: 16 CPUs, 32 GB RAM, ~1–4 h for a full PE-set run; an 8 GB / 30 min job is fine for a ~100k sub-sample.

---

## 4. What you get back

In your data folder:
- `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat` — vectorized features (auto-created on first run; reused if re-running).

Where you pointed `model.txt`:
- `model.txt` — trained LightGBM booster.
- `model.txt.dropped_features.json` — the feature-selection report (which features to drop, gain/split per feature, information cost, timings, test accuracy).

In `run_<jobid>.out`:
- Training log + the `=== Feature Selection Report ===` summary.

---

## 5. Common errors

| Message | Fix |
|---|---|
| `No .jsonl file containing 'train'` | Rename your file so the name contains `train` (and the other one `test`). |
| `Expected binary labels {0, 1}` | Your JSONL has labels other than 0/1 — filter them out. |
| `This script is binary-only` | You picked the wrong config; use `examples/lgbm_config.json`. |
| OOM kill | Ask SLURM for more memory, or work on a smaller sub-sample. |
