# Running `train_custom_lgbm_fs.py` on an HPC Cluster

A short, copy-pasteable guide. Assumes you have already moved your `train.jsonl` and `test.jsonl` to the cluster.

---

## 1. Minimal files to upload

add data

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

## 3. Job

CPU

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
