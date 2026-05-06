# Running `shap_cluster_feature_selection.py` on an HPC Cluster

A short, copy-pasteable guide. Assumes you have already moved your `train.jsonl` and `test.jsonl` to the cluster.

---

## 1. Minimal files to upload

```
EMBER2024/
├── pyproject.toml
├── setup.cfg
├── src/thrember/...
├── examples/lgbm_config.json
├── custom_scripts/
│   ├── train_custom_lgbm.py             # imported by the SHAP script
│   └── shap_cluster_feature_selection.py
├── Documentation/
│   └── feature_index_map.json           # optional, for nicer block/field columns in the CSV
└── data/
    ├── PE_train.jsonl                   # filename must contain "train"
    └── PE_test.jsonl                    # filename must contain "test"
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

That installs `lightgbm`, `scikit-learn`, `pefile`, `polars`, `tqdm`, `signify`, `huggingface_hub`, `matplotlib`, `numpy`, **and `shap`** automatically. `scipy` (used for the hierarchical clustering) is pulled in transitively by `scikit-learn`.

If your cluster blocks internet on compute nodes, run the `pip` commands on a login node.

---

## 3. Job

**CPU-only.** LightGBM is configured with `"device_type": "cpu"` and TreeSHAP runs on CPU — no GPU needed.

### Recommended resources (100k train + 25k test)

| Resource | Recommended | Notes |
|---|---|---|
| Hours | **2** | ~10–30 min `lgb.train`, ~15–45 min TreeSHAP at `--shap-sample 20000`, plus vectorization on first run. 2 h leaves comfortable headroom. |
| Nodes | **1** | Single-node, multi-threaded job. The script doesn't do any cross-node work. |
| CPUs | **16** | LightGBM + TreeSHAP scale well with threads. 8 also works (slower); >16 sees diminishing returns at this dataset size. |
| RAM (GB) | **32** | Peak is dominated by the two feature matrices (~1.2 GB combined for 100k+25k × 2,381 float32) and a 20k×2,381 SHAP matrix (~190 MB). 32 GB is generous; 16 GB would also work. |

Memory drivers, in order of size:
1. Full train + test feature matrices in RAM (`n_rows × dim × 4` bytes each).
2. The `(n_shap_sample, dim)` SHAP matrix as float32.
3. The `dim × dim` correlation/distance matrix as float32 (~22 MB for dim ≈ 2.4k — negligible).

If you OOM, lower `--shap-sample` first (it caps the SHAP cost) before touching the data sizes.

### Run line

From the `EMBER2024/` folder, with the venv active:

```bash
mkdir -p runs

python custom_scripts/shap_cluster_feature_selection.py \
    data \
    runs/baseline_model.txt \
    --config-file examples/lgbm_config.json \
    --drop-fraction 0.10 \
    --shap-sample 20000 \
    --cluster-threshold 0.10 \
    --feature-map Documentation/feature_index_map.json \
    --dropped-out runs/dropped_features.csv
```

### Argument cheat sheet

| Flag | Default | Notes |
|---|---|---|
| `data_dir` | — | Folder holding `*train*.jsonl` and `*test*.jsonl`. `.dat` files are written here. |
| `model_path` | — | Where to save the trained baseline LightGBM booster. |
| `--config-file` | — | Required. Must have `"objective": "binary"`. |
| `--early-stopping` | `50` | Rounds on the val fold. `0` disables. |
| `--drop-fraction` | `0.10` | Fraction of `dim` to drop. Must be in (0, 1). |
| `--shap-sample` | `20000` | Stratified subsample for TreeSHAP. Capped at `len(X_tr)`. |
| `--cluster-threshold` | `0.10` | `fcluster` distance cutoff. `0.10 ≈ R ≥ 0.995`. Must be in (0, √2]. |
| `--shap-var-eps` | `1e-12` | Variance floor; columns below this are tagged `inactive`. |
| `--feature-map` | `../Documentation/feature_index_map.json` | Optional. Falls back to stub `block`/`field` names if missing or dim-mismatched. |
| `--dropped-out` | `dropped_features.csv` | Output CSV path. |

---

## 4. What you get back

In your data folder:
- `X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat` — vectorized features (auto-created on first run; reused if re-running).

Where you pointed `model_path`:
- `runs/baseline_model.txt` — trained baseline LightGBM booster (saved at `best_iteration` if early stopping fired).

Where you pointed `--dropped-out`:
- `runs/dropped_features.csv` — ranked drop list with columns:
  `rank, index, block, field, hashed, mean_abs_shap, cluster_id, drop_reason`
  where `drop_reason ∈ {inactive, redundant, low_importance}`.

In stdout (redirect with `... | tee run.log` if you want a copy on disk):
- Baseline test metrics (acc / AUC / log_loss).
- Inactive-feature count, cluster count, and the per-tier drop summary.

**Next step (separate manual run):** feed `dropped_features.csv` into `custom_scripts/drop_features.py` to produce a reduced-feature dataset and retrain.

---

## 5. Common errors

| Message | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'shap'` | You installed before `shap` was added. Re-run `pip install -e .` (or `pip install shap`). |
| `No .jsonl file containing 'train'` | Rename your file so the name contains `train` (and the other one `test`). |
| `Expected binary labels {0, 1}` | Your JSONL has labels other than 0/1 — filter them out. |
| `This script is binary-only` | Wrong config; use `examples/lgbm_config.json`. |
| `--cluster-threshold must be in (0, sqrt(2)~=1.414214]` | Pass a value in that range; `0.10` is the default. |
| `feature index map dim mismatch` (warning, not fatal) | Your `feature_index_map.json` is stale relative to the extractor — the run still finishes with stub `block`/`field` columns. Regenerate the map if you want pretty names. |
| OOM kill | Lower `--shap-sample`, then request more RAM. |
