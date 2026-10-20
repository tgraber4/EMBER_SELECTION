# EMBER Selection

Feature-selection experiments on the EMBER2024 PE-malware dataset.

This repo is based on the original **[EMBER2024](https://github.com/FutureComputing4AI/EMBER2024)** repository ([paper](https://arxiv.org/pdf/2506.05074)). That repo provides `thrember`, the library that turns a Windows PE file into a 2,568-dimension feature vector and trains LightGBM classifiers on it. This fork adds a pipeline on top of it. The pipeline samples the dataset, ranks features with three different selection methods, drops the weakest 10% (257 features), and retrains and evaluates a model on the smaller feature set for comparison.

For dataset details, download sizes, label types, and the Capa update, see the [original EMBER2024 README](https://github.com/FutureComputing4AI/EMBER2024#readme).

---

## Original vs. added files

### Files from the original EMBER2024 repo

```
LICENSE
README.md                      (replaced by this file)
pyproject.toml
setup.cfg
src/__init__.py
src/thrember/__init__.py
src/thrember/download.py
src/thrember/features.py
src/thrember/model.py
src/thrember/pefile_warnings.txt
examples/ember2024-notebook.ipynb
examples/eval_lgbm.py
examples/lgbm_config.json
examples/lgbm_config_family.json
examples/lgbm_config_tag.json
examples/train_lgbm.py
```

Only small edits have been made to these files:
- `src/thrember/features.py`: updated to the newer `signify` API (`signed_file.SignedPEFile`, `iter_embedded_signatures`, and wrapping `signed_data.certificates` in `list()` so extraction doesn't crash on signed files).
- `pyproject.toml` / `setup.cfg`: changed `signify` to `>=0.9.1,<0.10` and added `shap`.
- `examples/lgbm_config.json`: seeds changed from `0` to `42`.
- `examples/eval_lgbm.py`: fixed a print label ("FPR 0.1" → "FPR 1%").

### Added files

**Everything else was added on top of the original repo.** That includes `custom_scripts/`, `src/thrember_lite/`, `Documentation/`, `output/`, `runs/`, and `tsne/`.

---

## Project structure

```
EMBER2024/
├── src/
│   ├── thrember/                      # [original] feature extraction + LightGBM library
│   └── thrember_lite/                 # [added] train/predict on a reduced feature set
│       └── tests/
├── examples/                          # [original] upstream train/eval scripts + configs
├── custom_scripts/                    # [added]
│   ├── bernoulli_sample.py
│   ├── vectorize_dataset.py
│   ├── run_thrember_lite.py
│   ├── eval_thrember_lite.py
│   ├── thrember_lite.md
│   ├── Helper Scripts/
│   └── Feature Selection/
│       ├── Mutual Informatin/
│       ├── Embedded/
│       └── SHAP/
├── Documentation/                     # [added] design docs, plans, reference files
│   ├── Overview/
│   ├── Example Row/
│   └── Feature Selection/
├── output/                            # [added] drop lists + reports from each selection method
│   ├── Mutual_Information/
│   ├── Embedded/
│   └── SHAP/
├── runs/                              # [added] retrained models + ROC curves
├── tsne/                              # [added] PCA + t-SNE exploration of the family-filtered sample
│   ├── Code/                          #   scripts (tsne_figures/ holds the plotting scripts)
│   ├── Documentation/
│   ├── Scaling Output/                #   vectorized + scaled matrices, fitted scalers (git-ignored)
│   ├── PCA Output/                    #   X_pca.npy, pca.joblib (git-ignored) + loadings CSVs
│   └── TSNE Output/                   #   X_tsne.npy, plots, tsne_figures/
├── pyproject.toml, setup.cfg          # [original] package metadata
└── LICENSE                            # [original]
```

### Original files

#### `src/thrember/`
- **`__init__.py`**: Re-exports the public `thrember` API: the feature-extractor classes, the data/model helpers, and the downloaders. `import thrember` is all you need to use the library.
- **`features.py`**: The PE feature extractor. Each `FeatureType` subclass parses one part of a PE file (headers, sections, imports, strings, etc.) into a fixed-length block, and `PEFeatureExtractor` joins the 12 blocks into the 2,568-dim vector.
- **`model.py`**: Streams the raw JSONL files, converts them into `X_<subset>.dat` / `y_<subset>.dat` memmap files, and reads them back. It also trains binary and one-vs-rest LightGBM models, runs a grid search, and predicts on raw file bytes.
- **`download.py`**: Downloads and unzips EMBER2024 from HuggingFace (`joyce8/EMBER2024`), filtered by split and file type. It can also fetch the pretrained benchmark `.model` files.
- **`pefile_warnings.txt`**: A list of `pefile` parser warning templates. `PEFormatWarnings` uses it to give each warning a fixed position in the feature vector.

#### `examples/`
- **`train_lgbm.py`**: Command-line script that trains a LightGBM model on the full 2,568-dim vectors using `thrember.train_model`. It reads parameters from a JSON config and saves the model.
- **`eval_lgbm.py`**: Scores a saved model on the test and challenge sets and prints ROC AUC, PR AUC, and TPR at 1% FPR. It also saves the ROC curve to `Classifier_ROC_AUC.pdf`.
- **`lgbm_config.json`, `lgbm_config_family.json`, `lgbm_config_tag.json`**: LightGBM parameter files for the binary malware/benign, malware family (multiclass), and tag (multilabel) tasks. This project only uses the binary config.
- **`ember2024-notebook.ipynb`**: The upstream notebook for exploring the dataset. It shows loading, vectorizing, and training in one place.

### Added files

#### `custom_scripts/`
- **`bernoulli_sample.py`**: Downloads the PE shards (Win32, Win64, .NET) for train and test, then keeps each sample with a fixed probability based on a deterministic hash of its md5. Upstream stores every sample twice, so the script removes the duplicate (keeping the copy with Capa results) and writes `ember_data/PE_train_sampled.jsonl` and `PE_test_sampled.jsonl`.
- **`vectorize_dataset.py`**: Wraps `thrember.create_vectorized_features` to turn a folder of train/test/challenge JSONL files into the `.dat` memmap files that `thrember_lite` trains and evaluates on. It includes every `.jsonl` in the folder whose name contains `train`, `test`, or `challenge`, and does nothing if the `.dat` files already exist unless `OVERWRITE` is set.
- **`run_thrember_lite.py`**: Runs the reduced-feature pipeline end to end: vectorize, build `spec.json` from a drop list, and train a binary LightGBM model, with an optional prediction on a single file. Paths and settings are edited at the top of `main()`.
- **`eval_thrember_lite.py`**: The `thrember_lite` version of `examples/eval_lgbm.py`. It loads a model bundle, applies the same column slice to the test and challenge sets, and reports ROC AUC, PR AUC, and TPR at 1% FPR with an ROC PDF.
- **`thrember_lite.md`**: Short notes on running `run_thrember_lite.py` and `eval_thrember_lite.py` on the HPC cluster. Lists the requested resources and the order of steps.

#### `custom_scripts/Helper Scripts/`
- **`build_feature_index_map.py`**: Generates `feature_index_map.json`, which maps each of the 2,568 vector positions to its block, position within the block, readable field name, and a hashed/not-hashed flag. It checks every block's size against `PEFeatureExtractor`, so the map can't drift out of sync.
- **`check_labels.py`**: Reads a sampled JSONL and confirms every row has a `label`. Prints benign, malware, and unlabeled counts.
- **`count_rows.py`**: Counts rows in one JSONL file or in every JSONL under `ember_data/`. Reads line by line, so memory use stays low on large shards.
- **`download_challenge.py`**: Downloads the EMBER2024 challenge set into `ember_data/`. `create_vectorized_features` and the evaluation scripts both need it.

#### `custom_scripts/Feature Selection/Mutual Informatin/`
- **`mi_feature_selection.py`**: Vectorizes up to 100k labeled rows from the sampled training file and scores each feature against the label with `mutual_info_classif`, treating the categorical columns as categorical. Writes the 257 lowest-scoring features to `output/Mutual_Information/dropped_features.{csv,json}`.

#### `custom_scripts/Feature Selection/Embedded/`
- **`train_custom_lgbm.py`**: Trains a binary LightGBM classifier from a folder holding exactly one JSONL with `train` in its name and one with `test`. It vectorizes only those two files, writing the `.dat` files into the same folder, so it doesn't need the challenge set; the SHAP script reuses its helper functions.
- **`train_custom_lgbm_fs.py`**: Trains the same baseline model, then ranks features by LightGBM gain and split count. It drops exactly 10%: features with zero gain first, then the lowest-gain ones. It saves a JSON report that includes the "information cost" (the share of total gain that was dropped).
- **`convert_embedded_to_mi_format.py`**: Converts the embedded report (`embedded_dropped_features.json`) into the ranked-list format the MI script produces, saved as `updated_dropped_features.json`. This step is required, because `build-spec` can't read the raw report.
- **`train_custom_lgbm.md`**: Design notes for `train_custom_lgbm.py`. Covers why it bypasses `examples/train_lgbm.py` and how inputs are detected.
- **`train_custom_lgbm_fs_HPC.md`**: Copy-paste guide for running the embedded selection script on the HPC cluster. Covers which files to upload, one-time setup, and the run command.

#### `custom_scripts/Feature Selection/SHAP/`
- **`shap_cluster_feature_selection.py`**: Trains a baseline LightGBM model, computes TreeSHAP values, ranks features by mean |SHAP|, and groups features whose SHAP values are highly correlated. It drops the bottom 10% in three tiers (unused → redundant → low importance) and writes `shap_dropped_features.csv` and the baseline model to `output/SHAP/`.
- **`convert_shap_to_mi_format.py`**: Converts the SHAP drop-list CSV into the JSON format the MI script produces, saved as `output/SHAP/shap_dropped_features.json`. This step is required, because `build-spec` only reads JSON.
- **`shap_cluster_feature_selection_HPC.md`**: Copy-paste guide for running the SHAP selection script on the HPC cluster. Lists the minimum files to upload and the run command.

#### `src/thrember_lite/`
A small companion library that trains and predicts on a subset of columns without modifying `thrember`.
- **`__init__.py`**: Exports the public API: `FeatureSpec`, `read_vectorized_features`, `apply_spec`, `train_binary`, `ModelBundle`, `predict_file`, and `predict_batch`. It is the entry point for all `thrember_lite` usage.
- **`spec.py`**: `FeatureSpec` turns a drop list into the list of kept column indices and records where the categorical columns end up after slicing. It is saved as `spec.json` next to every model, so predictions always use the same columns as training.
- **`data.py`**: Loads `X_/y_<subset>.dat` and copies only the kept columns in 50k-row chunks, so the full 2,568-wide matrix is never held in RAM. `apply_spec` slices an array that is already in memory.
- **`train.py`**: `train_binary` follows the same steps as `thrember.train_model` (drop unlabeled rows, 90/10 stratified split, `lgb.train`), but passes the remapped categorical indices. It applies one seed to both the data split and all of LightGBM's random-number settings, so differences between runs come from the feature set rather than random noise.
- **`predict.py`**: `ModelBundle` saves and loads `model.txt` and `spec.json` together. `predict_file` and `predict_batch` extract the full vector, slice it with the spec, and score it.
- **`cli.py`**: Command-line interface with `build-spec`, `train`, and `predict` subcommands. Run it as `python -m thrember_lite.cli`.
- **`tests/`**: Unit tests for `FeatureSpec` and the chunked loader, plus a training smoke test on synthetic data. The smoke test also checks save/load round-trips and that runs are reproducible.

#### `Documentation/`
- **`Overview/process_overview.md`**: Walks through the pipeline from HuggingFace download → sampling → vectorization → training → evaluation. At each step it compares upstream `thrember` with the feature-dropping path; some file names in it are older than the current layout (for example, scripts at the repo root and a single `PE_both_sampled.jsonl`).
- **`feature_index_map.json`**: Maps every vector index (0–2567) to its block, field name, and hashed flag, plus the index range of each block. The selection scripts use it to attach readable names to column numbers.
- **`feature_vector_membership.md`**: Lists which fields in a raw JSONL row feed the feature vector and which are metadata, labels, or tags. Useful when reading a raw row.
- **`finish_thrember_lite_plan.md`**: What's left to finish `thrember_lite`: the tests still to write, test environment setup, and related fixes. Lists them in a suggested order.
- **`Example Row/`**: One sample row saved in several views (full row, feature fields only, non-feature fields only, top-level schema, extracted vector). `example_row_summary.md` explains each section of the row.
- **`Feature Selection/Mutual Information/`**: `mi_feature_selection_plan.md` is the design and parameter doc for the MI script. `mutual_info.md` holds short notes on the method and ideas for improving it.
- **`Feature Selection/Embedded/`**: `lightgbm_feature_selection_plan.md` is the four-phase gain/split selection plan. `embedded_process_audit.md` reviews how the script implements it and how the training config can bias which features end up with zero gain.
- **`Feature Selection/SHAP/`**: `EMBER2024_Feature_Selection_Plan-v4.md` is the plan for combining SHAP ranking with correlation clustering. It includes the math and the three-tier pruning rule.

#### `output/`
- **`Mutual_Information/`**: The MI drop list (`dropped_features.csv` / `.json`) and a run report with the per-block breakdown and runtime.
- **`Embedded/`**: The raw embedded drop list and report, the converted `updated_dropped_features.json`, and the baseline `model.txt`.
- **`SHAP/`**: The SHAP drop list (CSV and converted JSON), the report, and the baseline model.

#### `tsne/`
A separate PCA + t-SNE exploration of the family-filtered sample (all 2,568 features kept). Scripts are in `tsne/Code/`, with outputs in `Scaling Output/`, `PCA Output/` and `TSNE Output/`. See [`tsne/README.md`](tsne/README.md) for the workflow, key values, and folder details.

#### `runs/`
- **`no_drop/`**: Baseline model trained on all 2,568 features, with its ROC curve.
- **`MI01/`, `EMBED01/`, `SHAP01/`**: Models retrained with `thrember_lite` after removing the MI, embedded, and SHAP drop lists. Each folder has `model.txt` and `roc_test.pdf`.

---

## Pipeline

```
HuggingFace shards
   │  bernoulli_sample.py                      sample ~4% + dedup by md5
   ▼
PE_train_sampled.jsonl / PE_test_sampled.jsonl
   │  one selection script                     each vectorizes the JSONL itself
   │    mi_feature_selection.py
   │    train_custom_lgbm_fs.py            →  convert_embedded_to_mi_format.py
   │    shap_cluster_feature_selection.py  →  convert_shap_to_mi_format.py
   ▼
drop list JSON  (257 indices)
   │  vectorize_dataset.py                     sampled train/test + challenge → full-width .dat
   │  thrember_lite: build-spec → train
   ▼
runs/<EXP>/model.txt + spec.json
   │  eval_thrember_lite.py
   ▼
ROC AUC, PR AUC, TPR @ 1% FPR (test + challenge)
```

**Why features are dropped after vectorization:** a drop list refers to positions in the output vector, not to keys in the raw JSON. Many of those positions don't exist until extraction runs: hashed import/section buckets, pefile warning counters, and fixed-size blocks. So the full vector is built once, and each experiment just picks a different set of columns at load time. See `Documentation/Overview/process_overview.md` for details.

### Data folders

Scripts pick their input files by name, so check what's in the folder before running them:
- **`vectorize_dataset.py` / `run_thrember_lite.py`:** these include every `.jsonl` whose name contains `train`, `test`, or `challenge`. `bernoulli_sample.py` leaves the full weekly shards in `ember_data/`, so move those out first, or the whole dataset gets vectorized instead of the sample.
- **Embedded and SHAP scripts:** these need a folder with exactly one `*train*.jsonl` and one `*test*.jsonl`, and stop with an error if the weekly shards are also there. They write `X_train.dat` / `X_test.dat` into that folder, the same names `vectorize_dataset.py` uses, so a separate folder is safest (the HPC guides use `data/`).

The separate t-SNE pipeline is described in [`tsne/README.md`](tsne/README.md).

## Setup

```bash
git clone https://github.com/tgraber4/EMBER_SELECTION.git
cd EMBER_SELECTION
pip install -e .          # installs both thrember and thrember_lite
```

## Typical workflow

```bash
# 1. Sample the dataset
python custom_scripts/bernoulli_sample.py

# 2. Rank features (pick one or more). <data_dir> holds only the sampled train + test JSONL.
python "custom_scripts/Feature Selection/Mutual Informatin/mi_feature_selection.py"

python "custom_scripts/Feature Selection/Embedded/train_custom_lgbm_fs.py" <data_dir> --config-file examples/lgbm_config.json
python "custom_scripts/Feature Selection/Embedded/convert_embedded_to_mi_format.py"

python "custom_scripts/Feature Selection/SHAP/shap_cluster_feature_selection.py" --data-dir <data_dir> --config-file examples/lgbm_config.json
python "custom_scripts/Feature Selection/SHAP/convert_shap_to_mi_format.py"

# 3. Fetch the challenge set and vectorize train/test/challenge (see "Data folders" above)
python "custom_scripts/Helper Scripts/download_challenge.py"
python custom_scripts/vectorize_dataset.py

# 4. Build a spec and retrain on the reduced feature set
#    (or edit the paths in custom_scripts/run_thrember_lite.py, which vectorizes, builds the spec, and trains in one go)
python -m thrember_lite.cli build-spec --drop output/Mutual_Information/dropped_features.json --out runs/MI01/spec.json
python -m thrember_lite.cli train ember_data runs/MI01/spec.json runs/MI01/ --config examples/lgbm_config.json --seed 42

# 5. Evaluate (edit MODEL_DIR / DATA_DIR at the top of the script)
python custom_scripts/eval_thrember_lite.py
```

Most scripts set their paths as constants at the top of the file, so check those before running. For HPC runs, see the `*_HPC.md` guides in `custom_scripts/`.

## Citing EMBER2024

```
@inproceedings{joyce2025ember,
      title={EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers},
      author={Robert J. Joyce and Gideon Miller and Phil Roth and Richard Zak and Elliott Zaresky-Williams and Hyrum Anderson and Edward Raff and James Holt},
      year={2025},
      booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```
