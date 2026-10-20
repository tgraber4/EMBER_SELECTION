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

**Everything else was added on top of the original repo, and all of it lives in `custom/`.** Nothing outside `custom/` (other than this README, `.gitignore`, and the small edits listed above) was created by this project, so the upstream layout is unchanged.

---

## Project structure

```
EMBER2024/
├── src/
│   └── thrember/                      # [original] feature extraction + LightGBM library
├── examples/                          # [original] upstream train/eval scripts + configs
├── pyproject.toml, setup.cfg          # [original] package metadata
├── LICENSE                            # [original]
├── README.md                          # replaces the upstream README
└── custom/                            # [added] everything created for this project
    ├── thrember_lite/                 #   train/predict on a reduced feature set (+ tests/)
    ├── run_thrember_lite.py           #   end-to-end reduced-feature pipeline
    ├── eval_thrember_lite.py          #   evaluate a thrember_lite model
    ├── common/                        #   shared scripts
    │   ├── bernoulli_sample.py
    │   ├── vectorize_dataset.py
    │   └── helper_scripts/
    ├── feature_selection/             #   one self-contained folder per method
    │   ├── mutual_information/        #     code/ docs/ output/ runs/
    │   ├── embedded/                  #     code/ docs/ output/ runs/
    │   └── shap/                      #     code/ docs/ output/ runs/
    ├── tsne/                          #   PCA + t-SNE exploration of the family-filtered sample
    │   ├── Code/                      #     scripts (tsne_figures/ holds the plotting scripts)
    │   ├── Documentation/
    │   ├── Scaling Output/            #     vectorized + scaled matrices, fitted scalers (git-ignored)
    │   ├── PCA Output/                #     X_pca.npy, pca.joblib (git-ignored) + loadings CSVs
    │   └── TSNE Output/               #     X_tsne.npy, plots, tsne_figures/
    ├── docs/                          #   overview, example row, feature types, feature_index_map.json
    ├── runs/                          #   baseline (no_drop) and other ad-hoc runs
    └── archive/                       #   old drop lists, earlier ROC PDF
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

#### `custom/common/`
- **`bernoulli_sample.py`**: Downloads the PE shards (Win32, Win64, .NET) for train and test, then keeps each sample with a fixed probability based on a deterministic hash of its md5. Upstream stores every sample twice, so the script removes the duplicate (keeping the copy with Capa results) and writes `ember_data/PE_train_sampled.jsonl` and `PE_test_sampled.jsonl`.
- **`vectorize_dataset.py`**: Wraps `thrember.create_vectorized_features` to turn a folder of train/test/challenge JSONL files into the `.dat` memmap files that `thrember_lite` trains and evaluates on. It includes every `.jsonl` in the folder whose name contains `train`, `test`, or `challenge`, and does nothing if the `.dat` files already exist unless `OVERWRITE` is set.

#### `custom/` (top-level scripts)
These sit directly in `custom/` so that `thrember_lite` is importable when they run.
- **`run_thrember_lite.py`**: Runs the reduced-feature pipeline end to end: vectorize, build `spec.json` from a drop list, and train a binary LightGBM model, with an optional prediction on a single file. Paths and settings are edited at the top of `main()`; by default it reads the embedded drop list and writes to `custom/feature_selection/embedded/runs/EMBED01/`.
- **`eval_thrember_lite.py`**: The `thrember_lite` version of `examples/eval_lgbm.py`. It loads a model bundle, applies the same column slice to the test and challenge sets, and reports ROC AUC, PR AUC, and TPR at 1% FPR with an ROC PDF. `MODEL_DIR` defaults to `custom/feature_selection/embedded/runs/EMBED01/`.

#### `custom/common/helper_scripts/`
- **`build_feature_index_map.py`**: Generates `custom/docs/feature_index_map.json`, which maps each of the 2,568 vector positions to its block, position within the block, readable field name, and a hashed/not-hashed flag. It checks every block's size against `PEFeatureExtractor`, so the map can't drift out of sync.
- **`check_labels.py`**: Reads a sampled JSONL and confirms every row has a `label`. Prints benign, malware, and unlabeled counts.
- **`count_rows.py`**: Counts rows in one JSONL file or in every JSONL under `ember_data/`. Reads line by line, so memory use stays low on large shards.
- **`download_challenge.py`**: Downloads the EMBER2024 challenge set into `ember_data/`. `create_vectorized_features` and the evaluation scripts both need it.

#### `custom/feature_selection/mutual_information/code/`
- **`mi_feature_selection.py`**: Vectorizes up to 100k labeled rows from the sampled training file and scores each feature against the label with `mutual_info_classif`, treating the categorical columns as categorical. Writes the 257 lowest-scoring features to `custom/feature_selection/mutual_information/output/dropped_features.{csv,json}`.

#### `custom/feature_selection/embedded/code/`
- **`train_custom_lgbm.py`**: Trains a binary LightGBM classifier from a folder holding exactly one JSONL with `train` in its name and one with `test`. It vectorizes only those two files, writing the `.dat` files into the same folder, so it doesn't need the challenge set; the SHAP script reuses its helper functions.
- **`train_custom_lgbm_fs.py`**: Trains the same baseline model, then ranks features by LightGBM gain and split count. It drops exactly 10% (257): features with zero gain and zero splits first, then the lowest-gain ones. It saves a JSON report that includes the "information cost" (the share of total gain that was dropped).
- **`convert_embedded_to_mi_format.py`**: Converts the embedded report (`embedded_dropped_features.json`) into the ranked-list format the MI script produces, saved as `updated_dropped_features.json`. This step is required, because `build-spec` can't read the raw report.

#### `custom/feature_selection/shap/code/`
- **`shap_cluster_feature_selection.py`**: Trains a baseline LightGBM model, computes TreeSHAP values, ranks features by mean |SHAP|, and groups features whose SHAP values are highly correlated. It drops the bottom 10% in three tiers (unused → redundant → low importance) and writes `shap_dropped_features.csv` and the baseline model to `custom/feature_selection/shap/output/`.
- **`convert_shap_to_mi_format.py`**: Converts the SHAP drop-list CSV into the JSON format the MI script produces, saved as `custom/feature_selection/shap/output/shap_dropped_features.json`. This step is required, because `build-spec` only reads JSON.

#### `custom/thrember_lite/`
A small companion library that trains and predicts on a subset of columns without modifying `thrember`.
- **`__init__.py`**: Exports the public API: `FeatureSpec`, `read_vectorized_features`, `apply_spec`, `train_binary`, `ModelBundle`, `predict_file`, and `predict_batch`. It is the entry point for all `thrember_lite` usage.
- **`spec.py`**: `FeatureSpec` turns a drop list into the list of kept column indices and records where the categorical columns end up after slicing. It is saved as `spec.json` next to every model, so predictions always use the same columns as training.
- **`data.py`**: Loads `X_/y_<subset>.dat` and copies only the kept columns in 50k-row chunks, so the full 2,568-wide matrix is never held in RAM. `apply_spec` slices an array that is already in memory.
- **`train.py`**: `train_binary` follows the same steps as `thrember.train_model` (drop unlabeled rows, 90/10 stratified split, `lgb.train`), but passes the remapped categorical indices. It applies one seed to both the data split and all of LightGBM's random-number settings, so differences between runs come from the feature set rather than random noise.
- **`predict.py`**: `ModelBundle` saves and loads `model.txt` and `spec.json` together. `predict_file` and `predict_batch` extract the full vector, slice it with the spec, and score it.
- **`cli.py`**: Command-line interface with `build-spec`, `train`, and `predict` subcommands. Run it as `python -m thrember_lite.cli`.
- **`tests/`**: Unit tests for `FeatureSpec` and the chunked loader, plus a training smoke test on synthetic data. The smoke test also checks save/load round-trips and that runs are reproducible.

#### `custom/docs/`
- **`overview/process_overview.md`**: Walks through the pipeline from HuggingFace download → sampling → vectorization → training → evaluation. At each step it compares upstream `thrember` with the feature-dropping path.
- **`feature_index_map.json`**: Maps every vector index (0–2567) to its block, field name, and hashed flag, plus the index range of each block. The selection scripts use it to attach readable names to column numbers.
- **`feature_vector_membership.md`**: Lists which fields in a raw JSONL row feed the feature vector and which are metadata, labels, or tags. Useful when reading a raw row.
- **`example_row/`**: One sample row saved in several views (full row, feature fields only, non-feature fields only, top-level schema, extracted vector). `example_row_summary.md` explains each section of the row.
- **`feature_types/`**: Notes on the data type of each feature.
- **`thrember_lite/`**: `thrember_lite.md` (HPC run notes for the two top-level scripts), `thrember_lite_overview.md`, and `finish_thrember_lite_plan.md` (what's left to finish the library, in a suggested order).

#### `custom/feature_selection/<method>/`
Each method (`mutual_information/`, `embedded/`, `shap/`) has the same four folders:
- **`code/`**: the scripts described above.
- **`docs/`**: the plan, notes and HPC guide for that method. For MI: `mi_feature_selection_plan.md` and `mutual_info.md`. For embedded: `lightgbm_feature_selection_plan.md`, `embedded_process_audit.md`, `train_custom_lgbm.md` (why it bypasses `examples/train_lgbm.py` and how inputs are detected) and `train_custom_lgbm_fs_HPC.md`. For SHAP: `EMBER2024_Feature_Selection_Plan-v4.md` (SHAP ranking plus correlation clustering and the three-tier pruning rule) and `shap_cluster_feature_selection_HPC.md`. The `*_HPC.md` files are copy-paste guides for the cluster: files to upload, one-time setup, and the run command.
- **`output/`**: the drop list(s), run report, and baseline model where the method produces one. Embedded also holds the converted `updated_dropped_features.json`.
- **`runs/`**: the model retrained with `thrember_lite` after removing that method's drop list (`MI01/`, `EMBED01/`, `SHAP01/`), each with `model.txt`, `spec.json` and `roc_test.pdf`.

#### `custom/tsne/`
A separate PCA + t-SNE exploration of the family-filtered sample (all 2,568 features kept). Scripts are in `Code/`, with outputs in `Scaling Output/`, `PCA Output/` and `TSNE Output/`. See [`custom/tsne/README.md`](custom/tsne/README.md) for the workflow, key values, and folder details.

#### `custom/runs/` and `custom/archive/`
- **`runs/no_drop/`**: Baseline model trained on all 2,568 features, with its ROC curve. `runs/exp02/` holds an extra experiment model (`model.txt` only).
- **`archive/`**: Older drop lists (`old_data/`), and an earlier ROC PDF.

---

## Pipeline

```
HuggingFace shards
   │  bernoulli_sample.py                      sample train ~1.3% / test ~4.3% + dedup by md5
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
feature_selection/<method>/runs/<EXP>/model.txt + spec.json
   │  eval_thrember_lite.py
   ▼
ROC AUC, PR AUC, TPR @ 1% FPR (test + challenge)
```

**Why features are dropped after vectorization:** a drop list refers to positions in the output vector, not to keys in the raw JSON. Many of those positions don't exist until extraction runs: hashed import/section buckets, pefile warning counters, and fixed-size blocks. So the full vector is built once, and each experiment just picks a different set of columns at load time. See `custom/docs/overview/process_overview.md` for details.

### Data folders

Scripts pick their input files by name, so check what's in the folder before running them:
- **`vectorize_dataset.py` / `run_thrember_lite.py`:** these include every `.jsonl` whose name contains `train`, `test`, or `challenge`. `bernoulli_sample.py` leaves the full weekly shards in `ember_data/`, so move those out first, or the whole dataset gets vectorized instead of the sample.
- **Embedded and SHAP scripts:** these need a folder with exactly one `*train*.jsonl` and one `*test*.jsonl`, and stop with an error if the weekly shards are also there. They write `X_train.dat` / `X_test.dat` into that folder, the same names `vectorize_dataset.py` uses, so a separate folder is safest (the HPC guides use `data/`).

The separate t-SNE pipeline is described in [`custom/tsne/README.md`](custom/tsne/README.md).

## Setup

```bash
git clone https://github.com/tgraber4/EMBER_SELECTION.git
cd EMBER_SELECTION
pip install -e .          # installs thrember (src/)
```

`thrember_lite` lives in `custom/`, so it is not installed by the command above. Scripts placed directly in `custom/` can import it as is. For `python -m thrember_lite.cli ...` or anything run from elsewhere, put `custom/` on the path first (`export PYTHONPATH=custom`, or `$env:PYTHONPATH="custom"` in PowerShell).

## Typical workflow

Run everything from the repo root.

```bash
FS=custom/feature_selection

# 1. Sample the dataset
python custom/common/bernoulli_sample.py

# 2. Rank features (pick one or more). <data_dir> holds only the sampled train + test JSONL.
python $FS/mutual_information/code/mi_feature_selection.py

python $FS/embedded/code/train_custom_lgbm_fs.py <data_dir> --config-file examples/lgbm_config.json
python $FS/embedded/code/convert_embedded_to_mi_format.py

python $FS/shap/code/shap_cluster_feature_selection.py --data-dir <data_dir> --config-file examples/lgbm_config.json
python $FS/shap/code/convert_shap_to_mi_format.py

# 3. Fetch the challenge set and vectorize train/test/challenge (see "Data folders" above)
python custom/common/helper_scripts/download_challenge.py
python custom/common/vectorize_dataset.py

# 4. Build a spec and retrain on the reduced feature set
#    (or edit the paths in custom/run_thrember_lite.py, which vectorizes, builds the spec, and trains in one go)
export PYTHONPATH=custom
python -m thrember_lite.cli build-spec --drop $FS/mutual_information/output/dropped_features.json --out $FS/mutual_information/runs/MI01/spec.json
python -m thrember_lite.cli train ember_data $FS/mutual_information/runs/MI01/spec.json $FS/mutual_information/runs/MI01/ --config examples/lgbm_config.json --seed 42

# 5. Evaluate (edit MODEL_DIR / DATA_DIR at the top of the script)
python custom/eval_thrember_lite.py
```

Most scripts set their paths as constants at the top of the file, so check those before running. For HPC runs, see the `*_HPC.md` guides in each method's `docs/` folder.

## Citing EMBER2024

```
@inproceedings{joyce2025ember,
      title={EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers},
      author={Robert J. Joyce and Gideon Miller and Phil Roth and Richard Zak and Elliott Zaresky-Williams and Hyrum Anderson and Edward Raff and James Holt},
      year={2025},
      booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```
