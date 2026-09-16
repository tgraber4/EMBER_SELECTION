# `thrember_lite` Overview

`thrember_lite` (`src/thrember_lite/`) is a small companion library to `thrember`. It lets you **train and run a LightGBM malware classifier on a subset of the 2,568 EMBER features**, without changing `thrember` itself.

You give it a list of feature columns to drop. It trains on the remaining columns and saves the model together with a record of which columns it used, so predictions later use exactly the same columns.

The only part of `thrember` it reuses is `PEFeatureExtractor`, which extracts the features. Everything else is custom code.

---

## 1. Why it exists

This repo's goal is feature selection: rank the 2,568 features with three methods (Mutual Information, embedded LightGBM gain, SHAP), drop the weakest 10% (257 features), retrain, and compare against the full-feature baseline. The upstream `thrember` library can't do the "retrain on fewer features" step, because it assumes all 2,568 columns everywhere:

| Where | What `thrember` does | Why that breaks with dropped features |
|---|---|---|
| Loading data: `read_vectorized_features` (`model.py:249`) | Reshapes the `.dat` file by `PEFeatureExtractor().dim` and returns every column. It also copies the full-width matrix into RAM (`model.py:265`). | There's no way to ask for a subset of columns. |
| Training: `train_model` (`model.py:347`) | Passes `categorical_feature=[2, 3, 4, 5, 6, 701, 702]` as a hardcoded list. | Once any column at or below 702 is dropped, the later columns shift left, and those indices point at the wrong features. |
| Training: `train_model` | Calls `train_test_split` and `lgb.train` with no seed. | Two runs give different models, so a small AUC change from dropping features can't be told apart from random noise. |
| Prediction: `predict_sample` (`model.py:409`) | Extracts the full 2,568-value vector and passes it straight to the model. | A model trained on 2,311 columns gets 2,568 values and fails. |
| Saving: `model.save_model` | Saves only the LightGBM model file. | LightGBM only knows column *positions*, not names. The file alone doesn't say which 2,311 of the 2,568 features the model was trained on. |

`thrember_lite` fixes each of these while keeping the rest of the training steps identical to `thrember`, so that "baseline vs. reduced" comparisons measure the dropped features, not a change in training code.

### Why features are dropped after vectorization, not before

A drop list refers to **positions in the output vector**, not keys in the raw JSONL rows. Many of those positions don't exist until extraction runs:
- **Hashed buckets:** `imports.libraries_hashed[105]` is bucket 105 of a hasher over library names. The raw row only has the names.
- **Computed counters:** `pewarn:...` values are counts built while scanning `pefilewarnings`.
- **Fixed-size blocks:** every feature block has a hardcoded width. Removing a key from the raw JSON doesn't shrink the vector.

So the approach is: **vectorize once at full width, then pick columns at load time.** Every experiment reuses the same `X_*.dat` / `y_*.dat` files; only `spec.json` and the model change. See `Overview/process_overview.md` §3 for the longer explanation.

### Design goals
- Don't modify anything in `src/thrember/`. Reuse `PEFeatureExtractor` and the `.dat` files as they are.
- Mirror `thrember.train_model` step for step, changing only what's needed.
- Make runs reproducible with a single seed.
- Always ship the column list with the model.

---

## 2. How it works

```
raw JSONL ──(thrember.create_vectorized_features)──► X_train.dat / X_test.dat   (N × 2,568, created once)
                                                              │
dropped_features.json ──(FeatureSpec.from_drop_columns)──► FeatureSpec
                                                              │
                                    read_vectorized_features: copy only the kept columns
                                                              ▼
                                    train_binary ──► lgb.Booster (N × 2,311)
                                                              │
                                    ModelBundle.save ──► runs/<EXP>/model.txt + spec.json
                                                              │
new PE file ──► PEFeatureExtractor (2,568 values) ──► slice with spec ──► booster.predict ──► score in [0, 1]
```

---

## 3. The modules

| File | Main API | What it does |
|---|---|---|
| `spec.py` | `FeatureSpec`, `DEFAULT_CATEGORICAL` | Turns a drop list into the sorted list of kept columns and works out where the categorical columns end up. Saves and loads `spec.json`. |
| `data.py` | `read_vectorized_features`, `apply_spec` | Loads `X_/y_<subset>.dat` and copies only the kept columns, in chunks. `apply_spec` slices an array that's already in memory. |
| `train.py` | `train_binary` | Same steps as `thrember.train_model`, but uses the remapped categorical indices and a seed. |
| `predict.py` | `ModelBundle`, `predict_file`, `predict_batch` | Saves/loads the model and spec together. Scores PE files by extracting the full vector and slicing it. |
| `cli.py` | `build-spec`, `train`, `predict` | Command-line wrapper, run as `python -m thrember_lite.cli`. |
| `tests/` | | Unit tests for the spec and the loader, plus a training smoke test on synthetic data. |

Everything in the "Main API" column is importable from the top level: `from thrember_lite import FeatureSpec, train_binary, ...`.

---

## 4. `FeatureSpec` and `spec.json`

`FeatureSpec` is the single source of truth for which columns a model uses.

### Two kinds of JSON file

| | Layer A: the drop list | Layer B: `spec.json` |
|---|---|---|
| Written by | Feature-selection scripts (e.g. `output/Mutual_Information/dropped_features.json`) | `thrember_lite`, saved next to the model |
| Contents | One entry per dropped feature: `rank`, `index`, `block`, `field`, `hashed`, score | `original_dim`, `kept_indices`, `original_categorical`, `new_categorical`, `block_ranges`, `source` |
| Used for | Humans reading the selection results. The library only reads `index`. | Everything at train and predict time. |

`FeatureSpec.from_drop_columns(path)` reads Layer A and builds Layer B. It also accepts a JSON file holding a plain list of ints, or a Python list of ints. `FeatureSpec.from_json(path)` reads Layer B back. Note that the embedded and SHAP selection scripts need their convert scripts run first (`convert_embedded_to_mi_format.py`, `convert_shap_to_mi_format.py`) to produce a JSON drop list this can read.

### `spec.json` fields

| Field | Meaning |
|---|---|
| `original_dim` | Width of the full vector (2,568). Checked against `PEFeatureExtractor().dim` when a model is loaded. |
| `kept_indices` | Sorted positions in the full vector that the model uses. The only field needed to slice data. Loading fails if it isn't sorted, since that would reorder columns. |
| `original_categorical` | Categorical positions in the full vector. Defaults to `[2, 3, 4, 5, 6, 701, 702]`: `is_pe`, `start_bytes[0..3]`, `coff.machine`, and `optional.subsystem`. |
| `new_categorical` | Where the surviving categorical columns sit after slicing. Computed from the two fields above and saved for easy checking. |
| `block_ranges` | Each feature block's `[start, end)` range in the full vector, so tools can label features without the index map. |
| `source` | Where the spec came from: `drop_count`, `drop_file`, and an optional `note`. |

`original_dim` and `block_ranges` come from `PEFeatureExtractor` by default, or from `Documentation/feature_index_map.json` if you pass `index_map_path` (useful where importing `thrember` is slow). The index map stores inclusive end values; the spec converts them to exclusive ends.

### Categorical remapping: real examples from `runs/`

When columns are dropped, the columns after them shift left. The three experiments in this repo show why `thrember`'s hardcoded list can't be reused:

| Run | `new_categorical` | What happened |
|---|---|---|
| `SHAP01` | `[2, 3, 4, 5, 6, 701, 702]` | Nothing at or below index 702 was dropped, so the list happens to be unchanged. |
| `MI01` | `[2, 3, 684, 685]` | `start_bytes[1..3]` (4, 5, 6) were dropped. 17 columns before 701 were dropped, so 701 → 684 and 702 → 685. |
| `EMBED01` | `[682, 683]` | All five `general` categoricals (2–6) were dropped, and 701 → 682, 702 → 683. |

For `EMBED01`, the hardcoded list would have told LightGBM to treat five ordinary numeric columns as categorical, and pointed 701/702 at two unrelated header fields.

---

## 5. Loading data

```python
X, y = read_vectorized_features(data_dir, "train", spec, block=50_000, in_memory=False)
```

- Same name and behavior as `thrember.read_vectorized_features`, plus the `spec` argument.
- Opens `X_<subset>.dat` as a lazy memmap and copies only the kept columns into the output, 50,000 rows at a time. The full-width matrix is never held in RAM; peak memory is about `N × len(kept_indices) × 4` bytes plus one chunk.
- `in_memory=True` does the slice in one step. Simpler, and fine for small subsets.
- **Does not remove unlabeled rows (`y == -1`).** Like `thrember`, the trainer does that. Evaluation scripts get every row and filter as needed.
- Raises an error if `y` is multilabel (longer than the number of rows) or doesn't match `X`'s row count, which usually means a wrong `original_dim` or a corrupt file.

`apply_spec(X, spec)` does the same slice on an array already in memory, and checks that its last dimension is `original_dim` first.

---

## 6. Training

```python
booster = train_binary(data_dir, spec, params, val_size=0.1, seed=42)
```

The body follows `thrember.train_model` line for line:
1. Load the train set with `read_vectorized_features` (sliced).
2. Remove rows with `y == -1`.
3. Split 90/10 into train and validation, stratified by label.
4. Build `lgb.Dataset`s and call `lgb.train(params, train_set, valid_sets=val_set)`.

What's different:
- **Categorical indices:** uses `spec.new_categorical` instead of the hardcoded list.
- **Seed:** when `seed` is set, it's used for the validation split (`random_state`) and for LightGBM's three random settings (`seed`, `bagging_seed`, `feature_fraction_seed`). These go on a copy of `params`, and a message is printed if an existing value is overwritten. `seed=None` behaves like `thrember` (not reproducible).
- **Binary only:** raises an error unless there are exactly two label classes. There's no multiclass or one-vs-rest option.
- **Checks** that the loaded data width matches `len(spec.kept_indices)`.

It returns a plain `lgb.Booster`, the same type `thrember.train_model` returns, so the two can be swapped in comparison scripts. Saving is a separate step.

**Why the seed matters:** the AUC difference from dropping 10% of features is often about the same size as run-to-run random variation. Using the same seed for the baseline and every reduced run means the same validation rows and the same LightGBM random choices, so the difference comes from the features. Use one seed (this repo uses 42) across a whole study.

The LightGBM config (`examples/lgbm_config.json`) doesn't refer to column positions, so it works unchanged with any spec.

---

## 7. Saving and predicting

```python
ModelBundle.save(booster, spec, "runs/MI01/")     # writes model.txt + spec.json

bundle = ModelBundle.load("runs/MI01/")
score  = predict_file(bundle, "suspicious.exe")    # path or raw bytes → float in [0, 1]
scores = predict_batch(bundle, [path1, path2])     # one extractor, one predict call
```

- **`ModelBundle.save`** creates the folder and writes both files. The two always go together; the model is unusable without the spec.
- **`ModelBundle.load`** reads both and checks that `PEFeatureExtractor().dim` still equals `spec.original_dim`. If `thrember`'s feature layout has changed since training, it fails instead of silently giving wrong scores.
- **`predict_file` / `predict_batch`** extract the full 2,568-value vector, keep only `kept_indices`, and call `booster.predict`. This is the `thrember_lite` version of `thrember.predict_sample`.

Dropping features does **not** make prediction faster: the extractor still computes all 2,568 values, and the slice happens afterward.

---

## 8. Usage

### Command line

```bash
# Build a spec from a drop list
python -m thrember_lite.cli build-spec \
    --drop output/Mutual_Information/dropped_features.json \
    --out runs/MI01/spec.json \
    [--index-map Documentation/feature_index_map.json] [--source-note "MI top-257"]

# Train and save model.txt + spec.json into runs/MI01/
python -m thrember_lite.cli train ember_data runs/MI01/spec.json runs/MI01/ \
    --config examples/lgbm_config.json --seed 42

# Score one file; prints "<path>\t<score>"
python -m thrember_lite.cli predict runs/MI01/ suspicious.exe
```

### Python

```python
import json
from thrember_lite import FeatureSpec, train_binary, ModelBundle, predict_file

spec = FeatureSpec.from_drop_columns("output/Mutual_Information/dropped_features.json")
params = json.load(open("examples/lgbm_config.json"))

booster = train_binary("ember_data", spec, params, seed=42)
ModelBundle.save(booster, spec, "runs/MI01/")

bundle = ModelBundle.load("runs/MI01/")
print(predict_file(bundle, "suspicious.exe"))
```

### Scripts in this repo that use it
- **`custom_scripts/run_thrember_lite.py`:** runs the whole flow (vectorize → build spec → train → optional predict). Paths are set at the top of `main()`.
- **`custom_scripts/eval_thrember_lite.py`:** loads a bundle, reads the test (and optionally challenge) set through `read_vectorized_features` so the slice is applied, and reports ROC AUC, PR AUC, and TPR at 1% FPR with an ROC PDF. It's the `thrember_lite` version of `examples/eval_lgbm.py`.
- **`custom_scripts/thrember_lite.md`:** notes for running both scripts on the HPC.
- **`runs/MI01/`, `runs/EMBED01/`, `runs/SHAP01/`:** models trained with `thrember_lite`, each keeping 2,311 of 2,568 features. `runs/no_drop/` is the full-feature baseline.

---

## 9. Limits and gotchas

- **Binary classification only.** Family (multiclass) and tag (multilabel) models are out of scope; a `train_family` module was left as a possible future addition.
- **Column-level only.** There's no "drop this whole block" option. To drop a block, write a drop list containing its indices (`block_ranges` gives the range).
- **Doesn't judge the drop list.** `from_drop_columns` checks that indices are in range and removes duplicates, but not whether the selection is any good.
- **`block` / `in_memory` can't be changed from `train_binary` or the CLI** yet; they always use the defaults.
- **Reproducibility needs more than the seed for exact matches.** Seeded runs give identical models only with `num_threads=1` and `force_row_wise=True`, because multithreaded LightGBM can differ slightly in floating point. Normal runs with many threads are close but not bit-identical.
- **Model files on Windows:** with `core.autocrlf=true`, Git can convert `model.txt` to Windows line endings, and LightGBM then fails with `Model format error, expect a tree here`. The repo's `.gitattributes` (`*model.txt -text`) prevents this; existing clones need to re-checkout their model files once.
- **Signed PE files** needed a fix in `thrember` (`features.py:955`, wrapping `signed_data.certificates` in `list()`) for extraction to work with `signify` 0.9.x. Without it, `predict_file` crashes on any file with an Authenticode signature.

---

## 10. Tests and further reading

Run the tests (each file has its own runner, so `pytest` isn't required):

```bash
PYTHONPATH=src python -m thrember_lite.tests.test_spec
PYTHONPATH=src python -m thrember_lite.tests.test_data
PYTHONPATH=src python -m thrember_lite.tests.test_train_smoke
```

- **`finish_thrember_lite_plan.md`:** remaining tests to write (real `ModelBundle.load`, `predict_file` on a real file, parity with `thrember`, the CLI) and related fixes.
- **`Overview/process_overview.md`:** the full pipeline from download to evaluation, comparing `thrember` and `thrember_lite` at each stage.
- **Original design plan:** deleted from the working tree; recover it with `git show 0d5ec8d:Documentation/thrember_lite_plan.md`. It has the full reasoning behind all 12 design decisions and memory-sizing estimates.
