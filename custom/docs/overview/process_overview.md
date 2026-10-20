# Process Overview

End-to-end walkthrough of how a sample moves from a HuggingFace shard to a scored prediction, and where feature-removal fits at each stage.

---

## 1. Downloading from HuggingFace

### What `thrember` ships
`src/thrember/download.py` exposes:
- `download_dataset(dir, split, file_type)` — picks one or more zip files from the `joyce8/EMBER2024` dataset repo, calls `hf_hub_download`, extracts them in place, and deletes the zip. `split` is `all|train|test|challenge`; `file_type` is `all|PE|Win32|Win64|Dot_Net|APK|ELF|PDF`.
- `download_models(dir)` — pulls every `.model` file from `joyce8/EMBER2024-benchmark-models`.

That is it. No filtering, no dedup, no sub-sampling, no awareness that the upstream dataset stores each PE sample twice.

### Why the custom path (`custom/common/bernoulli_sample.py`) exists
The upstream PE shards contain **two rows per md5**: one with an empty `caps` field and one with Capa analysis populated. `thrember.download_dataset` happily writes both to disk, but downstream code that streams the file naively will see duplicate samples, and a random row pick can land on the caps-empty version even when the caps-rich version exists.

`bernoulli_sample.py` solves three problems at once:
1. **PE-only convenience** — downloads `Win32`, `Win64`, `Dot_Net` for both `train` and `test` in one call (thrember would need six calls or `file_type="PE"`).
2. **Bernoulli sub-sampling, per split** — keeps each md5 with probability `p = 0.012856` for train (about 30k unique samples) and `p = 0.0427` for test, using a deterministic BLAKE2 hash of `seed:split:md5` (seed 42), so both duplicate rows of a sample agree on whether to keep it. The two splits are sampled and written separately, so the official temporal train/test boundary is preserved.
3. **Dedup with caps preference** — when both copies of an md5 are sampled, the caps-populated row replaces the caps-empty row.

Output: `ember_data/PE_train_sampled.jsonl` (~600 MB) and `ember_data/PE_test_sampled.jsonl` (~415 MB).

The thrember downloader is still used **inside** `bernoulli_sample.py` (it calls `hf_hub_download` directly), so the custom script is a layer on top, not a replacement.

---

## 2. Building a sub-sample of rows

`bernoulli_sample.py` is the one-stop command. It:
1. Calls `download_pe_dataset` to fetch any missing `<file_type>_<split>.zip` and extract weekly shards into `ember_data/`.
2. Calls `iter_pe_files_by_split` to enumerate every `*_<file_type>_<split>.jsonl` shard, grouped by split.
3. Streams each shard line-by-line (so memory stays bounded) and applies `_sample_decision(md5, split, p, seed)` to decide keep/drop deterministically.
4. Maintains a `dict[md5 → (has_caps, raw_line)]`. If a duplicate md5 arrives with `has_caps=True` and the kept copy has `has_caps=False`, the row is replaced.
5. Writes everything in `kept` to `ember_data/PE_train_sampled.jsonl` or `PE_test_sampled.jsonl`, one file per split.

Two helper scripts validate the result:
- `custom/common/helper_scripts/count_rows.py` — line-counts every JSONL under `ember_data/` (or a single file).
- `custom/common/helper_scripts/check_labels.py` — confirms every row in `PE_train_sampled.jsonl` has a `label` and tallies benign/malware/unlabeled.

Important: the sampled file keeps **every original field verbatim** — all 2,568 feature inputs and the nested label dict. No vectorization happens here.

---

## 3. Vectorization

### The thrember "all-in-one" path
`thrember.create_vectorized_features(data_dir, label_type)` does everything internally:
- Globs `*_train.jsonl`, `*_test.jsonl`, `*_challenge.jsonl` in `data_dir`.
- Counts rows per subset.
- Spins up a multiprocessing pool, calls `PEFeatureExtractor.process_raw_features` on each row, and writes a memmap'd `X_<subset>.dat` of shape `(N, 2568)` plus a `y_<subset>.dat`.
- Builds a label map for non-binary tasks (drops classes with fewer than `class_min` instances).

After this, `read_vectorized_features(data_dir, subset)` reshapes the memmap back to `(N, 2568)`.

This path is fixed-width: it always emits 2,568 columns, hard-codes the categorical indices, and writes `.dat` files keyed off `extractor.dim`.

### Customization — what is and isn't possible
- **`PEFeatureExtractor.process_raw_features`** is fully reusable on a single dict. You don't have to go through `create_vectorized_features` — `thrember_lite.predict_file` calls the extractor directly on a single sample.
- **The `.dat` memmap layout is not customizable** in shape. Every cell is `float32`, every row is `extractor.dim` wide. `read_vectorized_features` reshapes by that constant.
- **The list of feature blocks is not parameterized.** `PEFeatureExtractor.__init__` builds the same 12 feature classes every time. To genuinely skip a block at extraction time you would have to modify `features.py` or fork it.

### Vectorizing with features removed (`thrember_lite`)

The current strategy is **extract all 2,568 features into the canonical `.dat`, then slice columns at training time**, never "skip during extraction." Concretely:

1. A selection script (`mi_feature_selection.py`, `train_custom_lgbm_fs.py` + `convert_embedded_to_mi_format.py`, or `shap_cluster_feature_selection.py` + `convert_shap_to_mi_format.py`) ranks features and writes a drop list as JSON with the indices to drop (default 10%, 257 features).
2. `thrember_lite.FeatureSpec.from_drop_columns` reads the drop list and produces a `spec.json` holding `kept_indices`, `original_dim`, and the remapped `new_categorical` positions.
3. `thrember_lite.read_vectorized_features` opens the canonical `X_<subset>.dat`, slices columns to `kept_indices` during a chunked load (so the full-width matrix is never materialized in RAM), and hands `(X, y)` to `train_binary`.

**What works under this design:**
- Any drop list, including dropping categorical indices, is mechanically valid — `FeatureSpec.new_categorical` remaps surviving categoricals to their post-slice positions automatically.
- The original index map (`custom/docs/feature_index_map.json`) still names every kept column.
- Re-running with a different drop list reuses the same `.dat` files — only `spec.json` and the model change.
- `bernoulli_sample.py` and the index map work on **raw** rows, and the selection scripts vectorize the sampled JSONL themselves; the actual column dropping happens afterwards, in `thrember_lite`.

**What breaks (and why) for the upstream `thrember` training path:**
- `thrember.train_model` / `train_ovr_model` set `categorical_feature=[2, 3, 4, 5, 6, 701, 702]` literally. After dropping any column ≤ 702, those indices point at the wrong feature.
- `thrember.predict_sample` calls `extractor.feature_vector(file_data)` and feeds the full 2,568-wide result straight to `booster.predict` — it has no notion of a column slice.

`thrember_lite` solves both: `train_binary` uses `spec.new_categorical` instead of the hardcoded list, and `predict_file` slices the extracted vector through `spec.kept_indices` before predicting. The model and the spec ship together as a `ModelBundle` so inference always replays the same slice it trained on.

### Why dropping must happen post-vectorization

The dropped-features list refers to **positions in the output vector**, not to keys in the raw JSON. There is no clean way to delete features from raw rows before vectorizing, for three structural reasons:

1. **Hashed buckets have no raw equivalent.** Entries like `imports.libraries_hashed[105]` are output bucket 105 of a `FeatureHasher` over import-library names. The raw JSON contains the original library *strings* — the bucket only exists after the hasher runs.
2. **Computed counters don't exist in raw form.** Entries like `pewarn:...` are per-template counts produced by scanning `pefilewarnings` during vectorization. The raw JSON has warning IDs, not the counters that get dropped.
3. **Block dimensions are hardcoded.** Each feature class has a fixed `dim` (e.g. `GeneralFileInfo.dim = 7`, `ByteHistogram.dim = 256`). Deleting a key from raw JSON does not shrink the output — the vectorizer either crashes or zero-fills, and the vector width stays 2,568.

So the only place the drop list maps cleanly is **after** vectorization, where each index points to exactly one output slot. Vectorize once on full raw data, then slice.

### What `thrember_lite` writes to disk

After `train_binary` + `ModelBundle.save(booster, spec, out_dir)` the run directory contains:

```
out_dir/
├── model.txt    # LightGBM text dump (full booster, all trees)
└── spec.json    # original_dim, kept_indices, new_categorical, block_ranges, source
```

The two files travel together: the model alone is unusable because LightGBM only knows column *positions*, not names. `spec.json` is what tells inference which 2,311 of the 2,568 features the booster actually trained on, and where the categoricals ended up after the slice.

---

## 4. Training a model

### The thrember path (full 2,568-dim vector)
`examples/train_lgbm.py`:
1. Loads a JSON config (e.g. `examples/lgbm_config.json` for binary).
2. Calls `thrember.train_model(data_dir, params)` which:
   - `read_vectorized_features(data_dir, "train")` → `(X, y)` from the `.dat` memmaps.
   - Drops rows with `y == -1`.
   - Stratified `train_test_split(test_size=0.1)` for a validation set.
   - Builds `lgb.Dataset` for train and val, **passing `categorical_feature=[2,3,4,5,6,701,702]`** verbatim.
   - Calls `lgb.train(params, train_set, valid_sets=val_set)` for binary, or wraps it with `objective="multiclass"` for multiclass.
3. `model.save_model(out_path, num_iteration=model.best_iteration)`.

Multilabel tasks go through `train_ovr_model`, which trains one binary booster per label.

### Training with features removed
`thrember_lite.train_binary(data_dir, spec, params, seed=...)` is the entry point. The four things it does that thrember doesn't:

1. **Slices columns at load time.** `read_vectorized_features` opens the canonical `X_train.dat` as a lazy memmap and copies only the kept columns into a pre-allocated output array (chunked at 50k rows by default). The full-width matrix is never materialized.
2. **Remaps the categorical indices.** `spec.new_categorical = [kept_indices.index(c) for c in [2,3,4,5,6,701,702] if c in kept_indices]` is computed at spec-build time and cached in `spec.json`. `train_binary` passes it to `lgb.Dataset(categorical_feature=...)` instead of the hardcoded list.
3. **Pins the seed across both RNGs.** `random_state=seed` for the val split *and* `seed`/`bagging_seed`/`feature_fraction_seed` for LightGBM (on a copy of `params`, with a printed warning if they were already set). This is what makes ablation deltas attributable to the feature set rather than RNG noise.
4. **Returns `lgb.Booster` directly**, matching `thrember.train_model`. `ModelBundle.save(booster, spec, out_dir)` is a separate, explicit step that writes `model.txt` + `spec.json`.

Things to watch out for:
- The drop list must come from a reproducible source (`mi_feature_selection.py`, SHAP-cluster, etc.) — `FeatureSpec.from_drop_columns` enforces sorted, in-range, deduped indices but doesn't validate the *quality* of the selection.
- For ablation studies, hold `--seed` constant across configs. AUC differences from feature drops are typically the same magnitude as RNG noise.
- The original LightGBM JSON config doesn't reference column indices, so you can reuse `examples/lgbm_config.json` as-is — `train_binary` only injects the seed knobs.

### Tracking what changed
You don't gain new training logic from feature removal — you lose work, because the column count drops. What you gain is the obligation to keep `kept_indices` (and the remapped categorical list) bundled with the model.

---

## 5. Model output format

### File format
`thrember.train_model` returns a `lgb.Booster`. `examples/train_lgbm.py` saves it with `model.save_model(path, num_iteration=model.best_iteration)`. LightGBM serializes the booster as a **plain-text dump** — a deterministic, human-readable file describing the trees, leaf values, feature names, and metadata. The extension is by convention only:
- The benchmark-model repo on HuggingFace uses `.model`.
- The repo's `train_lgbm.py` accepts any path the user passes (`.txt`, `.model`, `.lgbm`, etc.).

### What to ship alongside the booster (with feature removal)
A bare `.model` file is enough for the full-width thrember flow because the column count is implicit (2,568) and the categorical indices are baked into training. With feature removal, the model alone is **not enough**:
- `kept_indices: list[int]` — sorted indices into the original 2,568-dim vector.
- `original_dim: int` — defensively, in case the upstream vector grows.
- `new_categorical: list[int]` — already-translated indices into the reduced vector (the spec also keeps `original_categorical` and `block_ranges`).

`thrember_lite` captures this as a `ModelBundle` (`{model.txt, spec.json}`) saved together; see `custom/docs/thrember_lite/thrember_lite_overview.md`. Without that sidecar you cannot reproduce predictions because you don't know which 2,311 of the 2,568 features the booster actually trained on.

---

## 6. Testing a model

### `examples/eval_lgbm.py` (full-width path)
1. Loads the booster: `lgb.Booster(model_file=args.model_path)`.
2. `read_vectorized_features(data_dir, "test")` → `X_test, y_test` (2,568 wide).
3. `y_pred = model.predict(X_test)`.
4. Computes:
   - `roc_auc_score(y_test, y_pred)`,
   - `precision_recall_curve` → `auc(recall, precision)` for PR AUC,
   - The full ROC curve, plotted on a log-x axis,
   - **TPR at FPR = 1%** by finding the closest threshold.
5. Saves `Classifier_ROC_AUC.pdf`.
6. Repeats on the **challenge set**: concatenates challenge samples with the benign rows from the test set (challenge has only malicious samples), then re-scores the same way. This exercises the model on out-of-distribution malware while keeping a realistic benign baseline.

The headline metrics for EMBER-style work are usually ROC AUC and TPR @ very-low-FPR, since malware classifiers operate at FPRs in the `1e-4`–`1e-2` range.

### Testing with features removed
The test path mirrors training and inherits the same constraints:
- The test vectors must be in the **same reduced shape** as training. Either pre-write a reduced test memmap with the same `kept_indices`, or load the full memmap and slice columns at evaluation time.
- Upstream `read_vectorized_features` cannot be used for the reduced data unless you wrote the memmap yourself with the new `dim_kept` width — it reshapes by `extractor.dim`. `thrember_lite.read_vectorized_features(data_dir, subset, spec)` avoids this by slicing while it loads, and `custom/eval_thrember_lite.py` uses it.
- For raw-bytes inference (`predict_sample` analogue), the path is: `extractor.feature_vector(file_bytes)` → 2,568-dim vector → `vec[kept_indices]` → `booster.predict([vec])`. Skipping the slice silently produces a shape mismatch (LightGBM will error).
- All of the metric calls (`roc_auc_score`, `precision_recall_curve`, `roc_curve`) are column-count-agnostic — they only see `y_pred` and `y_test`.

So evaluation under feature removal needs exactly one new piece of plumbing (apply the slice before `predict`) and zero changes to the metrics.

---

## Resolved design decisions

1. **Reduced-vector training is implemented as `thrember_lite`.** The package at `custom/thrember_lite/` consumes a `dropped_features.json`, slices columns at load time off the canonical `.dat` files, trains a binary LightGBM booster, and saves a `ModelBundle` (`model.txt` + `spec.json`) for reproducible inference. See `custom/docs/thrember_lite/thrember_lite_overview.md` for the design and the end-to-end commands.
2. **Feature removal stays post-extraction for now.** `PEFeatureExtractor` always emits the full 2,568-dim vector and downstream scripts slice it via `kept_indices`. No block-level skipping inside `features.py`.
3. **Raw-bytes inference is `thrember_lite.predict_file` / `predict_batch`**, which take a `ModelBundle` (model plus `FeatureSpec`) rather than reading a sidecar at call time. They extract the full 2,568-dim vector via `PEFeatureExtractor`, apply the slice, and call `booster.predict` on the reduced array.
