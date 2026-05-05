# Process Overview

End-to-end walkthrough of how a sample moves from a HuggingFace shard to a scored prediction, and where feature-removal fits at each stage.

---

## 1. Downloading from HuggingFace

### What `thrember` ships
`src/thrember/download.py` exposes:
- `download_dataset(dir, split, file_type)` — picks one or more zip files from the `joyce8/EMBER2024` dataset repo, calls `hf_hub_download`, extracts them in place, and deletes the zip. `split` is `all|train|test|challenge`; `file_type` is `all|PE|Win32|Win64|Dot_Net|APK|ELF|PDF`.
- `download_models(dir)` — pulls every `.model` file from `joyce8/EMBER2024-benchmark-models`.

That is it. No filtering, no dedup, no sub-sampling, no awareness that the upstream dataset stores each PE sample twice.

### Why the custom path (`bernoulli_sample.py`) exists
The upstream PE shards contain **two rows per md5**: one with an empty `caps` field and one with Capa analysis populated. `thrember.download_dataset` happily writes both to disk, but downstream code that streams the file naively will see duplicate samples, and a random row pick can land on the caps-empty version even when the caps-rich version exists.

`bernoulli_sample.py` solves three problems at once:
1. **PE-only convenience** — downloads `Win32`, `Win64`, `Dot_Net` for both `train` and `test` in one call (thrember would need six calls or `file_type="PE"`).
2. **Bernoulli sub-sampling** — keeps each md5 with probability `p = 0.0347` using a deterministic BLAKE2 hash of `seed:md5`, so both duplicate rows of a sample agree on whether to keep it. Result: ~100k rows instead of ~3M.
3. **Dedup with caps preference** — when both copies of an md5 are sampled, the caps-populated row replaces the caps-empty row.

Output: `ember_data/PE_both_sampled.jsonl` (~2 GB).

The thrember downloader is still used **inside** `bernoulli_sample.py` (it calls `hf_hub_download` directly), so the custom script is a layer on top, not a replacement.

---

## 2. Building a sub-sample of rows

`bernoulli_sample.py` is the one-stop command. It:
1. Calls `download_pe_dataset` to fetch any missing `<file_type>_<split>.zip` and extract weekly shards into `ember_data/`.
2. Calls `iter_pe_files` to enumerate every `*_<file_type>_<split>.jsonl` shard.
3. Streams each shard line-by-line (so memory stays bounded) and applies `_sample_decision(md5, p, seed)` to decide keep/drop deterministically.
4. Maintains a `dict[md5 → (has_caps, raw_line)]`. If a duplicate md5 arrives with `has_caps=True` and the kept copy has `has_caps=False`, the row is replaced.
5. Writes everything in `kept` to `ember_data/PE_both_sampled.jsonl`.

Two helper scripts validate the result:
- `count_rows.py` — line-counts every JSONL under `ember_data/` (and the sampled file separately).
- `check_labels.py` — confirms every sampled row has a `label` and tallies benign/malware/unlabeled.

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
- **`PEFeatureExtractor.process_raw_features`** is fully reusable on a single dict. You don't have to go through `create_vectorized_features` — `drop_features.py` calls the extractor directly.
- **The `.dat` memmap layout is not customizable** in shape. Every cell is `float32`, every row is `extractor.dim` wide. `read_vectorized_features` reshapes by that constant.
- **The list of feature blocks is not parameterized.** `PEFeatureExtractor.__init__` builds the same 12 feature classes every time. To genuinely skip a block at extraction time you would have to modify `features.py` or fork it.

### Vectorizing with features removed (`drop_features.py`)

The current strategy is **extract all 2,568 features, then drop columns**, never "skip during extraction." Concretely:

1. `mi_feature_selection.py` ranks features by mutual information against the binary label and writes `dropped_features.{csv,json}` containing the bottom-N indices (default N=257, ~10%).
2. `drop_features.py` reads the drop list, builds a boolean keep-mask of length 2,568, runs `PEFeatureExtractor.process_raw_features` on every input row, slices `vec[keep_idx]`, and writes a JSONL with `{<passthrough metadata>, "vector": [reduced floats...]}`.

**What works under this design:**
- Any drop list, including dropping categorical indices, is mechanically valid.
- The original index map (`Documentation/feature_index_map.json`) still names every kept column — you just filter it by the keep-mask.
- Re-running with a different drop list is a fresh JSONL; the source data is untouched.
- `bernoulli_sample.py`, `mi_feature_selection.py`, and the index map all work on **raw** rows, so they are immune to feature-removal — drop selection happens after them in the pipeline, not before.

**What breaks (and why):**
- `thrember.create_vectorized_features` cannot consume the reduced JSONL — it expects the full nested dict, not a `vector` key.
- `thrember.read_vectorized_features` is hard-wired to `extractor.dim = 2568`. A reduced `.dat` would be reshaped wrong.
- `thrember.train_model` / `train_ovr_model` set `categorical_feature=[2, 3, 4, 5, 6, 701, 702]` literally. After dropping any column ≤ 702, those indices point at the wrong feature.
- `thrember.predict_sample` calls `extractor.feature_vector(file_data)` and feeds the full 2,568-wide result straight to `booster.predict` — it has no notion of a column slice.

So feature-removal **bypasses the entire thrember training entry-point** and needs its own thin layer. The design for that layer is in `Documentation/thrember_lite_plan.md` (`FeatureSpec` + `train_binary` + `ModelBundle`); `drop_features.py` is a precursor that produces reduced vectors but does not yet write `.dat` memmaps or train.

A practical bridge: write the reduced vectors as a memmap of shape `(N, dim_kept)` and keep `kept_indices` + the remapped `categorical_feature` list alongside the model so inference can re-apply the same slice.

### Why dropping must happen post-vectorization

The dropped-features list refers to **positions in the output vector**, not to keys in the raw JSON. There is no clean way to delete features from raw rows before vectorizing, for three structural reasons:

1. **Hashed buckets have no raw equivalent.** Entries like `imports.libraries_hashed[105]` are output bucket 105 of a `FeatureHasher` over import-library names. The raw JSON contains the original library *strings* — the bucket only exists after the hasher runs.
2. **Computed counters don't exist in raw form.** Entries like `pewarn:...` are per-template counts produced by scanning `pefilewarnings` during vectorization. The raw JSON has warning IDs, not the counters that get dropped.
3. **Block dimensions are hardcoded.** Each feature class has a fixed `dim` (e.g. `GeneralFileInfo.dim = 7`, `ByteHistogram.dim = 256`). Deleting a key from raw JSON does not shrink the output — the vectorizer either crashes or zero-fills, and the vector width stays 2,568.

So the only place the drop list maps cleanly is **after** vectorization, where each index points to exactly one output slot. Vectorize once on full raw data, then slice.

### What `drop_features.py` outputs

A new JSONL at `--out`, one line per input sample:

```json
{"md5": "...", "sha1": "...", "sha256": "...", "label": 0, "file_type": "Win32", "vector": [5785.0, 307.0, 174.0, ...]}
```

- Passthrough metadata copied from each input line (`md5`, `sha1`, `sha256`, `tlsh`, `label`, `file_type`, `family`, etc.).
- `"vector"`: a flat list of floats — the full 2,568-element vector with the dropped indices removed. Length = `2568 − len(drop_list)`.

Each line is self-contained for training: `(label, vector)`. The script does both vectorization and trimming in one pass — no pre-vectorized input is required.

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
This is where the design changes the most. The reduced JSONL emitted by `drop_features.py` has shape `(N, dim_kept)` per row. To train on it you need to:

1. **Stack into an array.** Either load all rows into memory (fine at ~100k rows) or write your own memmap of the reduced shape.
2. **Remap the categorical indices.** If `kept_indices = sorted indices kept after the drop`, the new categorical list is `[kept_indices.index(c) for c in [2,3,4,5,6,701,702] if c in kept_indices]`. Any categorical that got dropped must be removed from the list, and every surviving categorical's position shifts down by however many lower-indexed columns were dropped before it.
3. **Build the `lgb.Dataset` and call `lgb.train` directly** with the same params dict you would have given thrember. The training algorithm itself is column-count-agnostic — only the `categorical_feature` indices need fixing.
4. **Persist the kept-indices list** next to the booster so inference can replay the same slice (this is the `FeatureSpec` / `ModelBundle` idea).

Things to watch out for:
- If you drop a categorical column **and** forget to remove it from `categorical_feature`, LightGBM will silently treat a continuous column as categorical — quietly garbage results.
- If you drop a low index (e.g. column 5) **and** forget to shift, the indices `701, 702` become `700, 701` and you'll be flagging the wrong column as categorical.
- Stratified splitting still works — only `X.shape[1]` changed, not `y`.
- The original LightGBM JSON config doesn't reference column indices, so you can reuse `examples/lgbm_config.json` as-is.

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
- `categorical_remapped: list[int]` — already-translated indices into the reduced vector.

The `thrember_lite_plan.md` design captures this as a `ModelBundle` (`{model.txt, spec.json}`) saved together. Without that sidecar you cannot reproduce predictions because you don't know which 2,311 of the 2,568 features the booster actually trained on.

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
- `read_vectorized_features` cannot be used for the reduced data unless you wrote the memmap yourself with the new `dim_kept` width — the function reshapes by `extractor.dim`.
- For raw-bytes inference (`predict_sample` analogue), the path is: `extractor.feature_vector(file_bytes)` → 2,568-dim vector → `vec[kept_indices]` → `booster.predict([vec])`. Skipping the slice silently produces a shape mismatch (LightGBM will error).
- All of the metric calls (`roc_auc_score`, `precision_recall_curve`, `roc_curve`) are column-count-agnostic — they only see `y_pred` and `y_test`.

So evaluation under feature removal needs exactly one new piece of plumbing (apply the slice before `predict`) and zero changes to the metrics.

---

## Resolved design decisions

1. **Reduced-vector training is not implemented yet.** `drop_features.py` produces reduced JSONL but no script in the repo currently trains a LightGBM model from it. The `thrember_lite_plan.md` plan is the path forward; until that lands, "training with features removed" is design-only.
2. **Feature removal stays post-extraction for now.** `PEFeatureExtractor` always emits the full 2,568-dim vector and downstream scripts slice it via `kept_indices`. No block-level skipping inside `features.py`.
3. **Raw-bytes inference will use a forked `predict_sample`** that takes an explicit slice argument (e.g. `kept_indices` or a `FeatureSpec`) rather than a wrapper that reads a sidecar at call time. The fork extracts the full 2,568-dim vector via `PEFeatureExtractor`, applies the slice in-place, and calls `booster.predict` on the reduced array.
