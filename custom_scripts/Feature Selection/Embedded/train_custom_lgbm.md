# `train_custom_lgbm.py` — Design Decisions

Summary of the main design choices behind `custom_scripts/train_custom_lgbm.py`. The script trains a binary LightGBM malware classifier on a custom train/test JSONL pair using thrember's `PEFeatureExtractor`.

## Scope

- **Binary classification only** (benign = 0, malicious = 1). Multiclass and multilabel paths were intentionally removed.
- **Two JSONL inputs in one folder.** No support for the EMBER `challenge` subset, weekly slices, or per-filetype filtering.
- **Drop-in for the standard thrember feature schema.** Each JSONL record must contain raw thrember PE features plus an integer `label` field.

## Why not `examples/train_lgbm.py`?

`train_lgbm.py` calls `thrember.train_model`, which expects pre-vectorized `X_train.dat` / `y_train.dat` memmaps in `data_dir`. The standard way to produce those is `thrember.create_vectorized_features`, which **also vectorizes a `challenge` subset** and raises if no `challenge*.jsonl` file is present (`src/thrember/model.py:62`, `:240`). Workarounds (e.g., a dummy challenge file) are possible but ugly.

The custom script calls `thrember.model.vectorize_subset` directly on just the train and test files, bypassing the challenge step entirely.

## Input layout

- A single folder (`data_dir`) contains both JSONLs.
- Auto-detected by substring: one filename contains `train`, the other contains `test`. This mirrors `thrember.gather_feature_paths` (`src/thrember/model.py:45`).
- `find_jsonl` raises if zero or multiple files match a substring — clean filenames (`train.jsonl`, `test.jsonl`) avoid ambiguity.
- Vectorized `.dat` outputs are written into the same folder.

## Vectorization

- Reuses `thrember.model.vectorize_subset` rather than reimplementing it. That function already handles the multiprocessing pool, memmap allocation, and per-row dispatch into `PEFeatureExtractor.process_raw_features`.
- `nrows` is computed by line-counting each JSONL once before vectorization — required because `vectorize_subset` pre-allocates the memmaps with that exact shape.
- `label_map={}` is passed because binary labels (`int` 0/1) hit the `isinstance(label, int)` branch in `vectorize` (`src/thrember/model.py:120`), which doesn't consult the map.

## Reading vectorized data

- `read_vectorized` materializes the full feature matrix into RAM via `np.array(X).reshape(-1, ndim)`.
  - The reshape doubles as a sanity check: a stale `.dat` whose size isn't a multiple of `extractor.dim * 4` raises here.
  - Adds an explicit row-count parity check between `X` and `y` to catch corrupted or partially written outputs from a previous failed run.
- Trade-off: simpler code at the cost of RAM headroom (~9 GB for 1M rows). For larger datasets, the memmap could be passed directly into `lgb.Dataset` instead.

## Training

- **10% stratified validation split**, matching `thrember.train_model` (`src/thrember/model.py:364`).
- `random_state=0` on the split for reproducibility.
- **Early stopping is enabled by default** (`lgb.early_stopping(50)`), unlike `thrember.train_model`, which passes a `valid_sets` it never acts on. Configurable via `--early-stopping`; pass `0` to disable.
- **Categorical features hardcoded to `[2, 3, 4, 5, 6, 701, 702]`** — same indices thrember itself uses. Tied to the current `PEFeatureExtractor` layout.
- LightGBM determinism comes from the config file's `seed`, `bagging_seed`, and `feature_fraction_seed`.

## Validation guards

The script fails fast on conditions that would silently produce a wrong model:

- **Config `objective` must be `binary`.** Rejects multiclass configs up front rather than letting them produce confusing errors deeper in.
- **Train and test labels must be exactly `{0, 1}`.** Catches mislabeled data, leaked `-1` "drop" sentinels, or accidental string labels.
- **Empty JSONL** raises explicitly rather than producing a zero-byte memmap that explodes downstream.
- **`data_dir` must exist** and be a directory; config file must exist.

## `best_iteration` handling

`model.best_iteration` can be `0`, `-1`, or a positive integer depending on whether early stopping fired. The script normalizes:

```python
bi = model.best_iteration
best_iter = int(bi) if bi is not None and bi > 0 else None
```

`None` is then passed to both `save_model` and `predict`, which both interpret it as "use best if recorded, else all trees" — so the model and evaluation stay consistent.

## Evaluation

- After training, the test memmaps are loaded and scored.
- Binary predictions are thresholded at `0.5` against the class-1 probability returned by `predict`.
- Reports test-set accuracy. Other metrics (AUC, precision/recall, FPR-bounded AUC) were left out to keep the script focused; the saved model can be loaded into `examples/eval_lgbm.py` for richer evaluation.

## Things deliberately not handled

- **Multiclass / multilabel.** Removed entirely. Use `thrember.train_model` / `thrember.train_ovr_model` for those.
- **`-1` drop labels.** Per the user's stated assumption ("no dropped vectors"), unlabeled rows aren't filtered.
- **Hyperparameter search.** The script trusts the config file. Use `thrember.optimize_model` for grid search.
- **Streaming / out-of-core training.** Full matrix is loaded into RAM.
- **Stale-file detection beyond size mismatch.** Re-running into a folder with leftover `.dat` files of the right size will silently train on the new vectors (which is correct) — but won't catch a case where a swapped-in JSONL happens to have the same row count as the previous one.
