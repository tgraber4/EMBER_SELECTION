# EMBER2024 Repo Overview

A working copy of the EMBER2024 PE-malware feature library (`thrember`) plus a layer of personal scripts and notes for sub-sampling the dataset, mapping the 2,568-dim feature vector, ranking features by mutual information, and dropping low-value columns. The upstream library sits in `src/thrember/`; everything in the repo root and `Documentation/` is local research work on top of it.

---

## Top-level layout

```
EMBER2024/
├── src/thrember/                  # The upstream feature/extraction library (installable package)
├── examples/                      # Upstream training/eval scripts + LightGBM configs
├── ember_data/                    # Raw EMBER2024 .jsonl shards + sub-sampled file (gitignored, ~100 GB)
├── Documentation/                 # Local notes, plans, feature-index map, example-row dumps
│   ├── Overview/                  # ← this file lives here
│   ├── Example Row/               # Pretty-printed JSON of one sample for reference
│   ├── feature_index_map.json     # 2,568 → {block, field, hashed} mapping
│   ├── feature_vector_membership.md
│   ├── mi_feature_selection_plan.md
│   ├── mutual_info.md
│   └── thrember_lite_plan.md
├── Test/                          # Local pytest tests
├── build/                         # setuptools build artifacts (regenerable)
├── .claude/, .vscode/             # Editor/agent configs
│
├── bernoulli_sample.py            # Build PE_both_sampled.jsonl (~100k row subsample)
├── build_feature_index_map.py     # Generate Documentation/feature_index_map.json
├── mi_feature_selection.py        # Score features via mutual information, write drop list
├── drop_features.py               # Vectorize JSONL and drop columns by index
├── check_labels.py                # Sanity-check label coverage of the sampled file
├── count_rows.py                  # Line-count every .jsonl under ember_data/
├── dropped_features.csv/.json     # Output of mi_feature_selection.py — bottom-MI features
├── overall_notes.txt              # Short personal cheat-sheet
│
├── README.md, LICENSE             # Upstream docs / license
├── pyproject.toml, setup.cfg      # Package metadata (`thrember`, version 0.1.0)
└── .git/, .venv/                  # VCS + virtualenv
```

---

## `src/thrember/` — the upstream library

Installable Python package (`thrember`, declared in `pyproject.toml` and `setup.cfg`). Imports the public API from three modules.

### `__init__.py`
Re-exports the public surface: feature classes (`PEFeatureExtractor`, `ByteHistogram`, `ByteEntropyHistogram`, `StringExtractor`, `HeaderFileInfo`, `SectionInfo`, `ImportsInfo`, `ExportsInfo`, `DataDirectories`, `RichHeader`, `AuthenticodeSignature`, `GeneralFileInfo`, `PEFormatWarnings`), the model helpers (`read_metadata`, `create_vectorized_features`, `optimize_model`, `predict_sample`, `raw_feature_iterator`, `read_vectorized_features`, `train_model`, `train_ovr_model`, `vectorize_subset`), and the dataset/model downloaders (`download_dataset`, `download_models`).

### `features.py`
The feature extractor. Defines a `FeatureType` base class and one subclass per feature block. Each subclass exposes `raw_features(bytez, pe)` (parses a PE into a JSON-able dict) and `process_raw_features(raw_obj)` (turns that dict into a fixed-length numeric vector). `PEFeatureExtractor` chains them in this order, producing a 2,568-dim vector:

| Block               | Dim   | What it captures                                          |
| ------------------- | ----- | --------------------------------------------------------- |
| `general`           | 7     | size, entropy, is_pe flag, first 4 bytes                  |
| `histogram`         | 256   | normalized byte-value histogram                           |
| `byteentropy`       | 256   | 16×16 byte/entropy 2D histogram (flattened)               |
| `strings`           | 124   | string stats + printable distribution + regex counts      |
| `header`            | 73    | COFF + Optional + DOS header fields (categorical at idx 701, 702) |
| `section`           | 224   | section counts/entropies + hashed name/size/entropy/char buckets + overlay |
| `imports`           | 1,282 | hashed library names + hashed lib::function pairs         |
| `exports`           | 129   | hashed exported-symbol names                              |
| `datadirectories`   | 34    | size + virtual address for each of 16 PE data dirs + relocs flags |
| `richheader`        | 33    | hashed Rich-header pair buckets                           |
| `authenticode`      | 8     | code-signing chain stats                                  |
| `pefilewarnings`    | ~142  | one-hot `pefile` parser warnings (template list in `pefile_warnings.txt`) |

Indices `[2, 3, 4, 5, 6, 701, 702]` are categorical (start_bytes, is_pe, machine type, subsystem) — treated specially by both LightGBM and the MI scorer.

### `model.py`
Wraps the data pipeline and LightGBM:
- `raw_feature_iterator`, `gather_feature_paths` — stream JSONL by subset / file_type / week.
- `vectorize_subset`, `create_vectorized_features` — multiprocessing fan-out that writes `X_train.dat` / `y_train.dat` (and test/challenge equivalents) as `np.memmap` files.
- `read_vectorized_features` — load those `.dat` files back into shape `(N, dim)`.
- `read_metadata` — pull the metadata columns (hashes, dates, labels, tags) into a Polars DataFrame.
- `train_model` / `train_ovr_model` — train a binary or multilabel-OvR LightGBM booster, with the categorical indices wired up.
- `optimize_model` — `GridSearchCV` over a small param grid using `TimeSeriesSplit` and `roc_auc_score(max_fpr=5e-3)`.
- `predict_sample(model, file_data)` — extract features from raw bytes and predict.

### `download.py`
- `download_dataset(dir, split, file_type)` — pulls from HuggingFace `joyce8/EMBER2024` and unzips. Splits: `all|train|test|challenge`. File types: `all|PE|Win32|Win64|Dot_Net|APK|ELF|PDF`.
- `download_models(dir)` — fetches the benchmark `.model` files from `joyce8/EMBER2024-benchmark-models`.

### `pefile_warnings.txt`
Newline-delimited list of `pefile` warning templates. Used by `PEFormatWarnings` to map each warning to a stable index in the vector.

---

## `examples/` — upstream usage

- `train_lgbm.py` — CLI wrapping `thrember.train_model`. Reads a LightGBM JSON config and saves the trained booster.
- `eval_lgbm.py` — loads a saved booster, scores the test + challenge sets, prints ROC AUC / PR AUC, draws a log-scale ROC curve, reports TPR @ 1% FPR.
- `lgbm_config.json` — binary classifier config (gbdt, 500 iters, lr 0.1, 64 leaves, is_unbalance).
- `lgbm_config_family.json` — multiclass family classifier config.
- `lgbm_config_tag.json` — multilabel tag classifier config.
- `ember2024-notebook.ipynb` — upstream walkthrough notebook.

---

## `ember_data/` — raw + sub-sampled data

Where `bernoulli_sample.py` and `download_dataset` deposit shards. Layout:

- `YYYY-MM-DD_YYYY-MM-DD_<Win32|Win64|Dot_Net>_<train|test>.jsonl` — one weekly shard per file-type, per split. Each line is a single PE sample's full feature dict (~100 GB total on disk for the PE family).
- `PE_both_sampled.jsonl` — local Bernoulli sub-sample (~100k rows, ~2 GB) produced by `bernoulli_sample.py`. Dedup'd by md5 with the caps-populated copy preferred over the empty one.
- `PE_both_sampled copy.jsonl` — backup of the same file.

This directory is large and not committed.

---

## `Documentation/` — local notes & artifacts

- `Overview/repo_overview.md` — this file.
- `Example Row/`
  - `example_row.json` — one full row from the sampled file, pretty-printed.
  - `example_row_feature_vector.json` — the same row, restricted to the fields that actually feed the 2,568-dim vector.
  - `example_row_non_feature_vector.json` — the leftover metadata/label fields.
  - `example_row_toplevel.json` — schema-only view (top-level keys with values stubbed) for at-a-glance structure.
  - `example_row_summary.md` — prose walkthrough of every section in a row.
  - `first_row_vector.json` — first row's vector after extraction, for spot-checks.
- `feature_index_map.json` — generated by `build_feature_index_map.py`. Maps every index `0..2567` to `{block, block_index, field, hashed}` plus `block_ranges` for each of the 12 blocks. Consumed by `mi_feature_selection.py` and useful any time a column index needs a human label.
- `feature_vector_membership.md` — reference list of which top-level row attributes feed the feature vector vs. which are metadata/labels/threat-intel tags.
- `mi_feature_selection_plan.md` — design doc for `mi_feature_selection.py`: parameters, pipeline, output format, caveats (hashed features have no meaningful names, MI misses feature interactions).
- `mutual_info.md` — quick notes on categorical-continuous MI, pros/cons, and the next-step idea of dropping one of any pair of mutually-redundant high-MI features.
- `thrember_lite_plan.md` — design for a still-unimplemented companion library `thrember_lite` that wraps `thrember` with a `FeatureSpec` so you can train/predict on a stripped column set without forking the upstream package.

---

## Root-level scripts

### `bernoulli_sample.py`
Builds `ember_data/PE_both_sampled.jsonl`. Downloads the PE shards (Win32/Win64/Dot_Net, train+test) from HuggingFace if missing, then streams every line and keeps it with probability `p = 0.0347` (deterministic per-md5 BLAKE2 hash with seed 42). Dedupes by md5 across the duplicate caps-empty/caps-populated rows the upstream dataset ships. Output is ~100k unique samples.

### `build_feature_index_map.py`
Emits `ember_data/feature_index_map.json` (the path comment is stale — the consumed copy lives in `Documentation/`). Walks each feature block, generates a human-readable label for every index, asserts the per-block dim against `PEFeatureExtractor`, and tags hashed buckets via substring match. Self-checks that the cumulative index equals `extractor.dim`.

### `mi_feature_selection.py`
Streams `PE_both_sampled.jsonl`, vectorizes up to 100k labeled rows, and runs `sklearn.feature_selection.mutual_info_classif` against the binary label using a categorical mask for indices `[2,3,4,5,6,701,702]`. Sorts ascending by MI, takes the bottom 257 (~10%), looks each one up in `feature_index_map.json`, and writes:
- a console table,
- a per-block tally,
- `dropped_features.csv`,
- `dropped_features.json`.

Runtime ~6–15 minutes.

### `custom_scripts/train_custom_lgbm_fs.py`
Trains a binary LightGBM classifier on a custom train/test JSONL pair, then runs the embedded LightGBM feature-selection plan from `lightgbm_feature_selection_plan.md`. Companion to `mi_feature_selection.py` — same end product (a drop list), different signal (model-internal Gain vs. mutual information).

**Pipeline:** auto-locate `*train*.jsonl` / `*test*.jsonl` in `data_dir` → `vectorize_subset` writes `X_<split>.dat` / `y_<split>.dat` → load into RAM, 10% stratified val split (`random_state=0`) → `lgb.train` with early stopping (default 50, configurable via `--early-stopping 0` to disable), timed via `perf_counter` → score test set (accuracy @ 0.5) → extract per-feature `gain` and `split` at `best_iteration` → tiered selection (absolute zeros first, top up from the bottom of the Gain ranking, **hard-capped at exactly `N_drop = round(total * drop_fraction)`**) → write report.

**Inputs:** positional `data_dir`, `model_path`; `--config-file` (required, `objective` must be `binary`); `--drop-fraction` (default 0.10, strict ceiling); `--dropped-out` (default `<model_path>.dropped_features.json`); `FEATURE_INDEX_MAP_PATH` constant at the top of the file (defaults to `Documentation/feature_index_map.json`) translates vector indices to `"<block>[<block_index>].<field>"` names, falling back to `Column_<i>` if missing or malformed.

**Outputs:**
- `<data_dir>/X_<train|test>.dat`, `y_<train|test>.dat` memmaps,
- `<model_path>` LightGBM text model (saved at `best_iteration`),
- JSON sidecar with `total_features`, `n_drop`, zero-gain/zero-split/absolute-zero/signal-dropped counts, `information_cost_pct` (Σ dropped Gain ÷ Σ total Gain × 100), total/dropped Gain, train and selection times, test accuracy, `best_iteration`, dropped indices + names, absolute-zero indices, signal-dropped indices, and a per-feature `feature_table` (index, name, gain, split, three zero flags),
- Console: vectorize progress + LightGBM log + `=== Feature Selection Report ===` summary table.

**Hard cap behavior:** if the absolute-zero set alone is larger than `N_drop`, the script truncates to the first `N_drop` zeros (deterministic by feature index) and leaves the rest in the active set — never drops more than the configured fraction.

### `drop_features.py`
Reads a drop list (CSV or JSON with an `index` column) and a raw EMBER JSONL, then writes a reduced-vector JSONL with structure `{sha256, label, ..., vector: [...]}`. Pass-through metadata keys are preserved verbatim. CLI: `--dropped`, `--in`, `--out`. Validates that drop indices are in `[0, dim)` before running.

**Pipeline:** raw JSONL → `PEFeatureExtractor.process_raw_features` (full 2,568-dim vector) → slice out dropped indices → write JSONL with passthrough metadata + a `"vector"` field of length `2568 − len(drop_list)`. Vectorization and trimming happen in one pass; no pre-vectorized input is required.

**Why post-vectorization (not raw-JSON edit):** the drop list refers to vector positions, not raw-JSON keys. Hashed buckets (e.g. `imports.libraries_hashed[105]`) and computed counters (e.g. `pewarn:...`) have no raw equivalent — they only exist after the hasher / warning scanner runs. And every block has a hardcoded `dim`, so deleting raw keys does not shrink the output vector. Slicing after extraction is the only place the drop list maps cleanly.

**Training-readiness:** the output carries the data LightGBM needs (features + labels, dimensionally reduced) but is **not** drop-in for `thrember.train_model`. Two gaps:
1. Format mismatch — `train_model` reads `.dat` memmaps via `read_vectorized_features`, not JSONL. To use the reduced JSONL, stack lines into a NumPy array (or write your own memmap of shape `(N, dim_kept)`) and call `lgb.train` directly.
2. Categorical-feature remap — thrember hardcodes `categorical_feature=[2, 3, 4, 5, 6, 701, 702]` for the full 2,568-wide vector. After dropping columns, those indices point at the wrong features; translate them to positions in the reduced vector before passing to `lgb.train`, or LightGBM will silently treat continuous columns as categorical.

### `check_labels.py`
Sanity script. Walks `PE_both_sampled.jsonl`, counts label values, flags any rows missing the `label` key, and prints a benign/malware/unlabeled tally.

### `count_rows.py`
Streams every `.jsonl` under `ember_data/` and prints the per-file row count, plus a separate count for `PE_both_sampled.jsonl`. Memory-bounded (line-by-line).

### `dropped_features.csv` / `dropped_features.json`
The most recent output of `mi_feature_selection.py`. Each row carries `rank, index, block, field, hashed, mi_score`. Consumed by `drop_features.py`.

### `overall_notes.txt`
Personal cheat sheet pointing at which script generates which artifact.

---

## Tests, build, and editor metadata

- `Test/test_get_file_type.py` — local pytest covering file-type detection.
- `build/` — setuptools `bdist`/`lib` output. Regenerated; safe to delete.
- `src/thrember.egg-info/` — packaging metadata generated when installing in editable mode.
- `.venv/` — local virtualenv.
- `.vscode/settings.json` — editor settings.
- `.claude/settings.json`, `.claude/settings.local.json` — Claude Code harness config.
- `__pycache__/` — Python bytecode cache.

---

## Typical workflows

**Reproduce the 100k subsample**
```
python bernoulli_sample.py
```

**Generate the index map** (only after `features.py` changes)
```
python build_feature_index_map.py
```

**Find the weakest 257 features**
```
python mi_feature_selection.py
```

**Apply the drop list to a JSONL**
```
python drop_features.py --dropped dropped_features.json --in ember_data/PE_both_sampled.jsonl --out reduced.jsonl
```

**Train / evaluate the upstream benchmark** (once `create_vectorized_features` has produced `.dat` memmaps)
```
python examples/train_lgbm.py ember_data model.txt --config-file examples/lgbm_config.json
python examples/eval_lgbm.py  ember_data model.txt
```
