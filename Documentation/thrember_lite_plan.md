# Plan: `thrember_lite` — a thin training/inference layer for stripped EMBER vectors

A small companion library that wraps `thrember` so you can drop arbitrary feature columns (or whole groups), train LightGBM on the reduced matrix, and run inference end-to-end without forking `thrember` itself.

## Goals
- Accept a "feature selection spec" (column-level or group-level).
- Load EMBER `.dat` memmaps and slice them to the selected columns.
- Train LightGBM with correctly-remapped categorical indices.
- Persist enough metadata (kept columns, original dim, categorical map) alongside the model so inference is reproducible.
- Provide a `predict()` that mirrors `thrember.predict_sample` but applies the same column slice.

## Non-goals
- Re-implementing feature extraction. We reuse `PEFeatureExtractor` from `thrember` unchanged.
- Hyperparameter search / OvR / multiclass. Binary classification only in v1; leave hooks for later.
- Modifying anything in `src/thrember/`.

## Layout
```
thrember_lite/
├── __init__.py
├── spec.py          # FeatureSpec: load/save, column<->group resolution
├── data.py          # load_vectors(), apply_spec()
├── train.py         # train_binary()
├── predict.py       # ModelBundle, predict_file(), predict_batch()
├── cli.py           # `thrember-lite train` / `predict` / `build-spec`
└── tests/
    ├── test_spec.py
    ├── test_data.py
    └── test_train_smoke.py
```

## Core data structures

**`FeatureSpec`** (`spec.py`) — the single source of truth for "what columns are kept."
- Fields: `original_dim: int`, `kept_indices: list[int]` (sorted, into the original vector), `original_categorical: list[int]` (default `[2,3,4,5,6,701,702]`), `group_layout: dict[str, tuple[int,int]]` (group name → `[start, end)` offsets in the original vector, captured from `PEFeatureExtractor().features`).
- Derived: `new_categorical` = `[kept_indices.index(c) for c in original_categorical if c in kept_indices]`.
- Constructors: `from_drop_columns(drop_idx)`, `from_keep_groups(group_names)`, `from_json(path)`.
- Serialization: `to_json(path)` / `from_json(path)` — must round-trip exactly so the model is reproducible.
- This subsumes what `build_feature_index_map.py` and `drop_features.py` are already doing — port their logic in.

## Module responsibilities

**`data.py`**
- `load_raw(data_dir, subset)` — thin wrapper around `np.memmap` + reshape using `original_dim` from the spec (not `extractor.dim`, in case you ever stored a vector built with a different `features_file`).
- `apply_spec(X, spec)` → `X[:, spec.kept_indices]` as a contiguous `np.ndarray` (LightGBM wants C-contiguous).
- `load_labeled(data_dir, subset, spec)` → `(X, y)` with `y == -1` rows dropped (matches `train_model` behavior at `model.py:360-361`).
- Memory note: full Win32 train is large. Default to chunked column extraction (read memmap in row blocks, slice cols, append) so we don't double-allocate. Provide `in_memory=True` for small subsets.

**`train.py`**
- `train_binary(data_dir, spec, params, val_size=0.1, seed=None) -> lgb.Booster`
- Mirrors `model.py:347-370` but: (a) no `extractor.dim` dependency, (b) uses `spec.new_categorical`, (c) returns the booster.
- Validates: `len(np.unique(y)) == 2`, `X.shape[1] == len(spec.kept_indices)`.
- No silent param mutation — pass `params` through verbatim.

**`predict.py`**
- `ModelBundle` = `{booster, spec}`. `save(dir)` writes `model.txt` + `spec.json`; `load(dir)` reads both.
- `predict_file(bundle, path_or_bytes)` — instantiate `PEFeatureExtractor()` once (reuse `thrember`), extract full-width vector, slice via `bundle.spec.kept_indices`, call `booster.predict`.
- `predict_batch(bundle, paths)` — vectorized version, single extractor instance.
- This is the piece that fills the `predict_sample` gap noted earlier (`model.py:409-416` always extracts full width).

**`cli.py`** (argparse, no extra deps)
- `thrember-lite build-spec --drop dropped_features.json --out spec.json`
  Bridges your existing `dropped_features.json` into a `FeatureSpec`.
- `thrember-lite train <data_dir> <spec.json> <out_dir> --config lgbm_config.json`
- `thrember-lite predict <model_dir> <file>`

## Reusing your existing scripts
Your tree already has `build_feature_index_map.py`, `drop_features.py`, `mi_feature_selection.py`, `bernoulli_sample.py`. Plan: keep those as **selection-strategy scripts** that produce a `dropped_features.json` (or equivalent), and have `FeatureSpec.from_drop_columns` consume that. Don't merge the selection logic into the library — keep selection (research-y, iterates a lot) separate from training (stable).

## Validation strategy
- **Unit**: spec round-trip; `apply_spec` shape correctness; categorical remap when dropping a categorical (e.g. drop col 3 → expect `[2, 3→removed, 4→3, 5→4, 6→5, 701→699, 702→700]`).
- **Integration smoke**: tiny synthetic `.dat` files (50 rows × `original_dim`), train 10 iterations, assert booster predicts within `[0,1]`.
- **Parity check**: with `kept_indices = list(range(original_dim))` (no drops), `train_binary` should produce a booster numerically equivalent to `thrember.train_model` given the same seed. Document if not exact and why (val split RNG).

## Open questions to resolve before coding
1. What does `dropped_features.json` look like today? The spec loader needs to match it exactly.
2. Do you want spec-level support for the `features_file` group toggle (line 1078 of `features.py`), or is per-column dropping enough? Per-column is strictly more general; group toggling is a UX shortcut.
3. Are you OK with v1 being binary-only, or do you need family/multiclass at the same time?
4. Storage budget — is full in-memory `X[:, kept]` acceptable on your machine, or do we need the chunked path on day one?

## Implementation order
1. `spec.py` + tests (spec is the contract; everything depends on it).
2. `data.py` + tests (in-memory path first; chunked later if needed).
3. `train.py` + smoke test.
4. `predict.py` + `ModelBundle` round-trip test.
5. `cli.py` last — it's just argparse glue.
6. Wire `dropped_features.json` → `FeatureSpec` and run end-to-end on a small subset (PDF or ELF) to validate before touching Win32.
