# Plan: `thrember_lite` — a thin training/inference layer for stripped EMBER vectors

A small companion library that wraps `thrember` so you can drop arbitrary feature columns (via a drop list), train LightGBM on the reduced matrix, and run inference end-to-end without forking `thrember` itself.

## Goals
- Accept a "feature selection spec" (column-level; group-toggle is out of scope per Decision #2).
- Load EMBER `.dat` memmaps and slice them to the selected columns.
- Train LightGBM with correctly-remapped categorical indices.
- Persist enough metadata (kept columns, original dim, categorical map) alongside the model so inference is reproducible.
- Provide a `predict_file()` that mirrors `thrember.predict_sample` but applies the same column slice.

## Non-goals
- Re-implementing feature extraction. We reuse `PEFeatureExtractor` from `thrember` unchanged.
- Hyperparameter search / OvR / multiclass. Binary classification only in v1; leave hooks for later.
- Modifying anything in `src/thrember/`.

## Layout

Lives at `src/thrember_lite/`, sibling to `src/thrember/`, so it's `import thrember_lite` once the project is installed (`pip install -e .`) and the CLI runs as `python -m thrember_lite.cli ...`.

```
src/thrember_lite/
├── __init__.py
├── spec.py          # FeatureSpec: load/save, column slicing
├── data.py          # read_vectorized_features(), apply_spec()  -- mirrors thrember.model
├── train.py         # train_binary()                            -- mirrors thrember.train_model
├── predict.py       # ModelBundle.save/load, predict_file(), predict_batch()
├── cli.py           # `python -m thrember_lite.cli train|predict|build-spec`
└── tests/
    ├── test_spec.py
    ├── test_data.py
    └── test_train_smoke.py
```

Naming choice: `data.py`'s loader reuses thrember's name (`read_vectorized_features`) so `train_binary`'s body reads as a near line-for-line copy of `thrember.train_model`. The lite changes are localized to (a) chunked + sliced load inside `read_vectorized_features`, (b) `spec.new_categorical` instead of the hardcoded categorical list, (c) seed plumbing, (d) deletion of the multiclass branch.

## Core data structures

**`FeatureSpec`** (`spec.py`) — the single source of truth for "what columns are kept."
- Fields: `original_dim: int`, `kept_indices: list[int]` (sorted, into the original vector), `original_categorical: list[int]` (default `[2,3,4,5,6,701,702]`), `block_ranges: dict[str, tuple[int,int]]` (block name → `[start, end)` offsets in the original vector).
- Derived: `new_categorical` = `[kept_indices.index(c) for c in original_categorical if c in kept_indices]`.
- Constructors: `from_drop_columns(drop_idx_or_path)`, `from_json(path)`. (No `from_keep_groups` in v1 — the CLI only takes a `dropped_features.json`; programmatic group-toggling can be added later if needed.)
- `block_ranges` source: derive in-process from `PEFeatureExtractor().features` by default (no file dependency, matches the running `thrember`). Allow `from_drop_columns(..., index_map_path=...)` to override for offline builds (loads `feature_index_map.json`'s `block_ranges` field).
- Serialization: `to_json(path)` / `from_json(path)` — must round-trip exactly so the model is reproducible.
- This subsumes the column-slicing logic that previously required a separate one-shot script.

### Two-layer JSON contract

Selection metadata and the runtime spec live in different files with different jobs:

**Layer A — selection input** (unchanged from today's `dropped_features.json`). Rich, human-readable, emitted by selection scripts (`mi_feature_selection.py`, `bernoulli_sample.py`, etc.):
```json
[
  {"rank": 1, "index": 2560, "block": "pefilewarnings",
   "field": "pewarn:...", "hashed": false, "mi_score": 0.0},
  ...
]
```
Only `index` is load-bearing for the library; the rest is provenance for humans.

**Layer B — `spec.json` saved next to the model**. Minimal and self-contained so inference doesn't need the original drop file:
```json
{
  "original_dim": 2568,
  "kept_indices": [0, 1, 2, 5, 6, 7, ...],
  "original_categorical": [2, 3, 4, 5, 6, 701, 702],
  "new_categorical": [2, 3, 4, 699, 700],
  "block_ranges": {"general": [0, 6], "histogram": [7, 262], ...},
  "source": {"drop_file": "dropped_features.json", "drop_count": 1234, "git_sha": "..."}
}
```
- `kept_indices` is the only field `apply_spec` and `predict` actually need at runtime.
- `original_dim` lets inference assert the extractor still produces the expected width.
- `block_ranges` lets downstream tools (SHAP plots, ablation tables) label features without re-running `build_feature_index_map.py`.
- `new_categorical` is derived but cached in JSON anyway — cheap, and makes "did the categorical remap drift?" debuggable from the file alone.
- Per-feature names/scores stay in Layer A and `feature_index_map.json`; they do **not** belong in `spec.json`.

`FeatureSpec.from_drop_columns(path_to_layer_a)` reads Layer A and computes Layer B. `FeatureSpec.from_json(spec.json)` reads Layer B directly. Layer B round-trips exactly.

## Module responsibilities

**`data.py`**
- `read_vectorized_features(data_dir, subset, spec, *, block=50_000, in_memory=False) -> tuple[np.ndarray, np.ndarray]` — same name and same semantics as `thrember.model.read_vectorized_features`, with two additions: takes a `spec` (so it slices columns during the load) and uses chunked materialization (so it never makes the full-width RAM copy that `model.py:265` does).
  - **Does not filter `y == -1`.** Mirrors thrember exactly: the loader returns raw rows and the trainer (`train_binary`) does the filter at `X = X[y != -1]` / `y = y[y != -1]`. Callers that want unfiltered data (test/challenge eval) get it for free.
  - Default path is chunked:
    ```python
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r").reshape(-1, original_dim)  # lazy
    y    = np.array(np.memmap(y_path, dtype=np.int32, mode="r"))
    if y.shape[0] > X_mm.shape[0]:
        raise ValueError("Encountered y with invalid shape. Use train_family() instead.")
    out = np.empty((X_mm.shape[0], len(kept_indices)), dtype=np.float32)
    for r0 in range(0, X_mm.shape[0], block):
        r1 = min(r0 + block, X_mm.shape[0])
        out[r0:r1] = X_mm[r0:r1, :][:, kept_indices]   # block-sized materialization
    return out, y
    ```
    Peak working set ≈ `len(kept_indices) × N × 4` bytes plus one transient `block × original_dim × 4` window — never the full-width copy thrember does.
  - `in_memory=True` bypasses chunking for small subsets where simplicity beats peak-RAM savings. Same `(X, y)` shape either way.
  - `block` is configurable per-call. 50 k is fine on 16 GB+ machines; lower it (e.g. 10 k) on tight RAM.
  - 1D-y assertion is loud and uses thrember's error wording (with the typo fixed): `"Encountered y with invalid shape. Use train_family() instead."` — same template as `model.py:355-356`.
- `apply_spec(X, spec)` → `X[:, spec.kept_indices]` as a contiguous `np.ndarray` (LightGBM wants C-contiguous). Used only on already-in-memory arrays (small subsets, predict path).

**Sizing reference (float32, one row = 10.27 KB at full width).** Two memory peaks matter:

1. **End of `read_vectorized_features`:** `out` array of shape `(N, len(kept))` is allocated. Peak so far ≈ `N × len(kept) × 4` B + transient `block × original_dim × 4` B window.
2. **Start of `train_binary` filter** (`X = X[y != -1]`): numpy fancy indexing allocates a new array. For a moment both the loader's `out` and the filtered copy coexist before GC reclaims `out`. Peak ≈ `2 × N_kept × len(kept) × 4` B (treating `N ≈ N_kept` for back-of-envelope).

Concrete numbers:

| Workload | Loader peak | Filter peak (~2× output) | thrember (full width) |
|---|---|---|---|
| 100 k rows, keep all 2568 | 980 MiB | ~1.9 GiB | ~3.3 GiB |
| 100 k rows, keep 1000 | 381 MiB | ~720 MiB | ~3.3 GiB |
| 6 M Win32, keep 1000 | ~22 GiB | ~42 GiB | ~60 GiB |

The trainer-side filter (matching thrember's `model.py:360-361` structure) costs a transient ~2× peak compared to filtering inside the chunked load. Acceptable for 100 k experiments — cheap; comfortable on 16 GB. At Win32 scale the filter peak becomes the binding constraint; if it bites, the optimization is to add an `optimize_memory=True` kwarg to `read_vectorized_features` that filters during the chunked load (loader needs `y` to do this — already loaded). Keep that as a v1.1 escape hatch; not needed for the 100 k workflow.

**`train.py`**
- `train_binary(data_dir, spec, params, val_size=0.1, seed=None) -> lgb.Booster`
- **Returns `lgb.Booster` directly** — same return type as `thrember.train_model`. `ModelBundle` is a save/load utility (in `predict.py`), not what training returns. Caller does `ModelBundle.save(booster, spec, out_dir)` explicitly. Makes `train_binary` a near-drop-in for `train_model` in comparison harnesses.
- **Binary classification only in v1.** No `objective` flip-to-multiclass surface; family/multiclass is a v2 file (`train_family`).
- Body is a near line-for-line copy of `model.py:347-370`:
  ```python
  def train_binary(data_dir, spec, params={}, val_size=0.1, seed=None):
      # NOTE: mutable default `params={}` mirrors thrember.train_model (model.py:347).
      # Footgun in principle, but `_inject_seeds` always copies before mutating, so the
      # shared default is never written to in practice. Kept as-is for thrember symmetry.
      X, y = read_vectorized_features(data_dir, "train", spec)
      X = X[y != -1, :]                                        # mirrors model.py:360
      y = y[y != -1]                                           # mirrors model.py:361
      assert len(np.unique(y)) == 2, "train_binary requires binary labels"
      assert X.shape[1] == len(spec.kept_indices), "spec/data column mismatch"
      X_tr, X_val, y_tr, y_val = train_test_split(
          X, y, test_size=val_size, stratify=y, random_state=seed,
      )
      if seed is not None:
          params = _inject_seeds(params, seed)                 # copy + warn-on-overwrite
      train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=spec.new_categorical)
      val_set   = lgb.Dataset(X_val, y_val, reference=train_set,
                              categorical_feature=spec.new_categorical)
      return lgb.train(params, train_set, valid_sets=val_set)
  ```
- Differences from `train_model`: (a) takes a `spec`; (b) uses `spec.new_categorical` (remapped) instead of the hardcoded `[2,3,4,5,6,701,702]`; (c) seed plumbing; (d) no multiclass branch — `read_vectorized_features` already raised on multilabel y.
- Validates: `len(np.unique(y)) == 2`, `X.shape[1] == len(spec.kept_indices)`.
- **Seed plumbing** (for comparison studies — the primary use case). When `seed` is set:
  - `train_test_split(..., random_state=seed, stratify=y)` — same rows go to val across runs.
  - `_inject_seeds(params, seed)` returns a copy of `params` with `seed`, `bagging_seed`, `feature_fraction_seed` set (LightGBM's three RNG knobs that demonstrably affect tree structure). `print` a warning if any are being overwritten. Never mutates the caller's `params`.
  - Rationale: ablation deltas (e.g. `AUC(keep=1000) - AUC(keep=500)`) are often the same magnitude as RNG noise (~0.001–0.003 std on 100k rows). Matched seeds remove the noise so the delta is attributable to the feature set, not the RNG state.
  - `seed=None` reproduces thrember's non-deterministic behavior — fine for one-shot training where you just want a model.
- No silent param mutation outside the seed knobs.

**`predict.py`**
- `ModelBundle` is a **save/load utility**, not a return type. Holds `(booster, spec)` after loading.
  - `ModelBundle.save(booster, spec, out_dir)` — staticmethod; writes `model.txt` + `spec.json`. Caller passes the booster from `train_binary` explicitly.
  - `ModelBundle.load(in_dir) -> ModelBundle` — reads both files, asserts `PEFeatureExtractor().dim == spec.original_dim` (fail loudly if thrember's feature width changed since training).
- `predict_file(bundle, path_or_bytes)` — instantiate `PEFeatureExtractor()` once (reuse `thrember`), extract full-width vector, slice via `bundle.spec.kept_indices`, call `bundle.booster.predict`.
- `predict_batch(bundle, paths)` — vectorized version, single extractor instance.
- This is the piece that fills the `predict_sample` gap noted earlier (`model.py:409-416` always extracts full width).
- Typical caller flow:
  ```python
  booster = train_binary(d, spec, params, seed=42)
  ModelBundle.save(booster, spec, "runs/exp01/")
  # ...later, possibly in a different process...
  bundle = ModelBundle.load("runs/exp01/")
  score  = predict_file(bundle, "suspicious.exe")
  ```

**`cli.py`** (argparse, no extra deps; invoked as `python -m thrember_lite.cli ...`)
- `build-spec --drop dropped_features.json --out spec.json [--source-note "..."]`
  Single input mode: always a Layer A `dropped_features.json`. No group-toggle CLI surface — keep the input shape uniform. The optional `--source-note` lands in `spec.json`'s `source` block as free-form provenance.
- `train <data_dir> <spec.json> <out_dir> --config lgbm_config.json [--seed N]`
- `predict <model_dir> <file>`

Logging style across the library: plain `print(...)` to stdout, matching `thrember`'s convention. No `logging` module, no levels — keeps it grep-friendly and consistent with the rest of your scripts.

## Reusing your existing scripts
Your tree already has `build_feature_index_map.py`, `mi_feature_selection.py`, `bernoulli_sample.py`. Plan: keep those as **selection-strategy scripts** that produce a `dropped_features.json` (or equivalent), and have `FeatureSpec.from_drop_columns` consume that. Don't merge the selection logic into the library — keep selection (research-y, iterates a lot) separate from training (stable).

## Validation strategy
- **Unit**: spec round-trip; `apply_spec` shape correctness; categorical remap when dropping a categorical (e.g. drop col 3 → expect `[2, 3→removed, 4→3, 5→4, 6→5, 701→699, 702→700]`); `read_vectorized_features` returns the same row count as `y` (no internal filtering).
- **Integration smoke**: tiny synthetic `.dat` files (50 rows × `original_dim`), `train_binary` 10 iterations → returns `lgb.Booster`; `ModelBundle.save` then `ModelBundle.load` round-trips; `predict_file` runs through the slice without shape errors and returns a value in `[0,1]`.
- **Behavioral parity**: with `kept_indices = list(range(original_dim))` (no drops), train both `thrember.train_model` and `train_binary` on a small subset (PDF or ELF) with the same `params`. Assert validation AUC is within a tolerance (e.g. ±0.005) and feature-importance top-10 overlap is high. Bytewise equivalence is **not** expected — `thrember.train_model` doesn't accept a seed, so its val split drifts even with `seed` set on the lite side.
- **Determinism check** (lite-only): two consecutive `train_binary(..., seed=42)` calls on the same data and spec should produce identical models. Catches regressions where a new RNG sneaks in unseeded.
  - **Primary check**: prediction equality on a fixed input — `np.array_equal(b1.predict(X_val), b2.predict(X_val))`. Robust to LightGBM dump-format quirks (timestamps, importance ordering).
  - **Secondary check**: `booster.model_to_string()` equality. Verify empirically that this holds in the LightGBM version pinned in this repo before relying on it; LightGBM's text dump can include nondeterministic header fields across versions.
  - **Required test config**: `params["num_threads"] = 1` and `params["force_row_wise"] = True`. LightGBM's parallel histogram aggregation is order-sensitive in floating point, so multithreaded runs can produce non-bytewise-identical models even with all RNG seeds pinned. The seed plumbing only governs *random* nondeterminism, not parallel-reduction nondeterminism. The determinism test is the only place this matters; production runs can use whatever thread count they want.

### Recommended workflow for comparison studies
- Pick a fixed `SEED` (e.g. 42) and use it for every config in the study. Same val rows, same LightGBM RNG state — deltas are attributable to the feature set, not noise.
- For headline numbers, repeat each config across `seed in [42, 43, 44, 45, 46]` and report mean ± std. Matched seeds eliminate within-pair noise; multiple seeds estimate run-to-run drift for context.

## Resolved decisions

1. **`dropped_features.json` schema** — keep the existing rich format (`rank`, `index`, `block`, `field`, `hashed`, `mi_score`) as the **selection input** (Layer A). The library only requires `index`; everything else is provenance for humans. The runtime spec written next to the model is a separate **Layer B** file (`spec.json`) holding `original_dim`, `kept_indices`, `original_categorical`, `new_categorical`, `block_ranges`, and a small `source` block. See "Two-layer JSON contract" above.
2. **Column-level only.** No group-toggle in v1 — not in the API and not in the CLI. `FeatureSpec` exposes only `from_drop_columns` and `from_json`. Group-based ablations are done by emitting a `dropped_features.json` from a small ad-hoc script.
3. **Binary-only v1.** `train.py` exposes only `train_binary`. No multiclass/family/OvR surface. Leave a stub comment for `train_family` as a future file.
4. **Chunked is the default path.** `read_vectorized_features` uses `BLOCK = 50_000` rows out of the box (configurable per-call). The in-memory path is opt-in via `in_memory=True` for small subsets where the simpler code path is faster. Same `(X, y)` return shape either way — callers don't branch.
5. **Package location**: `src/thrember_lite/`, sibling to `src/thrember/`. Installable, importable, and the CLI runs as `python -m thrember_lite.cli`.
6. **`block_ranges` source**: derive in-process from `PEFeatureExtractor().features` by default. `from_drop_columns(..., index_map_path=...)` overrides for offline builds. No hardcoded dict.
7. **Subset filtering happens in the trainer, not the loader.** Mirrors thrember's structure: `read_vectorized_features` returns raw rows (no `y == -1` filter), and `train_binary` does `X = X[y != -1]; y = y[y != -1]` itself — same lines as `model.py:360-361`. Test/challenge evaluation calls `read_vectorized_features` directly to get all rows. No `drop_unlabeled` parameter.
8. **Seed plumbing for comparison studies.** `train_binary(seed=N)` plumbs to both `train_test_split(random_state=N)` and a copy of `params` with `seed`/`bagging_seed`/`feature_fraction_seed` set (warn-on-overwrite). `seed=None` reproduces thrember's non-determinism. Justification: feature-ablation deltas are typically the same magnitude as RNG noise, so matched seeds across configs are required for a 1-trial comparison to be meaningful. Bytewise parity with `thrember.train_model` is still **not** in scope — thrember has no seed surface — but lite↔lite reproducibility is.
9. **CLI input is always a `dropped_features.json`**: one input mode for `build-spec`. No `--keep-groups` / `--drop-groups` flags. Optional `--source-note` for provenance.
10. **1D-y assertion**: `read_vectorized_features` fails loudly if `y` reshapes to 2D, mirroring `model.py:355-356`. Error message uses thrember's wording: `"Encountered y with invalid shape. Use train_family() instead."`
11. **Logging**: plain `print(...)` to stdout — matches `thrember`'s style.
12. **`train_binary` returns `lgb.Booster`, not `ModelBundle`.** Matches `thrember.train_model`'s return type so the two are interchangeable in comparison harnesses. `ModelBundle` becomes a save/load utility (`ModelBundle.save(booster, spec, dir)` / `ModelBundle.load(dir)`), called explicitly by the user after training.

## Implementation order
1. `spec.py` + tests (spec is the contract; everything depends on it).
2. `data.py` + tests. Chunked path is the default per Decision #4; `in_memory=True` is the simpler fallback for small subsets. Both paths share the same `(X, y)` return shape so tests can drive both with the same fixtures.
3. `train.py` + smoke test (including the determinism check with `num_threads=1`).
4. `predict.py` + `ModelBundle` round-trip test.
5. `cli.py` last — it's just argparse glue.
6. Wire `dropped_features.json` → `FeatureSpec` and run end-to-end on a small subset (PDF or ELF) to validate before touching Win32.
