# Plan: find the data type of every feature in the EMBER feature vector

**Status:** Not started. Nothing has been implemented yet.

## Context
The goal is to know what kind of data each of the 2,568 feature-vector columns holds. These are the 12 blocks listed in `Documentation/feature_vector_membership.md`.

What I found while exploring:
- **The storage type is float32 for every column.** Every `process_raw_features` in `src/thrember/features.py` ends in `.astype(np.float32)`. `PEFeatureExtractor.process_raw_features` (`features.py:1099`) and the `.dat` memmaps (`model.py:137`) are float32 too. So `dtype` says nothing useful on its own.
- **The useful answer is the semantic type** of each column: continuous, proportion, count, binary flag, categorical code, hashed bucket, large integer/address, or constant.
- **Labels already exist.** `Documentation/feature_index_map.json` gives `index`, `block`, `block_index`, `field` and `hashed` for all 2,568 columns. It was built by `custom_scripts/Helper Scripts/build_feature_index_map.py`, so labeling is solved and only the type is missing.
- **Two sources of truth need to be combined:**
  1. **The code** says what each column is meant to be. For example, `coff.machine` is a categorical index, `histogram` is normalized to sum to 1, and `imports.*_hashed` uses `alternate_sign=False`, so it holds counts of 0 or more.
  2. **The real data** confirms it and catches surprises:
     - `exports.count_quirk` always equals 128 (`features.py:559`).
     - `DataDirectories` skips the last directory (`features.py:873`, `range(1, len-1)`), so `RESERVED` is probably always 0.
     - Hashers in section, exports and richheader use the default `alternate_sign=True`, so they can be negative.
     - Large integers like `image_base` and timestamps lose precision in float32.

## Recommended approach
Write one script that assigns a **declared type** from the code, measures an **observed type** from sampled rows, and reports where the two disagree.

### 1. New script: `custom_scripts/Helper Scripts/profile_feature_types.py`
**Inputs**
- `Documentation/feature_index_map.json`, reused for labels.
- `ember_data/PE_train_sampled.jsonl`, the existing Bernoulli sample. Read it with `thrember.model.raw_feature_iterator` (`model.py:35`).
- Arguments: `--n-rows` (default 20,000), `--out-dir` (default `Documentation/Feature Types/`).

**Step A: declared type (rule table keyed on block and field)**

| Declared kind | Where it comes from |
|---|---|
| `continuous` | general.entropy, strings.avlength, strings_entropy, section entropy/ratio min/max, overlay.size_ratio/entropy |
| `proportion` (0–1) | histogram, byteentropy, strings.printabledist |
| `count` (int ≥ 0) | general.size, numstrings, printables, string_count_*, section.n_*, imports.n_*, richheader.number_of_pairs, auth.num_certs/chain_max_depth, pewarn.total_count, coff.number_of_*, overlay.size |
| `binary` (0/1) | coff.characteristics:*, optional.dll_characteristics:*, has_relocs, has_dynamic_relocs, auth flags, pewarn:* one-hots |
| `categorical` | is_pe, start_bytes[0..3], coff.machine(cat), optional.subsystem(cat). Same set as `DEFAULT_CATEGORICAL` [2,3,4,5,6,701,702] |
| `hashed_unsigned` | imports.libraries_hashed, imports.lib_func_hashed |
| `hashed_signed` | section.*_hashed, exports.names_hashed, richheader.pair_hashed |
| `int_large` (address/size/timestamp; float32 may lose precision) | header sizeof_*/address/image_base/checksum/timestamp, dos.*, datadir.*.size/virtual_address, auth.latest_signing_time/signing_time_diff |
| `constant_quirk` | exports.count_quirk |

The rule table must cover every one of the 2,568 indices. Assert this, the same way `build_feature_index_map.py` checks dimensions.

**Step B: observed stats**
- Parse `n_rows` rows and call `PEFeatureExtractor().process_raw_features(row)` on each to build an N×2568 float32 matrix. About 50 MB for 20k rows.
- Compute these per column with NumPy: `min`, `max`, `n_unique`, `frac_zero`, `all_integer`, `is_binary` (values ⊆ {0,1}), `has_negative`, `is_constant`, and `exceeds_float32_exact` (any |x| > 2^24).

**Step C: observed kind and mismatch**
- Derive `observed_kind` from the stats: constant → binary → integer (count or signed int) → bounded 0–1 float → continuous.
- Set `mismatch=True` when the observed kind contradicts the declared one. Examples: a declared `count` that has non-integer values, or a declared `binary` that holds a value above 1.

**Outputs**
- `Documentation/Feature Types/feature_types.csv`: one row per index, with `index, block, field, hashed, declared_kind, observed_kind, mismatch` plus the Step B stats.
- `Documentation/Feature Types/feature_types_summary.md`: for each block, counts per kind, the list of constant/dead columns, the mismatches, and the float32-precision columns.

### 2. Why not the alternatives
- **Reading only `Documentation/Example Row/example_row_raw_feature_data.json`:** one row shows raw JSON types (int, float, list), not vector semantics, and it can't expose constant columns or signed hashing.
- **Profiling data only:** a categorical code such as `coff.machine` looks exactly like a count, so the code is needed to tell them apart.

## Files
- **New:** `custom_scripts/Helper Scripts/profile_feature_types.py`
- **New outputs:** `Documentation/Feature Types/feature_types.csv`, `feature_types_summary.md`
- **Reused, not modified:** `Documentation/feature_index_map.json`, `thrember.PEFeatureExtractor`, `thrember.model.raw_feature_iterator`, `ember_data/PE_train_sampled.jsonl`
- **Unchanged:** nothing in `src/thrember/`

## Verification
1. Run `.venv\Scripts\python "custom_scripts/Helper Scripts/profile_feature_types.py" --n-rows 2000` as a quick smoke test, then the full default run.
2. Check that the CSV has 2,568 rows and that every index has a `declared_kind`.
3. Spot-check values that are known from the code:
   - Rows 7–262 (histogram) are `proportion` with max ≤ 1.
   - Index 2276 (`exports.count_quirk`) holds only {0, 128}.
   - Indices 701 and 702 are integer-valued.
   - `pewarn:*` columns are binary.
   - At least one `section.*_hashed` column has `has_negative=True`.
   - `datadir.RESERVED.*` is constant 0.
4. Read the mismatch list in the summary. Each entry is either a wrong rule (fix the table) or a real finding worth noting for feature selection.
