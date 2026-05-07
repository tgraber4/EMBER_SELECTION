# MI Feature Selection Plan

## Goal

Run Categorical-Continuous Mutual Information on a subsection of the EMBER2024 dataset to identify the weakest features and output the list of features to drop.

- **Total features**: 2,568
- **Features to drop (n)**: `round(0.10 × 2568)` = **257** — changeable via the `N_DROP` variable in `main()`
- **Input**: single `.jsonl` file
- **Output**: console table + `dropped_features.csv` + `dropped_features.json`

---

## Script

**File**: `mi_feature_selection.py` (repo root)

### Parameters (edit inside `main()`)

| Variable | Default | Description |
|---|---|---|
| `JSONL_FILE` | `"ember_data/train_week1.jsonl"` | Path to one `.jsonl` file |
| `N_DROP` | `257` | Features to mark for dropping (~10%) |
| `MAX_SAMPLES` | `100_000` | Max labeled rows to stream |
| `OUTPUT` | `"dropped_features.csv"` | Output CSV path |
| `FEATURE_MAP` | `"Documentation/feature_index_map.json"` | Label map (already exists) |
| `SEED` | `42` | Random seed |

### Pipeline

1. **Load labels** — reads `Documentation/feature_index_map.json` → `dict[index → {block, field, hashed}]`
2. **Stream .jsonl** — reads one file line-by-line, skips unlabeled rows (`label == -1`), stops at `MAX_SAMPLES`
3. **Vectorize** — calls `PEFeatureExtractor.process_raw_features(obj)` on each row → `X[N, 2568]`
4. **Compute MI** — `sklearn.feature_selection.mutual_info_classif` with categorical mask for indices `[2, 3, 4, 5, 6, 701, 702]`
5. **Rank & select** — `np.argsort(mi_scores)[:N_DROP]` → bottom 257 features
6. **Output** — print ranked table to console, save CSV + JSON

### Categorical feature indices

| Index | Block | Field |
|---|---|---|
| 2–5 | general | start_bytes[0–3] (first 4 file bytes) |
| 6 | general | is_pe flag |
| 701–702 | header | machine_type, subsystem_type |

All other features are treated as continuous.

---

## Key Considerations

**Runtime**: ~6–15 minutes at 100k samples × 2568 features. Reduce `MAX_SAMPLES` if slow — MI quality degrades gracefully with fewer samples (10k–20k is still informative).

**Timing**: Total elapsed time is tracked with `time.perf_counter()` from the start to the end of `main()` and printed to the terminal at the very end:
```
Total runtime: 487.3s (8.1 min)
```

**Hashed features**: About 1,600 of 2,568 features are hash bucket encodings (ImportsInfo, SectionInfo, ExportsInfo, RichHeader). Their field names in the output will be generic (e.g. `imports_lib_hash[143]`). This is expected — hash features don't have meaningful individual names.

**What low MI means here**: A feature near-zero MI against the binary malware/benign label is individually uninformative. However, due to the Cons noted in `mutual_info.md`, combined features (e.g. keylogging + network access) may be valuable even if each scores low alone. Treat the drop list as a starting point, not a final answer.

**n is fully adjustable**: Change `N_DROP` in `main()` to any integer from 1 to 2568. For example, set it to `round(0.05 * 2568)` = 128 for a more conservative 5% cut.

---

## Output Format

### Console

```
Rank   Index            Block     MI Score  Hashed  Field
--------------------------------------------------------------------
   1    2404          exports   0.00000000     yes  exports.names_hashed[128]
   2     ...
...

Drop candidates by block:
         imports: 121
 pefilewarnings:  47
         exports:  38
         ...
```

### dropped_features.csv

```
rank,index,block,field,hashed,mi_score
1,2404,exports,exports.names_hashed[128],True,0.0
...
```

### dropped_features.json

Same data as the CSV in JSON array format, for programmatic use.
