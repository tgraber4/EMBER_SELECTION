# Finishing `thrember_lite`

Follow-up to the original `thrember_lite` design plan, which was deleted on 2026-09-15 and can be recovered with `git show 0d5ec8d:Documentation/thrember_lite_plan.md`. In this doc, "the plan" means that original. The library code is complete and matches it: all five modules, the CLI, and all 12 of its design decisions are in place, and the 30 existing tests pass (14 spec, 9 data, 7 training smoke). What's left is mostly **testing**, plus a few fixes found while reviewing the code.

**Already checked by hand, but not automated:** the full prediction path (`ModelBundle.load` → `predict_file`) through the CLI, on the MI01, EMBED01, and SHAP01 models with one unsigned and one signed Windows file. This only worked after the `signify` fix (`src/thrember/features.py:955`). The model line-ending problem that also blocked it is now fixed (see §3.1).

**Decided not needed (left out of this doc):**
- multi-seed reporting
- the second determinism check (`model_to_string()`)
- a PDF/ELF trial run
- `optimize_memory=True`
- four small differences from the plan: `predict_file` creating a new extractor per call, `predict_batch` skipping the width check, the wrong `block_ranges` example in the plan, and `git_sha` missing from `source`

---

## 1. Tests to implement

All automatic tests go in `custom/thrember_lite/tests/`.

| # | Test | Source |
|---|---|---|
| 1.1 | Real `ModelBundle.load` round-trip | From the plan, never finished |
| 1.2 | `predict_file` / `predict_batch` on a real file | From the plan, never finished |
| 1.3 | Parity with `thrember` (synthetic, automatic) | Automatic version of the plan's parity check |
| 1.4 | Signed-file extraction (Authenticode) | New: guards the `signify` fix |
| 1.5 | CLI smoke test | New: found in review |
| 1.6 | Block ranges derived from `PEFeatureExtractor` | New: found in review |
| 1.7 | Full parity run on real data | From the plan; one-time script, not an automatic test |

### 1.1 Real `ModelBundle.load` round-trip
**Why:** `test_model_bundle_save_load_round_trip` never calls `ModelBundle.load`. Its toy data is 8 features wide, but `load` checks that `spec.original_dim` equals `PEFeatureExtractor().dim` (2,568). So the real `load` and its width check never run in the tests.

**How:**
- Write a synthetic `X_train.dat` / `y_train.dat`, about 200 rows × 2,568 columns (~2 MB), with a learnable signal in one column.
- Build a spec with `original_dim=2568` that drops a few columns, including at least one categorical.
- Train with the smoke params (`num_threads=1`, `force_row_wise=True`, 10 iterations), then `ModelBundle.save` and `ModelBundle.load`.
- Assert:
  - the loaded spec matches the saved one (`kept_indices`, `original_categorical`, `new_categorical`)
  - `booster.num_feature() == len(kept_indices)`
  - predictions on a fixed input match before and after loading
- **Failure case:** save a bundle whose `original_dim` isn't 2,568 and assert `ModelBundle.load` raises `RuntimeError`.

### 1.2 `predict_file` / `predict_batch` on a real file
**Why:** the plan's smoke test says `predict_file` should run through the column slice without errors and return a score in `[0, 1]`. The current smoke test skips it on purpose.

**How:**
- Reuse the bundle from 1.1.
- Input: the running Python executable (`sys.executable`), which exists on every machine.
- Assert:
  - the score is in `[0, 1]`
  - `predict_file(path) == predict_file(path_bytes)`
  - `predict_batch([path, path])` returns two scores, both equal to `predict_file(path)`
  - the score equals `booster.predict([apply_spec(PEFeatureExtractor().feature_vector(bytes), spec)])`
- **Platform note:** on Linux (the HPC), `sys.executable` is an ELF file, not a PE. Check that extraction still returns 2,568 values there; if not, skip the test on non-Windows machines.

### 1.3 Parity with `thrember` (synthetic, automatic)
**Why:** `custom/runs/no_drop` was most likely trained with the original `thrember` path, since its `Classifier_ROC_AUC.pdf` is what `examples/eval_lgbm.py` writes. MI01, EMBED01, and SHAP01 were trained with `thrember_lite`. This test confirms that "baseline vs. drop" comparisons measure the dropped features, not a difference in training code.

**How:**
- Write synthetic `X_train.dat` / `y_train.dat` at 2,568 columns, including some `y == -1` rows.
- Build a spec that keeps all 2,568 columns.
- Assert:
  - `thrember_lite.read_vectorized_features(dir, "train", spec)` returns exactly the same `X` and `y` as `thrember.read_vectorized_features(dir, "train")`. Neither function filters rows (`model.py:250-272`, `data.py:23-87`).
  - `spec.new_categorical == [2, 3, 4, 5, 6, 701, 702]`, the list `thrember.train_model` hardcodes (`model.py:365-366`)
- **Reasoning:** the rest of the two training functions is identical: remove `y == -1` rows, split 90/10 stratified, then `lgb.train(params, train_set, valid_sets=val_set)`. So identical data and categoricals mean identical training inputs. The only built-in difference is that `train_model`'s validation split has no seed.

### 1.4 Signed-file extraction (Authenticode)
**Why:** with `signify` 0.9.x, `signed_data.certificates` became a `CertificateStore` that can't be sliced. `certs[:-1]` raised an uncaught `TypeError` for every PE with an embedded signature (24 of 62 System32 programs tested). This broke `predict_file`, `predict_batch`, and `thrember.predict_sample`. It's fixed with `list(...)` at `features.py:955`, but no test guards it.

**How:**
- Use `C:\Windows\explorer.exe`, and skip the test if the file doesn't exist (e.g. on the HPC).
- Run `AuthenticodeSignature().raw_features(bytez, pefile.PE(data=bytez, fast_load=True))` and `PEFeatureExtractor().feature_vector(bytez)`.
- Assert:
  - no exception
  - `num_certs >= 1`
  - `parse_error == 0`
  - the vector has 2,568 values
- **Decide where it lives:** this tests `thrember`, not `thrember_lite`. Either put it in `custom/thrember_lite/tests/` (the only test folder) or start a test folder for `thrember`.

### 1.5 CLI smoke test
**Why:** `cli.py` has no tests. The plan called it "just argparse glue", but it's the documented way to build specs and train (see the README workflow).

**How:**
- Call `thrember_lite.cli.main([...])` in-process (it returns an exit code), reusing the synthetic data from 1.1/1.2:
  - `build-spec --drop <Layer-A json> --out <spec.json> --index-map custom/docs/feature_index_map.json --source-note test`
  - `train <data_dir> <spec.json> <out_dir> --config <small params json> --seed 42`
  - `predict <out_dir> <sys.executable>`
- Assert:
  - each command returns `0`
  - `spec.json` and `model.txt` are created
  - `spec.json`'s `source` includes the note
  - `predict` prints `<path>\t<score>` with a score in `[0, 1]`

### 1.6 Block ranges derived from `PEFeatureExtractor`
**Why:** by design, `from_drop_columns` derives `original_dim` and `block_ranges` from `PEFeatureExtractor` by default, and `run_thrember_lite.py` relies on that. But every existing test passes `index_map_path`, so `_derive_layout_from_extractor` is never tested.

**How:**
- Build one spec with no `index_map_path` and one with `index_map_path=custom/docs/feature_index_map.json`.
- Assert both have the same `original_dim` (2,568) and `block_ranges`. Both are end-exclusive after loading; the index map stores end-inclusive ranges and `_read_index_map` converts them.
- Checked by hand on 2026-09-15: the two match exactly, so this test should pass as-is.

### 1.7 Full parity run on real data (one-time script)
**Why:** this is the plan's full parity check. It's too slow for an automatic test and needs the dataset.

**How:**
- Suggested file: `custom/compare_thrember_lite_parity.py`, writing its results to a new folder under `custom/`.
- Train `thrember.train_model` and `thrember_lite.train_binary` (spec keeping all columns) on the same vectorized sample with `examples/lgbm_config.json`.
- Score both on the test set.
- Pass if:
  - ROC AUC is within **±0.005**
  - the top-10 features by gain overlap heavily
- The models won't be identical, because `train_model`'s validation split is unseeded.

---

## 2. Test environment

- **Current state:** `pytest` isn't installed in `.venv`. Each test file runs through its own built-in runner:
  ```bash
  PYTHONPATH=custom python -m thrember_lite.tests.test_spec
  PYTHONPATH=custom python -m thrember_lite.tests.test_data
  PYTHONPATH=custom python -m thrember_lite.tests.test_train_smoke
  ```
- **Add `pytest` as an optional dev dependency** so all tests run with one command:
  - `pyproject.toml`: `[project.optional-dependencies]` → `dev = ["pytest"]`
  - `setup.cfg`: `[options.extras_require]` → `dev = pytest`
  - install: `pip install -e ".[dev]"`
  - run: `PYTHONPATH=custom pytest custom/thrember_lite/tests` (`thrember_lite` is no longer installed by `pip install -e .`, since it moved to `custom/`)
- **Keep the built-in `__main__` runners** in new test files too, so they still run without `pytest`.
- **Shared helpers:** tests 1.1, 1.2, 1.3, and 1.5 all need 2,568-wide synthetic `.dat` files and a small trained bundle. Put the helpers in a plain module (e.g. `tests/_helpers.py`) rather than a `pytest` `conftest.py`, so the built-in runners can use them too.

---

## 3. Other improvements

### 3.1 ✅ Done: saved model files didn't load in the Windows checkout
- **Problem:** every `model.txt` in `runs/` and `output/` fails to load with `Model format error, expect a tree here`.
- **Cause:** `core.autocrlf=true`. Git stores the files with plain newlines, but the working copies have Windows line endings (CRLF), which LightGBM's text model reader can't handle. Copies converted back to plain newlines load fine (e.g. MI01: 2,311 features, 500 trees).
- **Affects on this machine:** `ModelBundle.load`, CLI `predict`, `eval_thrember_lite.py`, and `examples/eval_lgbm.py` with committed models. Linux (HPC) and newly trained models are unaffected.
- **Fixed on 2026-09-15:** added a `.gitattributes` with `*model.txt -text` and `*.model -text`, then re-checked out the 7 model files. All of them load again, and CLI `predict` works on the real `runs/MI01`.
- `.gitattributes` is committed. After pulling it, any other Windows clone has to re-checkout its model files once (delete them, then `git checkout -- <files>`).

### 3.2 SHAP converter reads the wrong file (affects `build-spec` input)
- `convert_shap_to_mi_format.py` reads `custom/feature_selection/shap/output/dropped_features.csv`, but `shap_cluster_feature_selection.py` now writes `shap_dropped_features.csv`.
- An old `custom/feature_selection/shap/output/dropped_features.csv` still exists, so the converter quietly converts old results instead of failing.
- **Fix:** point `INPUT_PATH` at `custom/feature_selection/shap/output/shap_dropped_features.csv`.

### 3.3 `block` / `in_memory` can't be set from `train_binary` or the CLI (low priority)
- The plan says to lower the loader's `block` size (e.g. to 10k) when RAM is tight, but `train_binary` always calls `read_vectorized_features` with the defaults, and the CLI has no flag for it.
- **Fix:** add `block` / `in_memory` keyword arguments to `train_binary` that pass straight through, and a `--block` flag to `cli.py train`. This only matters for much larger datasets than the current sample.

### Noted elsewhere (not `thrember_lite`)
- Fixed on 2026-10-20: `check_labels.py` and `count_rows.py` used to look for `ember_data/` next to the scripts. They now resolve it from the repo root.
- Fixed on 2026-10-20: `custom/docs/overview/process_overview.md` now matches the current sampling script (separate train/test files, per-split probabilities) and no longer links to missing files.

---

## 4. Suggested order

1. ~~**§3.1 model files**~~: done.
2. **§2 test environment:** `pytest` dev dependency and shared helpers.
3. **1.1 → 1.2 → 1.5:** these share the same synthetic data and bundle.
4. **1.3 and 1.6:** quick, no training needed.
5. **1.4:** signed-file test, once its location is decided.
6. **§3.2 SHAP converter:** before any new SHAP run.
7. **1.7 parity script:** needs vectorized real data.
8. **§3.3:** only if RAM becomes a problem.
