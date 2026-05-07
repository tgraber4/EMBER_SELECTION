# Feature Selection Plan: EMBER2024 with SHAP & Correlation Clustering (v4)

This document outlines the strategic plan for performing feature selection on
the **EMBER2024** malware dataset using a combination of **SHAP (SHapley
Additive exPlanations)** and **Correlation Clustering**. The objective is to
reduce the feature space by dropping the bottom 10% of features while
maintaining or improving the performance of a **LightGBM** classifier.

**Method at a glance — SHAP + Correlation Clustering:**

1. **SHAP** provides the per-feature importance signal (mean-|SHAP|), used
   both to rank features and to pick a representative within each cluster.
2. **Correlation Clustering** groups redundant features via Pearson
   correlation followed by hierarchical (`average`-linkage) clustering on a
   distance derived from that correlation. We correlate each feature's
   *SHAP-contribution vector* (one value per sample) rather than its raw
   input column — same correlation-clustering technique, applied in the
   model-attribution space so it is robust to categorical features and to
   hashed-bucket sparsity. See §3 Phase 2 for the math.

Together: SHAP ranks importance, correlation clustering identifies
redundancy, and the pruning rule in §3 Phase 3 combines both to drop exactly
the bottom 10%.

---

## 1. Project Overview

* **Dataset:** EMBER2024 (v3 feature set).
* **Total Features:** ~2,568 dimensions (flattened vector; assert against
  `PEFeatureExtractor().dim` at runtime — never hard-code).
* **Dataset Status:** 100,000 samples already extracted and prepared.
* **Target:** Identify and remove the bottom 10% (~257 features) based on
  importance and redundancy.
* **Model:** LightGBM (binary).

---

## 2. Deliverable

A new Python script — `custom_scripts/shap_cluster_feature_selection.py` —
built off `train_custom_lgbm.py` (reuses `find_jsonl`, `vectorize_split`,
`read_vectorized`, `assert_binary`, `CAT_FEATURES`, `RANDOM_STATE = 0`).

**Inputs (CLI):**

```
python shap_cluster_feature_selection.py <data_dir> <model_path> \
    --config-file ../examples/lgbm_config.json \
    [--drop-fraction 0.10] \
    [--shap-sample 20000] \
    [--cluster-threshold 0.10] \
    [--shap-var-eps 1e-12] \
    [--feature-map ../Documentation/feature_index_map.json] \
    [--dropped-out dropped_features.csv]
```

* `<data_dir>` — folder containing exactly one `*train*.jsonl` and one
  `*test*.jsonl` (same convention as `train_custom_lgbm.py`). Vectorized
  `X_train.dat` / `y_train.dat` / `X_test.dat` / `y_test.dat` are written
  into this folder.
* `<model_path>` — where to save the trained LightGBM model used to derive
  SHAP values.
* `--feature-map` — path to `feature_index_map.json` (see §2.1). If the
  file is missing or its `dim` does not match `PEFeatureExtractor().dim`,
  fall back to generic `block="unknown"`, `field=f"Column_{i}"`,
  `hashed=False` per the same convention used in
  `train_custom_lgbm_fs.py::load_feature_names`.

### 2.1 Feature index map (`feature_index_map.json`)

This file maps each integer feature index `0 .. dim-1` to a human-readable
identifier. It is the single source of truth for the `block`, `field`, and
`hashed` columns of the CSV output.

Expected structure (already used by `mi_feature_selection.py` and
`train_custom_lgbm_fs.py`):

```json
{
  "dim": 2568,
  "entries": [
    {"index": 0, "block": "general", "block_index": 0, "field": "size",       "hashed": false},
    {"index": 1, "block": "general", "block_index": 1, "field": "vsize",      "hashed": false},
    ...
  ]
}
```

The script must:

1. Load the JSON, build `index_map: dict[int, dict] = {e["index"]: e for e in entries}`.
2. Validate `data["dim"] == PEFeatureExtractor().dim` and
   `len(entries) == dim`. On mismatch, print a warning and fall back to
   stub entries (so the pipeline still produces a valid CSV).
3. Use `index_map[i].get("block", "unknown")`,
   `index_map[i].get("field", f"Column_{i}")`,
   `index_map[i].get("hashed", False)` when populating the CSV.

**Output:** a single CSV of dropped features at `--dropped-out` (default
`dropped_features.csv`) with the schema:

```
rank,index,block,field,hashed,mean_abs_shap,cluster_id,drop_reason
```

* `index` is required — the downstream consumer (`thrember_lite.FeatureSpec.from_drop_columns`) keys on it.
* `block`, `field`, `hashed` are populated from the feature index map (§2.1).
* `cluster_id` is the integer cluster label from `fcluster`, or `-1` for
  features that were excluded from clustering (low/zero SHAP variance — see
  Phase 2 step 4).
* `drop_reason ∈ {"inactive", "redundant", "low_importance"}`
  (see Phase 3 for the meaning of each).
* Sorted by `rank` ascending (rank 1 = first feature dropped).

**Downstream handoff:** the CSV (or its `.json` sibling) is consumed by
`thrember_lite.FeatureSpec.from_drop_columns` **as a separate manual step** —
this script does **not** invoke training. The dropped-features file is the
entire output of this stage.

---

## 3. Methodology Phases

### Phase 1: Data Ingestion & Baseline

1. **Data loading:** Reuse `find_jsonl()` + `vectorize_split()` +
   `read_vectorized()` from `train_custom_lgbm.py` to materialize
   `X_train`, `y_train`, `X_test`, `y_test` from `<data_dir>`.
2. **Train/val split:** 90/10 stratified split on `y_train`, with
   `random_state=0` (matches `train_custom_lgbm.py`).
3. **Baseline training:** Train LightGBM with
   `categorical_feature=CAT_FEATURES` on the train split, using
   `lgb.early_stopping(50)` on the val split so the importance signal isn't
   overfitted to 100k samples. Capture `best_iter` defensively (matches the
   pattern in `train_custom_lgbm.py`):

   ```python
   bi = model.best_iteration
   best_iter = int(bi) if bi is not None and bi > 0 else None
   ```

   This avoids letting `-1` (the LightGBM "early stopping never fired"
   sentinel) leak into `tree_limit` / `num_iteration`.
4. **Record baseline metrics on the test set:**
   accuracy, ROC AUC (`sklearn.metrics.roc_auc_score`), log-loss
   (`sklearn.metrics.log_loss`). All test-set predictions use
   `num_iteration=best_iter`.
5. Save the model to `<model_path>` with `num_iteration=best_iter`.

### Phase 2: SHAP Importance & Correlation Clustering

1. **SHAP value calculation:**
   * Use `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")`.
     No background data is needed for `tree_path_dependent`.
   * **Subsample** the training data:
     - Compute `n = min(args.shap_sample, len(X_tr))`.
     - If `n >= len(X_tr)`, skip subsampling and use `X_sub = X_tr`,
       `y_sub = y_tr` directly. (`StratifiedShuffleSplit` rejects
       `train_size == n_samples` because `test_size` would be 0.)
     - Otherwise use
       `StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=0)`
       stratified on `y_tr` and take the train fold.
     - Default `--shap-sample 20000`.
   * **Gate on `best_iter`:** call `explainer.shap_values(X_sub, tree_limit=best_iter, check_additivity=False)`.
     This ensures SHAP reflects the same trees the saved model uses and
     avoids overfit-tree contamination.
   * **Normalize the return shape.** SHAP's API differs across versions for
     binary models:
     * If the return is a list of length 2 (`[shap_class0, shap_class1]`),
       use `S = shap_values[1]` (positive class).
     * If it is an `ndarray` of shape `(n, dim)`, use it directly.
     * If it is `(n, dim, 2)`, use `S = shap_values[..., 1]`.
   * Assert `S.shape == (n, dim)` and cast to `float32`.
   * **Note:** TreeSHAP on a binary LightGBM Booster returns values in
     **log-odds space** (the model's raw output), not probability. This
     does not affect the pipeline — `mean(|SHAP|)` and Pearson correlation
     of SHAP columns are both well-defined in log-odds space and that's
     the space the model actually decides in. Do **not** apply a logistic
     transform.

2. **Global feature importance** (mean absolute SHAP per feature):

$$\text{Importance}_j = \frac{1}{N} \sum_{i=1}^{N} \left| \phi_{i,j} \right|$$

3. **Correlation matrix on SHAP columns** (replaces Spearman on raw
   inputs — same hierarchical-correlation-clustering technique, applied to
   the SHAP attribution columns):
   * Treat each feature `j` as a vector `S[:, j]` of its per-sample SHAP
     contributions.
   * Compute Pearson correlation `R` over the columns of `S`
     (`R[i, j] = corr(S[:, i], S[:, j])`) via `np.corrcoef(S, rowvar=False)`.
     `np.corrcoef` returns float64; cast to float32 immediately to keep the
     matrix at ~26 MB rather than ~52 MB at `dim ≈ 2,568`.
   * Convert to distance: $\text{Dist}_{ij} = \sqrt{2 \cdot (1 - R_{ij})}$.

   **Naming note:** throughout this document, `dim` is the integer total
   feature count (= `PEFeatureExtractor().dim`), and `Dist` is the
   `dim × dim` (or `n_active × n_active`) distance matrix. They are not
   the same `D`.

4. **Pre-cluster filter — handle zero/near-zero SHAP variance.**
   Features the model never used (or used so little their SHAP column is
   noise) produce undefined Pearson correlation (`Var = 0 ⇒ NaN`) and
   would crash `linkage`. Mitigation:
   * Compute `var_S = S.var(axis=0)`.
   * Define `inactive_mask = var_S < args.shap_var_eps` (default `1e-12`).
   * Features in `inactive_mask` are **excluded from clustering**, assigned
     `cluster_id = -1`, and routed to **Rank 0** in the Phase 3 drop
     ordering (with their actual mean-|SHAP| score, which will be at or
     near zero). Rank 0 is dropped first.
   * Build `R` and `Dist` only over the active features.

5. **Sanitize `R` and `Dist` before clustering.**
   * Replace any residual `NaN` or `inf` in `R` with `0` (correlation
     undefined ⇒ treat as orthogonal).
   * Symmetrize: `R = 0.5 * (R + R.T)` (absorbs floating-point asymmetry
     before distance conversion).
   * Compute `Dist = np.sqrt(2.0 * (1.0 - R))`.
   * Force the diagonal: `np.fill_diagonal(Dist, 0.0)` (guards against
     `squareform`'s strict zero-diagonal check).
   * Clip to `[0, 2]` to absorb floating-point overshoot.

6. **Hierarchical clustering** — use **`method="average"`**, not Ward.
   Ward's dendrogram heights are variance increases, not distances, so a
   threshold of `t=0.10` would not correspond to "R ≥ 0.995." With
   `average` linkage, `fcluster(..., t, criterion="distance")` cuts at the
   actual mean pairwise distance, so the threshold has the meaning we want:

   ```python
   from scipy.spatial.distance import squareform
   from scipy.cluster.hierarchy import linkage, fcluster
   condensed = squareform(Dist, checks=False)
   Z = linkage(condensed, method="average")
   labels_active = fcluster(Z, t=args.cluster_threshold, criterion="distance")
   ```

   Default `t = 0.10` corresponds to mean SHAP-vector correlation ≥ 0.995.
   **Validate** `0 < args.cluster_threshold <= sqrt(2)` at CLI parse time
   (any larger value collapses every active feature into one cluster).

   Reassemble full-length cluster labels: `fcluster` already returns labels
   in `1..K` (always positive), so no shift is needed. Active features
   carry their `fcluster` label directly; inactive features get `-1`.

   **Why average linkage is appropriate here:**
   * The threshold `t` is interpretable in the same units as `Dist`.
   * `Dist = sqrt(2(1-R))` is a Euclidean distance, so average linkage is
     statistically well-defined.
   * Categorical features (indices 2, 3, 4, 5, 6, 701, 702) are clustered
     on their SHAP outputs like any other feature — categoricals have no
     defined rank correlation on raw inputs, but their SHAP columns are
     real-valued and well-behaved.
   * Hashed-bucket sparsity inflates raw-feature correlation via co-zeros;
     SHAP vectors are non-zero only where the model actually used the
     feature, so co-zero noise disappears.

### Phase 3: The "Bottom 10%" Pruning Logic

The goal: avoid dropping a feature with low mean-|SHAP| only because its
contribution is split across several near-identical "clones."

**Selection algorithm:**

1. **Identify clusters** via the Phase 2 cut. Inactive features
   (cluster_id `-1`) are not in any active cluster.
2. **Identify cluster representatives:** within each cluster of size > 1,
   keep the feature with the **highest** mean-|SHAP|. Ties on mean-|SHAP|
   for representative selection are broken by **lower feature index wins**
   (canonical / earliest feature stays). Mark the rest as **Redundant
   Candidates**.
3. **Ranking** (three tiers, dropped in order):
   * **Rank 0** — Inactive features (`cluster_id = -1`, zero-variance SHAP,
     i.e., never used by the model), sorted by mean-|SHAP| ascending.
     These are the most obvious things to drop and go first.
   * **Rank A** — Redundant Candidates, sorted by mean-|SHAP| ascending.
   * **Rank B** — Active cluster representatives + active singleton
     clusters, sorted by mean-|SHAP| ascending.
4. **Pruning to exactly `n_drop = round(dim × drop_fraction)`:**
   * Drop from **Rank 0** first (model never used them).
   * If `|Rank 0| < n_drop`, top up from **Rank A** (redundant clones
     before any unique-but-low-importance feature).
   * If `|Rank 0| + |Rank A| < n_drop`, top up from the bottom of
     **Rank B**.
   * Stop the moment `n_drop` is reached. The 10% ceiling is a hard
     cap — we never drop more than the quota, even if more inactive or
     redundant features exist.
5. **Tie-breaking on the drop ranking** (deterministic): within all three
   tiers, ties on mean-|SHAP| are broken by **higher feature index
   dropped first**.
6. **Tag** each dropped row with:
   * `drop_reason = "inactive"` (came from Rank 0),
   * `drop_reason = "redundant"` (came from Rank A), or
   * `drop_reason = "low_importance"` (came from Rank B).

Validation of the reduced feature set (re-training on the kept features
and comparing metrics against the baseline) is **out of scope for this
script** — that happens downstream by feeding the dropped-features file
into `thrember_lite` and re-training.

---

## 4. Implementation Checklist

- [ ] **Environment Setup:** Install `lightgbm`, `shap`, `scipy`,
      `scikit-learn`. Imports needed:
      `sklearn.metrics.roc_auc_score`, `sklearn.metrics.log_loss`,
      `sklearn.model_selection.StratifiedShuffleSplit`,
      `scipy.cluster.hierarchy.linkage`, `scipy.cluster.hierarchy.fcluster`,
      `scipy.spatial.distance.squareform`.
- [ ] **Script skeleton:** Copy/import the helpers from `train_custom_lgbm.py`
      (`find_jsonl`, `vectorize_split`, `read_vectorized`, `assert_binary`,
      `CAT_FEATURES`, `RANDOM_STATE`).
- [ ] **Feature map:** Load `feature_index_map.json` per §2.1, with stub
      fallback on dim mismatch / missing file.
- [ ] **Baseline:** Train LightGBM, save to `<model_path>` with
      `num_iteration=best_iter`, record test accuracy / AUC / log-loss.
- [ ] **SHAP:** Compute SHAP on a stratified subsample of size
      `min(--shap-sample, len(X_tr))`, gated to `best_iter`. Normalize
      return shape to `(n, dim)`.
- [ ] **Variance filter:** Mark features with `Var(S[:, j]) < --shap-var-eps`
      as inactive (cluster_id `-1`), exclude from clustering.
- [ ] **Clustering:** Pearson on active SHAP columns → `Dist = sqrt(2(1-R))` →
      sanitize NaN/diag → `linkage(method="average")` →
      `fcluster(t=--cluster-threshold, criterion="distance")`.
- [ ] **Selection:** Generate exactly `n_drop` indices via
      Rank 0 → Rank A → Rank B with the tie-break rules in Phase 3.
- [ ] **Export CSV:** Write `dropped_features.csv` with the schema in §2
      (must include `index` so the downstream consumer works). Do **not**
      invoke training here — that's a separate downstream step.

---

## 5. Technical Considerations

* **SHAP cost is the bottleneck**, not clustering. TreeSHAP on 100k × ~2,568
  with ~1,000 trees takes 10–60 min and produces a ~1 GB float32 array
  (100,000 × 2,568 × 4 B ≈ 1.03 GB). The default `--shap-sample 20000`
  keeps memory at ~205 MB and runtime under ~5 min while preserving the
  ranking.
* **RAM:** Cast `X` and the SHAP matrix to `float32`. The SHAP-vector
  correlation matrix at `dim ≈ 2,568` is ~52 MB at float64 (the
  `np.corrcoef` default) or ~26 MB after casting to float32 — both
  trivial.
* **Determinism:** All randomness (train/val split, SHAP subsample) uses
  `random_state=0`. TreeExplainer is itself deterministic.
* **Categorical features** (indices `{2, 3, 4, 5, 6, 701, 702}`) are passed
  to LightGBM via `categorical_feature=CAT_FEATURES` and clustered on their
  SHAP outputs like any other feature — no special-casing needed.
* **Tie-break rules (consolidated):**
  * Cluster representative: highest mean-|SHAP|; ties → **lower** index
    wins (keep the canonical feature).
  * Rank 0 / Rank A / Rank B drop ordering: lowest mean-|SHAP| dropped
    first; ties → **higher** index dropped first.
* **Fallback behavior** when `feature_index_map.json` is missing or
  malformed: emit the CSV with `block="unknown"`, `field=f"Column_{i}"`,
  `hashed=False` — the pipeline still produces a valid drop list keyed on
  `index`, which is all the downstream consumer needs.

---

*Created for: Feature Selection Workflow for EMBER2024*
