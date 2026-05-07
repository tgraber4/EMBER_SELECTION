"""
SHAP + Correlation-Clustering feature selection for EMBER2024.

Implements EMBER2024_Feature_Selection_Plan-v4.md. Trains a baseline LightGBM
binary classifier on the provided <data_dir>, then uses TreeSHAP attributions
to (1) rank features by mean-|SHAP| and (2) hierarchically cluster features by
the Pearson correlation of their per-sample SHAP-contribution vectors. The
bottom `--drop-fraction` (default 10%) of features is selected via a three-tier
ranking (inactive -> redundant clones -> low importance) and written to a CSV
that is consumed downstream by `thrember_lite` (e.g.
`custom_scripts/run_thrember_lite.py`) as a separate manual step.

Built off `train_custom_lgbm.py` -- reuses `find_jsonl`, `vectorize_split`,
`read_vectorized`, `assert_binary`, `CAT_FEATURES`, and `RANDOM_STATE` from it
so the data-prep / training side stays in lock-step with the baseline pipeline.

Usage:
    python custom_scripts/shap_cluster_feature_selection.py data runs/baseline_model.txt --config-file examples/lgbm_config.json --drop-fraction 0.10 --shap-sample 20000 --cluster-threshold 0.10 --feature-map Documentation/feature_index_map.json --dropped-out runs/dropped_features.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from thrember.features import PEFeatureExtractor

# Reuse the data-prep helpers from train_custom_lgbm.py so this script stays
# locked to the same baseline conventions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_custom_lgbm import (  # noqa: E402
    CAT_FEATURES,
    RANDOM_STATE,
    assert_binary,
    find_jsonl,
    read_vectorized,
    vectorize_split,
)


DEFAULT_FEATURE_MAP = (
    Path(__file__).resolve().parent.parent / "Documentation" / "feature_index_map.json"
)


def _format_duration(seconds: float) -> str:
    """Render a wall-clock duration as `Hh Mm Ss` (or shorter), with seconds."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s ({seconds:.2f}s)"
    if m:
        return f"{m}m {s}s ({seconds:.2f}s)"
    return f"{seconds:.2f}s"


# --------------------------------------------------------------------------- #
# Feature index map                                                           #
# --------------------------------------------------------------------------- #

def _stub_index_map(dim: int) -> dict:
    """Generic fallback when feature_index_map.json is missing/malformed."""
    return {
        i: {"block": "unknown", "field": f"Column_{i}", "hashed": False}
        for i in range(dim)
    }


def load_index_map(map_path: Path, expected_dim: int) -> dict:
    """Load feature_index_map.json -> {index: {block, field, hashed, ...}}.

    On any error (missing file, bad JSON, dim mismatch, malformed entries),
    print a warning and fall back to stub entries so the pipeline still
    produces a valid CSV keyed on `index`.
    """
    try:
        with map_path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read feature index map at {map_path}: {e}. "
              f"Falling back to stub block/field/hashed.")
        return _stub_index_map(expected_dim)

    map_dim = data.get("dim")
    entries = data.get("entries", [])
    if map_dim != expected_dim or len(entries) != expected_dim:
        print(f"WARNING: feature index map dim mismatch "
              f"(map dim={map_dim}, entries={len(entries)}, "
              f"extractor dim={expected_dim}). Falling back to stub entries.")
        return _stub_index_map(expected_dim)

    try:
        index_map = {int(e["index"]): e for e in entries}
    except (KeyError, TypeError, ValueError) as err:
        print(f"WARNING: malformed entry in feature index map ({err}). "
              f"Falling back to stub entries.")
        return _stub_index_map(expected_dim)

    if set(index_map.keys()) != set(range(expected_dim)):
        print(f"WARNING: feature index map indices do not cover 0..{expected_dim-1}. "
              f"Falling back to stub entries.")
        return _stub_index_map(expected_dim)
    return index_map


# --------------------------------------------------------------------------- #
# SHAP helpers                                                                #
# --------------------------------------------------------------------------- #

def stratified_subsample(X, y, n, random_state):
    """Return (X_sub, y_sub) of size `n`, stratified on `y`.

    StratifiedShuffleSplit rejects train_size == n_samples (test fold would be
    empty), so we short-circuit when the request meets/exceeds the population.
    """
    if n >= len(X):
        return X, y
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n, random_state=random_state
    )
    idx, _ = next(splitter.split(X, y))
    return X[idx], y[idx]


def normalize_shap_shape(shap_values, n: int, dim: int) -> np.ndarray:
    """Normalize SHAP's binary-classifier return shape to (n, dim) float32.

    SHAP's API differs across versions:
      * list of length 2 -> [shap_class0, shap_class1]; use class 1.
      * ndarray (n, dim) -> use directly.
      * ndarray (n, dim, 2) -> use [..., 1].
    """
    if isinstance(shap_values, list):
        if len(shap_values) != 2:
            raise ValueError(
                f"Expected SHAP list of length 2 for binary model, got {len(shap_values)}."
            )
        S = shap_values[1]
    else:
        S = shap_values
        if S.ndim == 3:
            if S.shape[-1] != 2:
                raise ValueError(
                    f"Expected SHAP last-axis size 2 for binary model, got shape {S.shape}."
                )
            S = S[..., 1]

    if S.shape != (n, dim):
        raise AssertionError(
            f"SHAP matrix has shape {S.shape}, expected {(n, dim)}."
        )
    return np.ascontiguousarray(S, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Clustering                                                                  #
# --------------------------------------------------------------------------- #

def correlation_distance(S_active: np.ndarray) -> np.ndarray:
    """Pearson correlation of SHAP columns -> Euclidean distance sqrt(2(1-R)).

    Cast to float32 immediately to keep the dim x dim matrix half the size.
    NaN/inf entries (undefined correlation) are treated as orthogonal.
    Returns a sanitized, symmetric distance matrix with zero diagonal,
    clipped to [0, 2].
    """
    R = np.corrcoef(S_active, rowvar=False).astype(np.float32, copy=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = 0.5 * (R + R.T)
    Dist = np.sqrt(np.clip(2.0 * (1.0 - R), 0.0, None)).astype(np.float32, copy=False)
    np.fill_diagonal(Dist, 0.0)
    np.clip(Dist, 0.0, 2.0, out=Dist)
    return Dist


def cluster_active_features(
    Dist: np.ndarray, threshold: float
) -> np.ndarray:
    """Average-linkage hierarchical clustering on a precomputed distance matrix.

    Average linkage gives a dendrogram whose heights are real mean pairwise
    distances, so `t` cuts in the same units as `Dist`. With
    `Dist = sqrt(2(1-R))`, t=0.10 corresponds to mean correlation >= 0.995.
    Returns labels in 1..K (always positive) over the active features.
    """
    condensed = squareform(Dist, checks=False)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=threshold, criterion="distance")


# --------------------------------------------------------------------------- #
# Selection                                                                   #
# --------------------------------------------------------------------------- #

def build_drop_list(
    dim: int,
    mean_abs_shap: np.ndarray,
    cluster_id: np.ndarray,
    inactive_mask: np.ndarray,
    n_drop: int,
):
    """Apply the Phase 3 three-tier ranking and return ordered drop records.

    Tiers (dropped in this order):
      Rank 0 -- inactive features (cluster_id == -1), sorted by |SHAP| asc.
      Rank A -- redundant clones (non-representative members of clusters of
                size > 1), sorted by |SHAP| asc.
      Rank B -- cluster representatives + active singletons, sorted by
                |SHAP| asc.

    Within each tier, ties on |SHAP| are broken by HIGHER feature index dropped
    first (i.e. canonical / earlier indices are kept when |SHAP| ties).
    Cluster-representative selection (max |SHAP|) breaks ties by LOWER feature
    index winning -- canonical feature stays.

    Returns a list of dicts:
      {index, mean_abs_shap, cluster_id, drop_reason}
    truncated/padded to exactly n_drop entries.
    """
    # -- Identify representatives within each active cluster -----------------
    redundant = np.zeros(dim, dtype=bool)
    active_idx = np.where(~inactive_mask)[0]
    # Group active indices by cluster id.
    clusters: dict[int, list[int]] = {}
    for i in active_idx:
        clusters.setdefault(int(cluster_id[i]), []).append(int(i))

    for members in clusters.values():
        if len(members) <= 1:
            continue
        # Representative: highest mean-|SHAP|; tie -> lower index wins.
        # argmax with the (-shap, index) key keeps lowest index on ties.
        rep = min(members, key=lambda j: (-mean_abs_shap[j], j))
        for j in members:
            if j != rep:
                redundant[j] = True

    # -- Sort each tier (ties: higher index dropped first) -------------------
    # Sort key for "drop first" ascending: (mean_abs_shap, -index).
    def _drop_key(j: int):
        return (float(mean_abs_shap[j]), -int(j))

    rank0 = sorted(np.where(inactive_mask)[0].tolist(), key=_drop_key)
    rankA = sorted(np.where(redundant)[0].tolist(), key=_drop_key)
    rep_or_singleton = sorted(
        [int(j) for j in active_idx if not redundant[j]],
        key=_drop_key,
    )

    # -- Pull from tiers in order until we hit n_drop -----------------------
    records = []
    for j in rank0:
        if len(records) >= n_drop:
            break
        records.append({
            "index": int(j),
            "mean_abs_shap": float(mean_abs_shap[j]),
            "cluster_id": int(cluster_id[j]),
            "drop_reason": "inactive",
        })
    for j in rankA:
        if len(records) >= n_drop:
            break
        records.append({
            "index": int(j),
            "mean_abs_shap": float(mean_abs_shap[j]),
            "cluster_id": int(cluster_id[j]),
            "drop_reason": "redundant",
        })
    for j in rep_or_singleton:
        if len(records) >= n_drop:
            break
        records.append({
            "index": int(j),
            "mean_abs_shap": float(mean_abs_shap[j]),
            "cluster_id": int(cluster_id[j]),
            "drop_reason": "low_importance",
        })

    return records


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP + correlation-clustering feature selection for EMBER2024."
    )
    parser.add_argument("data_dir", type=str,
                        help="Folder containing the train and test .jsonl files. "
                             "Vectorized .dat outputs are written here too.")
    parser.add_argument("model_path", type=str,
                        help="Path to save the trained baseline LightGBM model.")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to LightGBM config JSON.")
    parser.add_argument("--early-stopping", type=int, default=50,
                        help="Early-stopping rounds on the val set. 0 disables.")
    parser.add_argument("--drop-fraction", type=float, default=0.10,
                        help="Fraction of features to drop (default 0.10).")
    parser.add_argument("--shap-sample", type=int, default=20000,
                        help="Stratified subsample size for TreeSHAP "
                             "(default 20000). Capped at len(X_tr).")
    parser.add_argument("--cluster-threshold", type=float, default=0.10,
                        help="fcluster distance threshold (default 0.10 ~ R>=0.995). "
                             "Must be in (0, sqrt(2)].")
    parser.add_argument("--shap-var-eps", type=float, default=1e-12,
                        help="Variance threshold below which a SHAP column is "
                             "treated as inactive (default 1e-12).")
    parser.add_argument("--feature-map", type=str, default=str(DEFAULT_FEATURE_MAP),
                        help="Path to feature_index_map.json (default: "
                             "../Documentation/feature_index_map.json).")
    parser.add_argument("--dropped-out", type=str, default="dropped_features.csv",
                        help="Output CSV of dropped features "
                             "(default dropped_features.csv).")
    args = parser.parse_args()

    # Validate cluster-threshold up-front per the plan.
    sqrt2 = math.sqrt(2.0)
    if not (0 < args.cluster_threshold <= sqrt2):
        parser.error(
            f"--cluster-threshold must be in (0, sqrt(2)~={sqrt2:.6f}], "
            f"got {args.cluster_threshold}."
        )
    if not (0 < args.drop_fraction < 1):
        parser.error(
            f"--drop-fraction must be in (0, 1), got {args.drop_fraction}."
        )
    if args.shap_sample <= 0:
        parser.error(f"--shap-sample must be positive, got {args.shap_sample}.")
    return args


def main():
    args = parse_args()

    # Per-phase wall-clock timings for the end-of-run report.
    timings: dict[str, float] = {}
    t_run_start = time.perf_counter()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")
    if not os.path.isfile(args.config_file):
        raise ValueError(f"Not a file: {args.config_file}")

    train_jsonl = find_jsonl(data_dir, "train")
    test_jsonl = find_jsonl(data_dir, "test")

    with open(args.config_file, "r") as f:
        fit_params = json.load(f)
    if fit_params.get("objective", "binary") != "binary":
        raise ValueError("This script is binary-only; config 'objective' must be 'binary'.")

    extractor = PEFeatureExtractor()
    dim = extractor.dim
    print(f"PEFeatureExtractor dim = {dim}")

    # ---- Phase 1: data ingestion & baseline -------------------------------
    t_vec_start = time.perf_counter()
    vectorize_split(data_dir, train_jsonl, "train", extractor)
    vectorize_split(data_dir, test_jsonl, "test", extractor)

    X_train_full, y_train_full = read_vectorized(data_dir, "train", dim)
    assert_binary(y_train_full, "training set")
    timings["vectorize_and_load"] = time.perf_counter() - t_vec_start

    # 90/10 stratified split, random_state=0 to match train_custom_lgbm.py.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1, stratify=y_train_full, random_state=RANDOM_STATE,
    )

    train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_FEATURES)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set,
                          categorical_feature=CAT_FEATURES)

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping))

    print("Training baseline LightGBM model")
    t_train_start = time.perf_counter()
    model = lgb.train(fit_params, train_set, valid_sets=[val_set], callbacks=callbacks)
    timings["baseline_training"] = time.perf_counter() - t_train_start

    # Defensive best_iter handling -- LightGBM uses -1 to mean "early stopping
    # never fired"; we don't want that leaking into tree_limit / num_iteration.
    bi = model.best_iteration
    best_iter = int(bi) if bi is not None and bi > 0 else None
    model.save_model(args.model_path, num_iteration=best_iter)
    print(f"Saved baseline model to {args.model_path} (best_iter={best_iter}).")

    # Baseline metrics on the full test set.
    t_eval_start = time.perf_counter()
    X_test, y_test = read_vectorized(data_dir, "test", dim)
    assert_binary(y_test, "test set")
    test_proba = model.predict(X_test, num_iteration=best_iter)
    test_pred = (test_proba >= 0.5).astype(np.int32)
    baseline_acc = float(np.mean(test_pred == y_test))
    baseline_auc = float(roc_auc_score(y_test, test_proba))
    baseline_ll = float(log_loss(y_test, test_proba, labels=[0, 1]))
    print(f"Baseline test metrics  acc={baseline_acc:.4f}  "
          f"AUC={baseline_auc:.4f}  log_loss={baseline_ll:.4f}  "
          f"({len(y_test)} samples)")
    timings["baseline_eval"] = time.perf_counter() - t_eval_start

    # ---- Phase 2: SHAP + correlation clustering ---------------------------
    t_shap_start = time.perf_counter()
    n_request = min(int(args.shap_sample), len(X_tr))
    X_sub, _y_sub = stratified_subsample(X_tr, y_tr, n_request, RANDOM_STATE)
    n = len(X_sub)
    print(f"Computing TreeSHAP on {n:,} samples x {dim} features "
          f"(tree_limit={best_iter})...")

    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent"
    )
    raw_shap = explainer.shap_values(
        X_sub, tree_limit=best_iter, check_additivity=False
    )
    S = normalize_shap_shape(raw_shap, n, dim)
    del raw_shap
    timings["shap_compute"] = time.perf_counter() - t_shap_start

    mean_abs_shap = np.mean(np.abs(S), axis=0).astype(np.float32, copy=False)
    var_S = S.var(axis=0)
    inactive_mask = var_S < args.shap_var_eps
    n_inactive = int(inactive_mask.sum())
    n_active = dim - n_inactive
    print(f"Inactive (SHAP-variance < {args.shap_var_eps}) features: "
          f"{n_inactive}/{dim}; active: {n_active}.")

    # Default cluster labels: -1 everywhere; active features get overwritten.
    cluster_id = np.full(dim, -1, dtype=np.int64)

    t_cluster_start = time.perf_counter()
    if n_active >= 2:
        active_idx = np.where(~inactive_mask)[0]
        S_active = S[:, active_idx]
        print(f"Computing Pearson correlation on SHAP columns "
              f"({n_active} x {n_active})...")
        Dist = correlation_distance(S_active)
        print(f"Hierarchical clustering (average linkage, "
              f"t={args.cluster_threshold})...")
        labels_active = cluster_active_features(Dist, args.cluster_threshold)
        cluster_id[active_idx] = labels_active.astype(np.int64, copy=False)
        n_clusters = int(labels_active.max())
        print(f"  -> {n_clusters} cluster(s) across {n_active} active features.")
        del Dist, S_active
    elif n_active == 1:
        # Single active feature: trivial singleton cluster.
        cluster_id[~inactive_mask] = 1
        print("Only one active feature -- clustering skipped.")
    else:
        print("No active features -- clustering skipped.")

    del S
    timings["clustering"] = time.perf_counter() - t_cluster_start

    # ---- Phase 3: bottom-X% selection -------------------------------------
    t_select_start = time.perf_counter()
    n_drop = int(round(dim * args.drop_fraction))
    n_drop = max(0, min(n_drop, dim))
    print(f"Selecting {n_drop} features to drop "
          f"(drop_fraction={args.drop_fraction}).")

    records = build_drop_list(
        dim=dim,
        mean_abs_shap=mean_abs_shap,
        cluster_id=cluster_id,
        inactive_mask=inactive_mask,
        n_drop=n_drop,
    )
    if len(records) < n_drop:
        # Should be impossible -- the three tiers cover every feature index.
        print(f"WARNING: only {len(records)} drop candidates produced; "
              f"requested {n_drop}.")

    timings["selection"] = time.perf_counter() - t_select_start

    # ---- Export CSV -------------------------------------------------------
    t_export_start = time.perf_counter()
    index_map = load_index_map(Path(args.feature_map), dim)

    out_path = Path(args.dropped_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank", "index", "block", "field", "hashed",
        "mean_abs_shap", "cluster_id", "drop_reason",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, rec in enumerate(records, start=1):
            entry = index_map.get(rec["index"], {})
            writer.writerow({
                "rank": rank,
                "index": rec["index"],
                "block": entry.get("block", "unknown"),
                "field": entry.get("field", f"Column_{rec['index']}"),
                "hashed": bool(entry.get("hashed", False)),
                "mean_abs_shap": f"{rec['mean_abs_shap']:.10g}",
                "cluster_id": rec["cluster_id"],
                "drop_reason": rec["drop_reason"],
            })

    timings["csv_export"] = time.perf_counter() - t_export_start

    total_runtime = time.perf_counter() - t_run_start

    # Console summary.
    by_reason = {"inactive": 0, "redundant": 0, "low_importance": 0}
    for r in records:
        by_reason[r["drop_reason"]] = by_reason.get(r["drop_reason"], 0) + 1
    print()
    print(f"Dropped {len(records)} features -> {out_path}")
    print(f"  inactive       : {by_reason['inactive']}")
    print(f"  redundant      : {by_reason['redundant']}")
    print(f"  low_importance : {by_reason['low_importance']}")
    print()
    print("Runtime breakdown:")
    phase_labels = [
        ("vectorize_and_load", "vectorize & load   "),
        ("baseline_training",  "baseline training  "),
        ("baseline_eval",      "baseline eval      "),
        ("shap_compute",       "SHAP compute       "),
        ("clustering",         "clustering         "),
        ("selection",          "selection          "),
        ("csv_export",         "CSV export         "),
    ]
    for key, label in phase_labels:
        if key in timings:
            print(f"  {label}: {_format_duration(timings[key])}")
    print(f"  {'total wall-clock   '}: {_format_duration(total_runtime)}")
    print()
    print("Next step: feed this CSV (or its .json sibling) to thrember_lite "
          "via custom_scripts/run_thrember_lite.py (separate manual step).")


if __name__ == "__main__":
    main()
