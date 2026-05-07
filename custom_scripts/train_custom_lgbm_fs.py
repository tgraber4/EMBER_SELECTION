"""
Train a binary LightGBM malware classifier (as in train_custom_lgbm.py) and then
run the embedded feature-selection plan from
`lightgbm_feature_selection_plan.md`:

  Phase 1 - Train the baseline and time it.
  Phase 2 - Extract per-feature Gain and Split counts; flag zero-utility features.
  Phase 3 - "Final cut" selection: drop absolute zeros first, then top up from
            the bottom of the gain ranking until exactly N_drop features are
            removed. The script NEVER drops more than N_drop (the 10% ceiling
            is a hard cap, even if the absolute-zero set is larger).
  Phase 4 - Print an impact + performance report and save the list of dropped
            feature indices/names to a JSON sidecar next to the model.

Usage:
    python train_custom_lgbm_fs.py <data_dir> <model_path> \
        --config-file ../examples/lgbm_config.json \
        [--drop-fraction 0.10] \
        [--dropped-out <path.json>]

The path to feature_index_map.json is set at the top of this file via the
`FEATURE_INDEX_MAP_PATH` constant -- edit it to point elsewhere.
"""

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from thrember.features import PEFeatureExtractor
from thrember.model import vectorize_subset

CAT_FEATURES = [2, 3, 4, 5, 6, 701, 702]
RANDOM_STATE = 0

# --- User-editable: path to the feature_index_map.json used to translate
# vector indices into human-readable feature names in the dropped-features
# report. Edit this string to point at a different map file. Falls back to
# generic Column_<i> names if the file is missing or its dim doesn't match
# PEFeatureExtractor.dim.
FEATURE_INDEX_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "Documentation" / "feature_index_map.json"
)


# LightGBM rejects feature names that contain any of these JSON-special chars
# (see LightGBM source: src/io/dataset.cpp -- "Do not support special JSON
# characters in feature name."). We replace them with '_' before handing names
# to lgb.Dataset.
_LGB_FORBIDDEN_CHARS = '"\\,:[]{}'
_LGB_FORBIDDEN_TABLE = str.maketrans({c: "_" for c in _LGB_FORBIDDEN_CHARS})


def _sanitize_lgb_name(name: str) -> str:
    return name.translate(_LGB_FORBIDDEN_TABLE)


def load_feature_names(map_path: Path, expected_dim: int) -> list:
    """Build LightGBM-compatible feature names from a feature_index_map.json.

    Each name starts as `"{block}[{block_index}].{field}"` to keep names unique
    across hashed buckets (which share `field` within a block), then has any
    JSON-special characters (which LightGBM rejects) replaced with '_'. If the
    sanitization collapses two names together, we suffix the index to preserve
    uniqueness. Returns generic `Column_<i>` names if the map cannot be used.
    """
    try:
        with map_path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read feature index map at {map_path}: {e}. "
              f"Falling back to Column_<i> names.")
        return [f"Column_{i}" for i in range(expected_dim)]

    map_dim = data.get("dim")
    entries = data.get("entries", [])
    if map_dim != expected_dim or len(entries) != expected_dim:
        print(f"WARNING: feature index map dim mismatch "
              f"(map dim={map_dim}, entries={len(entries)}, extractor dim={expected_dim}). "
              f"Falling back to Column_<i> names.")
        return [f"Column_{i}" for i in range(expected_dim)]

    names = [None] * expected_dim
    try:
        for e in entries:
            i = e["index"]
            if not isinstance(i, int) or not (0 <= i < expected_dim):
                raise ValueError(f"index out of range: {i!r}")
            raw = f"{e['block']}[{e['block_index']}].{e['field']}"
            names[i] = _sanitize_lgb_name(raw)
    except (KeyError, TypeError, ValueError) as err:
        print(f"WARNING: malformed entry in feature index map ({err}). "
              f"Falling back to Column_<i> names.")
        return [f"Column_{i}" for i in range(expected_dim)]
    if any(n is None for n in names):
        print(f"WARNING: feature index map produced missing names. "
              f"Falling back to Column_<i> names.")
        return [f"Column_{i}" for i in range(expected_dim)]
    # Sanitization can collapse e.g. `field[0]` and `field_0_` into the same
    # string. Disambiguate any collisions by appending the original index.
    if len(set(names)) != expected_dim:
        seen = {}
        for i, n in enumerate(names):
            seen.setdefault(n, []).append(i)
        for n, idxs in seen.items():
            if len(idxs) > 1:
                for i in idxs:
                    names[i] = f"{n}__i{i}"
        if len(set(names)) != expected_dim:
            print(f"WARNING: feature index map produced duplicate names even "
                  f"after disambiguation. Falling back to Column_<i> names.")
            return [f"Column_{i}" for i in range(expected_dim)]
    return names


def find_jsonl(data_dir: Path, subset: str) -> Path:
    """Find the unique .jsonl file in data_dir whose name contains `subset`."""
    matches = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix == ".jsonl" and subset in p.name
    )
    if not matches:
        raise ValueError(f"No .jsonl file containing '{subset}' found in {data_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple .jsonl files containing '{subset}' in {data_dir}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def vectorize_split(data_dir: Path, jsonl_path: Path, subset: str,
                    extractor: PEFeatureExtractor) -> None:
    X_path = data_dir / f"X_{subset}.dat"
    y_path = data_dir / f"y_{subset}.dat"
    with jsonl_path.open("r") as f:
        nrows = sum(1 for _ in f)
    if nrows == 0:
        raise ValueError(f"{jsonl_path} is empty")
    print(f"Vectorizing {subset} set ({nrows} rows from {jsonl_path.name}) "
          f"-> {X_path.name}, {y_path.name}")
    vectorize_subset(X_path, y_path, [jsonl_path], extractor, nrows, "label", {})


def read_vectorized(data_dir: Path, subset: str, ndim: int):
    X = np.memmap(data_dir / f"X_{subset}.dat", dtype=np.float32, mode="r")
    X = np.array(X).reshape(-1, ndim)
    y = np.memmap(data_dir / f"y_{subset}.dat", dtype=np.int32, mode="r")
    y = np.array(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Row count mismatch in {subset}: X has {X.shape[0]}, y has {y.shape[0]}"
        )
    return X, y


def assert_binary(y: np.ndarray, name: str) -> None:
    unique = np.unique(y)
    if not np.array_equal(unique, np.array([0, 1], dtype=y.dtype)):
        raise ValueError(
            f"Expected binary labels {{0, 1}} in {name}, got {unique.tolist()}."
        )


def select_features_to_drop(gain: np.ndarray, split: np.ndarray, n_drop: int):
    """Apply the Phase 3 "Final Cut" tiered selection.

    HARD CONTRACT: returns exactly `n_drop` indices. The drop count never
    exceeds the quota even if the absolute-zero set is larger -- the 10%
    ceiling is enforced.

    Returns (dropped_idx, absolute_zero_idx, signal_dropped_idx) where each is a
    sorted np.ndarray of feature indices. `signal_dropped_idx` is the subset
    of `dropped_idx` whose Gain > 0.
    """
    n_features = gain.shape[0]
    if n_drop < 0 or n_drop > n_features:
        raise ValueError(f"n_drop={n_drop} out of range for {n_features} features")

    absolute_zero_mask = (gain == 0) & (split == 0)
    absolute_zero_idx = np.where(absolute_zero_mask)[0]

    if absolute_zero_idx.size >= n_drop:
        # Step 2 (capped): zeros alone meet/exceed the quota. Take the first
        # n_drop absolute zeros by feature index (deterministic) and stop --
        # we never drop more than n_drop, regardless of how many zeros exist.
        dropped_idx = absolute_zero_idx[:n_drop]
    else:
        # Step 3 + 4: rank survivors by ascending Gain and top up to n_drop.
        survivor_mask = ~absolute_zero_mask
        survivors = np.where(survivor_mask)[0]
        # Stable sort by gain ascending; ties broken by feature index.
        order = np.argsort(gain[survivors], kind="stable")
        ranked_survivors = survivors[order]
        needed = n_drop - absolute_zero_idx.size
        top_up = ranked_survivors[:needed]
        dropped_idx = np.sort(np.concatenate([absolute_zero_idx, top_up]))

    if dropped_idx.size != n_drop:
        raise RuntimeError(
            f"Selection bug: dropped {dropped_idx.size} features, expected {n_drop}"
        )
    signal_dropped_idx = dropped_idx[gain[dropped_idx] > 0]
    return dropped_idx, absolute_zero_idx, signal_dropped_idx


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as `HhMmSs` (skipping zero-leading units)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{seconds:.2f}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Folder containing the train and test .jsonl files. "
                             "Vectorized .dat outputs are written here too.")
    parser.add_argument("model_path", type=str,
                        help="Path to save the trained LightGBM model.")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to LightGBM config JSON.")
    parser.add_argument("--early-stopping", type=int, default=50,
                        help="Early-stopping rounds on the val set. 0 disables.")
    parser.add_argument("--drop-fraction", type=float, default=0.10,
                        help="Hard cap on the fraction of features to drop "
                             "(default: 0.10). The script will never drop more "
                             "than ceil(total_features * drop_fraction).")
    parser.add_argument("--dropped-out", type=str, default=None,
                        help="Path to write the dropped-feature report JSON. "
                             "Defaults to <model_path>.dropped_features.json.")
    args = parser.parse_args()

    t_script_start = time.perf_counter()

    if not (0.0 < args.drop_fraction < 1.0):
        raise ValueError(f"--drop-fraction must be in (0, 1); got {args.drop_fraction}")

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
    feature_names = load_feature_names(FEATURE_INDEX_MAP_PATH, extractor.dim)

    vectorize_split(data_dir, train_jsonl, "train", extractor)
    vectorize_split(data_dir, test_jsonl, "test", extractor)

    X_train_full, y_train_full = read_vectorized(data_dir, "train", extractor.dim)
    assert_binary(y_train_full, "training set")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1, stratify=y_train_full, random_state=RANDOM_STATE,
    )
    train_set = lgb.Dataset(
        X_tr, y_tr,
        categorical_feature=CAT_FEATURES, feature_name=feature_names,
    )
    val_set = lgb.Dataset(
        X_val, y_val, reference=train_set,
        categorical_feature=CAT_FEATURES, feature_name=feature_names,
    )

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping))

    # ---------- Phase 1: Training and Time Tracking ----------
    total_features = extractor.dim
    n_drop = int(round(total_features * args.drop_fraction))
    print(f"Phase 1: training LightGBM model "
          f"(features={total_features}, target drop={n_drop} @ {args.drop_fraction:.0%})")
    t_train_start = time.perf_counter()
    model = lgb.train(fit_params, train_set, valid_sets=[val_set], callbacks=callbacks)
    train_seconds = time.perf_counter() - t_train_start

    bi = model.best_iteration
    best_iter = int(bi) if bi is not None and bi > 0 else None
    model.save_model(args.model_path, num_iteration=best_iter)
    print(f"Saved model to {args.model_path}")

    # Score on the held-out test set (unchanged from train_custom_lgbm.py).
    X_test, y_test = read_vectorized(data_dir, "test", extractor.dim)
    assert_binary(y_test, "test set")
    preds = model.predict(X_test, num_iteration=best_iter)
    pred_labels = (preds >= 0.5).astype(np.int32)
    test_acc = float(np.mean(pred_labels == y_test))
    print(f"Test accuracy: {test_acc:.4f} on {len(y_test)} samples")

    # ---------- Phase 2: Metric Extraction and Zero-Value Audit ----------
    t_proc_start = time.perf_counter()
    gain = np.asarray(
        model.feature_importance(importance_type="gain", iteration=best_iter),
        dtype=np.float64,
    )
    split = np.asarray(
        model.feature_importance(importance_type="split", iteration=best_iter),
        dtype=np.int64,
    )
    if gain.shape[0] != total_features or split.shape[0] != total_features:
        raise RuntimeError(
            f"Importance length mismatch: gain={gain.shape[0]}, split={split.shape[0]}, "
            f"expected={total_features}"
        )

    zero_gain_mask = gain == 0
    zero_split_mask = split == 0
    absolute_zero_mask = zero_gain_mask & zero_split_mask
    zero_gain_idx = np.where(zero_gain_mask)[0]
    zero_split_idx = np.where(zero_split_mask)[0]
    absolute_zero_idx = np.where(absolute_zero_mask)[0]

    # Phase 2 Step 3: per-feature table (name -> gain, split, zero-status flags).
    feature_table = [
        {
            "index": int(i),
            "name": feature_names[i],
            "gain": float(gain[i]),
            "split": int(split[i]),
            "zero_gain": bool(zero_gain_mask[i]),
            "zero_split": bool(zero_split_mask[i]),
            "absolute_zero": bool(absolute_zero_mask[i]),
        }
        for i in range(total_features)
    ]

    # ---------- Phase 3: The "Final Cut" Selection Logic ----------
    dropped_idx, _abs_zeros, signal_dropped_idx = select_features_to_drop(
        gain, split, n_drop
    )
    proc_seconds = time.perf_counter() - t_proc_start

    # ---------- Phase 4: Impact Reporting and Performance Audit ----------
    total_gain = float(gain.sum())
    dropped_gain = float(gain[dropped_idx].sum())
    info_loss_pct = (dropped_gain / total_gain * 100.0) if total_gain > 0 else 0.0

    print()
    print("=== Feature Selection Report ===")
    print(f"Total features                : {total_features}")
    print(f"Target drop (N_drop)          : {n_drop}  ({args.drop_fraction:.0%})")
    print(f"0-Gain features               : {zero_gain_idx.size}")
    print(f"0-Split features              : {zero_split_idx.size}")
    print(f"Absolute zeros (Gain & Split) : {absolute_zero_idx.size}")
    print(f"Signal features dropped       : {signal_dropped_idx.size}  "
          f"(features with Gain > 0 removed to meet the quota)")
    print(f"Total global gain             : {total_gain:.6g}")
    print(f"Sum of dropped gain           : {dropped_gain:.6g}")
    print(f"Information cost              : {info_loss_pct:.4f}%")
    print(f"Training time                 : {train_seconds:.2f} s")
    print(f"Selection processing time     : {proc_seconds:.4f} s")

    # Persist the dropped-feature decision for downstream consumption by
    # thrember_lite.FeatureSpec.from_drop_columns.
    out_path = Path(args.dropped_out) if args.dropped_out else Path(
        f"{args.model_path}.dropped_features.json"
    )
    report = {
        "total_features": int(total_features),
        "drop_fraction": float(args.drop_fraction),
        "n_drop": int(n_drop),
        "counts": {
            "zero_gain": int(zero_gain_idx.size),
            "zero_split": int(zero_split_idx.size),
            "absolute_zero": int(absolute_zero_idx.size),
            "signal_dropped": int(signal_dropped_idx.size),
        },
        "information_cost_pct": info_loss_pct,
        "total_gain": total_gain,
        "dropped_gain": dropped_gain,
        "train_seconds": train_seconds,
        "selection_seconds": proc_seconds,
        "total_script_seconds": time.perf_counter() - t_script_start,
        "test_accuracy": test_acc,
        "best_iteration": best_iter,
        "dropped_indices": dropped_idx.tolist(),
        "dropped_feature_names": [feature_names[i] for i in dropped_idx.tolist()],
        "absolute_zero_indices": absolute_zero_idx.tolist(),
        "signal_dropped_indices": signal_dropped_idx.tolist(),
        "feature_table": feature_table,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved dropped-feature report to {out_path}")

    total_script_seconds = time.perf_counter() - t_script_start
    print(f"Total script runtime          : "
          f"{total_script_seconds:.2f} s ({_format_duration(total_script_seconds)})")


if __name__ == "__main__":
    main()
