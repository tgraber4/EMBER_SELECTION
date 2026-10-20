"""
Train a binary LightGBM malware classifier on a folder containing two JSONL
files -- one with 'train' in the filename, one with 'test' -- using thrember's
PEFeatureExtractor.

Each JSONL record must contain raw thrember PE features plus an integer
`label` field (0 = benign, 1 = malicious). Vectorized .dat outputs are written
alongside the JSONLs in <data_dir>.

Usage:
    python train_custom_lgbm.py <data_dir> <model_path> \
        --config-file ../examples/lgbm_config.json
"""

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from thrember.features import PEFeatureExtractor
from thrember.model import vectorize_subset

CAT_FEATURES = [2, 3, 4, 5, 6, 701, 702]
RANDOM_STATE = 0


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
    args = parser.parse_args()

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

    vectorize_split(data_dir, train_jsonl, "train", extractor)
    vectorize_split(data_dir, test_jsonl, "test", extractor)

    X_train_full, y_train_full = read_vectorized(data_dir, "train", extractor.dim)
    assert_binary(y_train_full, "training set")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1, stratify=y_train_full, random_state=RANDOM_STATE,
    )
    train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_FEATURES)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=CAT_FEATURES)

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping))

    print("Training LightGBM model")
    model = lgb.train(fit_params, train_set, valid_sets=[val_set], callbacks=callbacks)

    bi = model.best_iteration
    best_iter = int(bi) if bi is not None and bi > 0 else None
    model.save_model(args.model_path, num_iteration=best_iter)
    print(f"Saved model to {args.model_path}")

    X_test, y_test = read_vectorized(data_dir, "test", extractor.dim)
    assert_binary(y_test, "test set")

    preds = model.predict(X_test, num_iteration=best_iter)
    pred_labels = (preds >= 0.5).astype(np.int32)
    acc = float(np.mean(pred_labels == y_test))
    print(f"Test accuracy: {acc:.4f} on {len(y_test)} samples")


if __name__ == "__main__":
    main()
