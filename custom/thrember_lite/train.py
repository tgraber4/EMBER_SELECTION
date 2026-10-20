"""Binary LightGBM training on a sliced EMBER vector.

Mirrors `thrember.model.train_model` (model.py:347-379) line-for-line:
 - calls read_vectorized_features
 - filters y == -1 in the trainer (model.py:360-361)
 - stratified 90/10 train/val split
 - lgb.Dataset + lgb.train

Differences (per the plan):
 (a) takes a FeatureSpec; uses spec.new_categorical instead of the hardcoded list
 (b) seed plumbing to both train_test_split and LightGBM's three RNG knobs
 (c) no multiclass branch — multilabel y already raised inside read_vectorized_features

Returns lgb.Booster directly (Decision #12). Persistence is the caller's job via
ModelBundle.save in predict.py.

NOTE: Family/multiclass would go in a future train_family() module — out of scope for v1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from .spec import FeatureSpec
from .data import read_vectorized_features


# LightGBM RNG knobs to pin together (the master `seed` doesn't override these
# if they're already in `params`). See plan §train.py and the seed-plumbing decision.
_SEED_KEYS = ("seed", "bagging_seed", "feature_fraction_seed")


def train_binary(
    data_dir: Path | str,
    spec: FeatureSpec,
    params: dict = {},                       # noqa: B006 — mirrors thrember.train_model
    val_size: float = 0.1,
    seed: int | None = None,
) -> lgb.Booster:
    """Train a binary LightGBM model on a sliced EMBER train set.

    `params` is passed to `lgb.train` verbatim except for the seed knobs (which
    are injected into a copy when `seed` is set). Mutable default mirrors
    `thrember.train_model` (model.py:347); `_inject_seeds` always copies before
    mutating, so the shared default is never written to.

    `seed=None` reproduces thrember's non-determinism. For ablation studies, set
    a fixed seed across configs so AUC deltas are attributable to the feature set.
    """
    # Load (chunked + sliced) — peak RAM is ~len(kept) × N × 4 bytes
    X, y = read_vectorized_features(data_dir, "train", spec)

    # Drop unlabeled rows — same lines as model.py:360-361
    X = X[y != -1, :]
    y = y[y != -1]

    # Validations (timing matters: must be after the y==-1 filter)
    if len(np.unique(y)) != 2:
        raise ValueError(
            f"train_binary requires binary labels; got {len(np.unique(y))} classes "
            f"({sorted(np.unique(y).tolist())})"
        )
    if X.shape[1] != len(spec.kept_indices):
        raise ValueError(
            f"spec/data column mismatch: X has {X.shape[1]} cols, "
            f"spec.kept_indices has {len(spec.kept_indices)}"
        )

    # Stratified 90/10 split — random_state=seed plumbs the val-split RNG
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed,
    )

    # Pin LightGBM's three RNG knobs on a copy of params
    if seed is not None:
        params = _inject_seeds(params, seed)

    train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=spec.new_categorical)
    val_set = lgb.Dataset(
        X_val, y_val, reference=train_set,
        categorical_feature=spec.new_categorical,
    )

    return lgb.train(params, train_set, valid_sets=val_set)


def _inject_seeds(params: dict, seed: int) -> dict:
    """Return a copy of `params` with the three LightGBM RNG knobs set to `seed`.

    Print a warning if any are being overwritten — silently clobbering a
    user-set seed is the kind of thing that causes "why don't my results match"
    bugs months later.
    """
    out = dict(params)  # shallow copy is enough; values are scalars
    for key in _SEED_KEYS:
        if key in out and out[key] != seed:
            print(
                f"thrember_lite: overwriting params[{key!r}]={out[key]!r} with seed={seed}"
            )
        out[key] = seed
    return out
