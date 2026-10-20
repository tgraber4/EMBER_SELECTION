"""Integration smoke + determinism check for train_binary and ModelBundle.

Per plan §Validation strategy:
  - Synthetic .dat (small), train 10 iters, returns lgb.Booster.
  - ModelBundle.save → load round-trips.
  - predict scores in [0, 1].
  - Determinism check: two runs with seed=42 produce identical predictions
    when num_threads=1 and force_row_wise=True.

Note: predict_file is NOT tested here — it requires a real PE byte stream and
PEFeatureExtractor; that's reserved for the end-to-end validation step.
The booster is exercised directly via booster.predict on numpy arrays.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import lightgbm as lgb

from thrember_lite.spec import FeatureSpec
from thrember_lite.train import train_binary, _inject_seeds
from thrember_lite.predict import ModelBundle


# ---- helpers ---------------------------------------------------------------

def _make_synthetic_train_dir(td: Path, *, n_rows: int, original_dim: int, seed: int) -> None:
    """Build X_train.dat and y_train.dat with a learnable signal in column 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, original_dim)).astype(np.float32)
    # Make column 0 carry the label signal so any 10-iter LightGBM picks it up
    y = (X[:, 0] > 0).astype(np.int32)
    # Sprinkle a few unlabeled rows to exercise the y==-1 filter in train_binary
    y[::25] = -1
    X.tofile(td / "X_train.dat")
    y.tofile(td / "y_train.dat")


def _toy_spec(original_dim: int, kept: list[int]) -> FeatureSpec:
    return FeatureSpec(
        original_dim=original_dim,
        kept_indices=kept,
        original_categorical=[],
        block_ranges={"toy": (0, original_dim)},
    )


def _smoke_params() -> dict:
    """Minimal LightGBM params for fast smoke runs."""
    return {
        "objective": "binary",
        "verbose": -1,
        "num_iterations": 10,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "num_threads": 1,
        "force_row_wise": True,
    }


# ---- _inject_seeds (unit) --------------------------------------------------

def test_inject_seeds_does_not_mutate_caller():
    params = {"objective": "binary"}
    out = _inject_seeds(params, 42)
    assert "seed" not in params
    assert out is not params
    assert out["seed"] == 42
    assert out["bagging_seed"] == 42
    assert out["feature_fraction_seed"] == 42


def test_inject_seeds_overwrites_existing_keys():
    params = {"seed": 1, "bagging_seed": 2}
    out = _inject_seeds(params, 99)
    assert out["seed"] == 99
    assert out["bagging_seed"] == 99
    assert params["seed"] == 1                  # caller still untouched
    assert params["bagging_seed"] == 2


# ---- integration smoke ----------------------------------------------------

def test_train_binary_returns_booster_and_predicts_in_range():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_synthetic_train_dir(td, n_rows=500, original_dim=8, seed=0)
        spec = _toy_spec(8, [0, 1, 2, 3, 4, 5, 6, 7])

        booster = train_binary(td, spec, _smoke_params(), seed=42)

    assert isinstance(booster, lgb.Booster)

    # Score a fixed input and check range
    rng = np.random.default_rng(123)
    X_check = rng.standard_normal((20, 8)).astype(np.float32)
    scores = booster.predict(X_check)
    assert scores.shape == (20,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_train_binary_with_dropped_columns_uses_correct_width():
    """Slicing the matrix down to keep=[0, 2, 5] should produce a booster that
    accepts 3-column inputs, not 8."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_synthetic_train_dir(td, n_rows=500, original_dim=8, seed=0)
        spec = _toy_spec(8, [0, 2, 5])

        booster = train_binary(td, spec, _smoke_params(), seed=42)

    assert booster.num_feature() == 3

    rng = np.random.default_rng(7)
    X_check = rng.standard_normal((10, 3)).astype(np.float32)
    scores = booster.predict(X_check)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ---- ModelBundle round-trip ----------------------------------------------

def test_model_bundle_save_load_round_trip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_dir = td / "data"
        data_dir.mkdir()
        _make_synthetic_train_dir(data_dir, n_rows=500, original_dim=8, seed=0)
        spec = _toy_spec(8, [0, 1, 2, 3, 4, 5, 6, 7])

        booster = train_binary(data_dir, spec, _smoke_params(), seed=42)

        out_dir = td / "bundle"
        ModelBundle.save(booster, spec, out_dir)
        assert (out_dir / "model.txt").is_file()
        assert (out_dir / "spec.json").is_file()

        # Re-load — but note ModelBundle.load asserts extractor.dim == original_dim,
        # which would fail because original_dim=8 != PEFeatureExtractor().dim=2568.
        # Bypass via direct construction for this round-trip test; predict_file
        # exercises the full load path against a real spec.
        from lightgbm import Booster
        b2 = Booster(model_file=str(out_dir / "model.txt"))
        s2 = FeatureSpec.from_json(out_dir / "spec.json")

    assert s2.original_dim == spec.original_dim
    assert s2.kept_indices == spec.kept_indices
    # Predictions on a fixed input should match (loading round-trip preserves trees)
    rng = np.random.default_rng(0)
    X_check = rng.standard_normal((30, 8)).astype(np.float32)
    assert np.allclose(booster.predict(X_check), b2.predict(X_check))


# ---- determinism check ----------------------------------------------------

def test_determinism_with_seed_and_single_thread():
    """Plan §Validation: two runs with seed=42 produce equal predictions
    when num_threads=1 and force_row_wise=True. Multithreaded runs are
    floating-point-order sensitive in LightGBM and may not be bytewise
    identical even with all RNG seeds pinned."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_synthetic_train_dir(td, n_rows=300, original_dim=8, seed=0)
        spec = _toy_spec(8, [0, 1, 2, 3, 4, 5, 6, 7])

        b1 = train_binary(td, spec, _smoke_params(), seed=42)
        b2 = train_binary(td, spec, _smoke_params(), seed=42)

    rng = np.random.default_rng(0)
    X_check = rng.standard_normal((50, 8)).astype(np.float32)
    p1 = b1.predict(X_check)
    p2 = b2.predict(X_check)
    # Primary: prediction equality (robust to model_to_string format quirks)
    assert np.array_equal(p1, p2), \
        f"determinism check failed: max abs diff = {np.max(np.abs(p1 - p2)):.6e}"


def test_train_binary_rejects_non_binary_labels():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Three classes — train_binary should refuse
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4)).astype(np.float32)
        y = rng.integers(0, 3, size=200).astype(np.int32)
        X.tofile(td / "X_train.dat")
        y.tofile(td / "y_train.dat")
        spec = _toy_spec(4, [0, 1, 2, 3])

        try:
            train_binary(td, spec, _smoke_params(), seed=42)
        except ValueError as e:
            assert "binary" in str(e).lower()
            return
    raise AssertionError("expected ValueError for 3-class labels")


# ---- runner ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    failures = 0
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
