"""data.py unit tests — synthetic .dat files, no thrember dependency.

Validates the chunked path, in_memory path, no-internal-filtering, the 1D-y
assertion, and `apply_spec` shape correctness.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from thrember_lite.spec import FeatureSpec
from thrember_lite.data import read_vectorized_features, apply_spec


# ---- helpers ---------------------------------------------------------------

def _write_synthetic_dat(td: Path, subset: str, X: np.ndarray, y: np.ndarray) -> None:
    """Write float32 X and int32 y as raw .dat files in thrember's layout."""
    X.astype(np.float32, copy=False).tofile(td / f"X_{subset}.dat")
    y.astype(np.int32, copy=False).tofile(td / f"y_{subset}.dat")


def _toy_spec(original_dim: int, kept: list[int]) -> FeatureSpec:
    return FeatureSpec(
        original_dim=original_dim,
        kept_indices=kept,
        original_categorical=[],
        block_ranges={},
    )


# ---- chunked path ----------------------------------------------------------

def test_chunked_load_slices_columns():
    rng = np.random.default_rng(0)
    N, D = 200, 8
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.integers(0, 2, size=N).astype(np.int32)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_synthetic_dat(td, "train", X, y)
        spec = _toy_spec(D, [0, 2, 5, 7])

        # Force chunking with a small block to exercise the loop boundary
        X_out, y_out = read_vectorized_features(td, "train", spec, block=37)

    assert X_out.shape == (N, 4)
    assert X_out.dtype == np.float32
    assert X_out.flags["C_CONTIGUOUS"]
    assert np.array_equal(X_out, X[:, [0, 2, 5, 7]])
    assert np.array_equal(y_out, y)


def test_in_memory_path_matches_chunked():
    rng = np.random.default_rng(1)
    N, D = 150, 12
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.integers(-1, 2, size=N).astype(np.int32)   # includes -1 rows

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_synthetic_dat(td, "test", X, y)
        spec = _toy_spec(D, [1, 3, 11])

        X_chunk, y_chunk = read_vectorized_features(td, "test", spec, block=20)
        X_mem, y_mem = read_vectorized_features(td, "test", spec, in_memory=True)

    assert np.array_equal(X_chunk, X_mem)
    assert np.array_equal(y_chunk, y_mem)


# ---- loader does not filter ------------------------------------------------

def test_loader_returns_unfiltered_y():
    """Per Decision #7: the loader keeps all rows, including y == -1."""
    N, D = 50, 4
    X = np.zeros((N, D), dtype=np.float32)
    y = np.full(N, -1, dtype=np.int32)        # all rows unlabeled
    y[::2] = 0                                 # half labeled

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_synthetic_dat(td, "train", X, y)
        spec = _toy_spec(D, [0, 1, 2, 3])

        X_out, y_out = read_vectorized_features(td, "train", spec)

    assert X_out.shape[0] == N
    assert y_out.shape[0] == N
    assert int((y_out == -1).sum()) == N // 2


# ---- 1D-y assertion --------------------------------------------------------

def test_multilabel_y_is_rejected():
    """Multilabel y is stored flat with length N*K. The loader must refuse it."""
    N, D, K = 30, 4, 3
    X = np.zeros((N, D), dtype=np.float32)
    y_flat = np.zeros(N * K, dtype=np.int32)   # length > N

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_synthetic_dat(td, "train", X, y_flat)
        spec = _toy_spec(D, [0, 1, 2, 3])

        try:
            read_vectorized_features(td, "train", spec)
        except ValueError as e:
            assert "train_family" in str(e)
            return
    raise AssertionError("expected ValueError for multilabel y shape")


# ---- missing files ---------------------------------------------------------

def test_missing_subset_files_raise():
    with tempfile.TemporaryDirectory() as td:
        spec = _toy_spec(4, [0, 1, 2, 3])
        try:
            read_vectorized_features(Path(td), "train", spec)
        except ValueError as e:
            assert "Invalid subset file" in str(e)
            return
    raise AssertionError("expected ValueError for missing .dat files")


# ---- apply_spec ------------------------------------------------------------

def test_apply_spec_2d():
    X = np.arange(24, dtype=np.float32).reshape(4, 6)
    spec = _toy_spec(6, [0, 2, 5])
    out = apply_spec(X, spec)
    assert out.shape == (4, 3)
    assert out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, X[:, [0, 2, 5]])


def test_apply_spec_1d():
    """Predict path: a single full-width vector gets sliced and stays 1-D."""
    vec = np.arange(6, dtype=np.float32)
    spec = _toy_spec(6, [1, 3, 4])
    out = apply_spec(vec, spec)
    assert out.shape == (3,)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.array([1, 3, 4], dtype=np.float32))


def test_apply_spec_rejects_wrong_width_2d():
    X = np.zeros((4, 8), dtype=np.float32)
    spec = _toy_spec(6, [0, 1, 2])
    try:
        apply_spec(X, spec)
    except ValueError as e:
        assert "expects 6" in str(e) or "expected" in str(e).lower() or "expects" in str(e).lower()
        return
    raise AssertionError("expected ValueError for width mismatch")


def test_apply_spec_rejects_wrong_width_1d():
    """1-D vectors must be validated too — predict_file passes a 1-D extractor
    output through apply_spec; a wrong-length input would otherwise silently
    produce wrong values or raise an opaque IndexError."""
    vec = np.zeros(8, dtype=np.float32)        # 8 elements
    spec = _toy_spec(6, [0, 1, 2])             # spec expects 6
    try:
        apply_spec(vec, spec)
    except ValueError as e:
        assert "spec expects 6" in str(e)
        return
    raise AssertionError("expected ValueError for 1-D width mismatch")


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
