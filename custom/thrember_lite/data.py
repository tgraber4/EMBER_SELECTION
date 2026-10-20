"""Memory-aware loader and column slicer for EMBER .dat files.

`read_vectorized_features` mirrors `thrember.model.read_vectorized_features`'s name
and semantics, with two additions: takes a `FeatureSpec` so it slices columns during
the load, and uses chunked materialization so it never makes the full-width RAM copy
that `thrember.model.read_vectorized_features` does at `model.py:265`.

Per Decision #7 the loader does NOT filter `y == -1`. The trainer does that, mirroring
`thrember.train_model` at `model.py:360-361`. Test/challenge eval calls this directly
to keep all rows.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .spec import FeatureSpec


def read_vectorized_features(
    data_dir: Path | str,
    subset: str,
    spec: FeatureSpec,
    *,
    block: int = 50_000,
    in_memory: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Read EMBER vectors from disk and slice columns to `spec.kept_indices`.

    Returns (X, y). X is C-contiguous float32 of shape `(N, len(spec.kept_indices))`.
    y is int32 of shape `(N,)`. No row filtering.

    Parameters
    ----------
    data_dir : path containing `X_<subset>.dat` / `y_<subset>.dat` (thrember layout).
    subset   : "train", "test", "challenge", etc.
    spec     : FeatureSpec; provides `original_dim` and `kept_indices`.
    block    : rows per materialization step (chunked path). Lower it on tight RAM.
    in_memory: if True, skip chunking and materialize the slice in one shot.
               Simpler and slightly faster for small subsets; same return shape.
    """
    data_path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    original_dim = spec.original_dim
    kept_indices = np.asarray(spec.kept_indices, dtype=np.int64)

    # X memmap stays lazy; we never call np.array(X_mm) at full width
    X_mm = np.memmap(X_path, dtype=np.float32, mode="r").reshape(-1, original_dim)
    n_rows = X_mm.shape[0]

    # y is small (4 bytes per row); read fully into RAM
    y_mm = np.memmap(y_path, dtype=np.int32, mode="r")
    y = np.array(y_mm)

    # 1D-y assertion — multilabel y is stored flat with length N*K, so flat length > N
    # detects it. Mirrors thrember.model.read_vectorized_features (model.py:269).
    if y.shape[0] > n_rows:
        raise ValueError(
            "Encountered y with invalid shape. Use train_family() instead."
        )
    if y.shape[0] != n_rows:
        raise ValueError(
            f"y length {y.shape[0]} does not match X rows {n_rows} "
            f"(corrupt .dat or wrong original_dim?)"
        )

    if in_memory:
        # Simple path: one big fancy-index. Faster for small subsets.
        out = np.ascontiguousarray(np.asarray(X_mm)[:, kept_indices])
    else:
        # Chunked path (default). Block window pages out between iterations.
        out = np.empty((n_rows, kept_indices.size), dtype=np.float32)
        for r0 in range(0, n_rows, block):
            r1 = min(r0 + block, n_rows)
            out[r0:r1] = X_mm[r0:r1][:, kept_indices]

    return out, y


def apply_spec(X: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Slice an in-memory array to `spec.kept_indices`. Returns C-contiguous float32.

    Used on the predict path (after `extractor.feature_vector`) and for ad-hoc
    in-memory slicing. For loading from disk, prefer `read_vectorized_features`,
    which is chunked.

    Validates the last-axis length against `spec.original_dim` for both 1-D and
    multi-D inputs. Without this, a wrong-length 1-D vector would either silently
    return wrong values (if all kept_indices happen to fit) or raise an unclear
    IndexError.
    """
    if X.shape[-1] != spec.original_dim:
        raise ValueError(
            f"apply_spec: X has {X.shape[-1]} elements on the last axis; "
            f"spec expects {spec.original_dim}"
        )
    if X.ndim == 1:
        # Single sample — kept as 1D so callers can wrap in [vec] for booster.predict
        return np.ascontiguousarray(X[spec.kept_indices], dtype=np.float32)
    return np.ascontiguousarray(X[..., spec.kept_indices], dtype=np.float32)
