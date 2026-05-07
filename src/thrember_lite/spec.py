"""FeatureSpec — single source of truth for which columns survive feature selection.

Layer A (selection input) is `dropped_features.json` — emitted by selection scripts,
contains rich per-feature metadata. Only `index` is load-bearing.

Layer B (runtime spec) is `spec.json` — minimal, ships with the model, holds
`kept_indices` plus the metadata `predict_file` and ablation tooling need.

`FeatureSpec.from_drop_columns(...)` reads Layer A and computes Layer B.
`FeatureSpec.from_json(spec.json)` reads Layer B directly. Layer B round-trips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

# thrember's hardcoded categorical indices (model.py:324, 365, 366, 403, 404).
# Default for EMBER2024 (original_dim = 2568).
DEFAULT_CATEGORICAL: list[int] = [2, 3, 4, 5, 6, 701, 702]


@dataclass
class FeatureSpec:
    """Describes a column slice over the full-width EMBER vector.

    Fields are the Layer-B JSON contract spelled out in the plan.
    """

    original_dim: int
    kept_indices: list[int]                          # sorted; positions into full vector
    original_categorical: list[int]                  # categorical positions in full vector
    block_ranges: dict[str, tuple[int, int]]         # block name -> [start, end_exclusive)
    source: dict[str, Any] = field(default_factory=dict)

    # ----- derived -----------------------------------------------------------

    @cached_property
    def new_categorical(self) -> list[int]:
        """Categorical indices remapped to post-slice positions.

        Cached on first access. FeatureSpec is treated as immutable after
        construction; if you mutate `kept_indices` or `original_categorical`
        in place, invalidate by deleting `self.__dict__["new_categorical"]`.

        Example: kept = [0,1,2,5,6,...,2567], original_categorical = [2,3,4,5,6,701,702]
                 → 3 and 4 dropped; 5→3, 6→4, 701→699, 702→700; result [2,3,4,699,700].
        """
        idx_to_pos = {idx: pos for pos, idx in enumerate(self.kept_indices)}
        return [idx_to_pos[c] for c in self.original_categorical if c in idx_to_pos]

    # ----- serialization -----------------------------------------------------

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        blob = {
            "original_dim": self.original_dim,
            "kept_indices": self.kept_indices,
            "original_categorical": self.original_categorical,
            "new_categorical": self.new_categorical,        # cached for debuggability
            "block_ranges": {k: list(v) for k, v in self.block_ranges.items()},
            "source": self.source,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "FeatureSpec":
        with Path(path).open(encoding="utf-8") as f:
            blob = json.load(f)
        kept_indices = [int(i) for i in blob["kept_indices"]]

        # Sort invariant — silently-unsorted kept_indices would make column
        # slicing produce a different column order than what the model was
        # trained on. Fail loudly rather than corrupting predictions.
        if kept_indices != sorted(kept_indices):
            raise ValueError(
                f"{path}: kept_indices must be sorted ascending; got out-of-order "
                f"entries (first divergence at position "
                f"{next(i for i in range(len(kept_indices) - 1) if kept_indices[i] >= kept_indices[i + 1])})"
            )

        return cls(
            original_dim=int(blob["original_dim"]),
            kept_indices=kept_indices,
            original_categorical=[int(c) for c in blob["original_categorical"]],
            block_ranges={k: tuple(v) for k, v in blob["block_ranges"].items()},
            source=blob.get("source", {}),
        )

    # ----- constructors ------------------------------------------------------

    @classmethod
    def from_drop_columns(
        cls,
        drop_idx_or_path: Iterable[int] | Path | str,
        *,
        original_categorical: list[int] | None = None,
        index_map_path: Path | str | None = None,
        source_note: str = "",
    ) -> "FeatureSpec":
        """Build a spec from a list of column indices to drop, or a Layer-A JSON file.

        Layer-A JSON is the existing `dropped_features.json` shape — a list of
        objects with at least an `index` field. A bare list of ints is also accepted.

        `index_map_path` (optional) loads `original_dim` and `block_ranges` from a
        `feature_index_map.json` file rather than instantiating PEFeatureExtractor.
        Useful when running where `thrember` import is slow or unavailable.
        """
        drop_idx = _resolve_drop_indices(drop_idx_or_path)

        # Resolve original_dim and block_ranges
        if index_map_path is not None:
            original_dim, block_ranges = _read_index_map(Path(index_map_path))
        else:
            original_dim, block_ranges = _derive_layout_from_extractor()

        # Validate drop indices are in range
        bad = [i for i in drop_idx if i < 0 or i >= original_dim]
        if bad:
            raise ValueError(
                f"drop indices out of range [0, {original_dim}): "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''}"
            )

        # Compute kept (sorted)
        drop_set = set(drop_idx)
        kept_indices = [i for i in range(original_dim) if i not in drop_set]

        # Default categorical list — preserve user override if given
        if original_categorical is None:
            original_categorical = list(DEFAULT_CATEGORICAL)

        # Provenance for debugging six months from now
        source: dict[str, Any] = {"drop_count": len(drop_idx)}
        if isinstance(drop_idx_or_path, (str, Path)):
            source["drop_file"] = str(drop_idx_or_path)
        if source_note:
            source["note"] = source_note

        return cls(
            original_dim=original_dim,
            kept_indices=kept_indices,
            original_categorical=original_categorical,
            block_ranges=block_ranges,
            source=source,
        )


# ----- helpers ---------------------------------------------------------------

def _resolve_drop_indices(drop_idx_or_path: Iterable[int] | Path | str) -> list[int]:
    """Accept a path to Layer-A JSON, or a bare iterable of ints."""
    if isinstance(drop_idx_or_path, (str, Path)):
        path = Path(drop_idx_or_path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Layer-A: list of {index, ...}
            try:
                return sorted({int(row["index"]) for row in data})
            except KeyError:
                raise ValueError(
                    f"{path}: Layer-A JSON entries must have an 'index' field; "
                    f"first entry has keys {sorted(data[0].keys())}"
                )
        if isinstance(data, list):
            # Bare list of ints
            return sorted({int(x) for x in data})
        raise ValueError(f"unrecognized drop file format: {path}")
    return sorted({int(i) for i in drop_idx_or_path})


def _read_index_map(path: Path) -> tuple[int, dict[str, tuple[int, int]]]:
    """Read original_dim and block_ranges from a feature_index_map.json file.

    feature_index_map.json stores ranges as [start, end_inclusive].
    We convert to [start, end_exclusive) internally.
    """
    with path.open(encoding="utf-8") as f:
        blob = json.load(f)
    original_dim = int(blob["dim"])
    block_ranges = {
        name: (int(rng[0]), int(rng[1]) + 1)
        for name, rng in blob["block_ranges"].items()
    }
    return original_dim, block_ranges


def _derive_layout_from_extractor() -> tuple[int, dict[str, tuple[int, int]]]:
    """Derive original_dim and block_ranges from a fresh PEFeatureExtractor.

    Each feature object has `.name` (e.g. "general") and `.dim`. They are
    concatenated in the order given by extractor.features.
    """
    from thrember import PEFeatureExtractor  # local import: tests can stub this
    extractor = PEFeatureExtractor()
    block_ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for fe in extractor.features:
        block_ranges[fe.name] = (offset, offset + fe.dim)
        offset += fe.dim
    return extractor.dim, block_ranges
