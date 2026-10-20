"""FeatureSpec unit tests.

Designed to run under pytest if installed (`pytest src/thrember_lite/tests`),
or directly via `python -m thrember_lite.tests.test_spec` for stdlib-only setups.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from thrember_lite.spec import FeatureSpec, _resolve_drop_indices


# ---- helpers ---------------------------------------------------------------

def _toy_block_ranges() -> dict[str, tuple[int, int]]:
    return {"alpha": (0, 5), "beta": (5, 10)}


# ---- new_categorical remap ------------------------------------------------

def test_new_categorical_drops_dropped_categoricals():
    """Drop original index 3; categorical at 3 disappears, others shift left."""
    spec = FeatureSpec(
        original_dim=10,
        kept_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9],
        original_categorical=[2, 3, 4, 5, 6],
        block_ranges=_toy_block_ranges(),
    )
    # Survivors: 2 (kept), 3 dropped, 4 (kept→3), 5 (kept→4), 6 (kept→5)
    assert spec.new_categorical == [2, 3, 4, 5]


def test_new_categorical_real_ember_shape():
    """Plan example: drop indices 3,4 from 2568-dim, originals [2,3,4,5,6,701,702]."""
    drop = {3, 4}
    kept = [i for i in range(2568) if i not in drop]
    spec = FeatureSpec(
        original_dim=2568,
        kept_indices=kept,
        original_categorical=[2, 3, 4, 5, 6, 701, 702],
        block_ranges={},
    )
    # 2 (kept), 3 dropped, 4 dropped, 5→3, 6→4, 701→699, 702→700
    assert spec.new_categorical == [2, 3, 4, 699, 700]


def test_new_categorical_no_drops_is_identity():
    spec = FeatureSpec(
        original_dim=10,
        kept_indices=list(range(10)),
        original_categorical=[2, 3, 4, 5, 6],
        block_ranges=_toy_block_ranges(),
    )
    assert spec.new_categorical == [2, 3, 4, 5, 6]


# ---- JSON round-trip -------------------------------------------------------

def test_json_round_trip():
    spec = FeatureSpec(
        original_dim=10,
        kept_indices=[0, 1, 2, 5, 6, 7, 8, 9],
        original_categorical=[2, 3, 4],
        block_ranges=_toy_block_ranges(),
        source={"drop_count": 2, "note": "unit test"},
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "spec.json"
        spec.to_json(p)
        loaded = FeatureSpec.from_json(p)

    assert loaded.original_dim == spec.original_dim
    assert loaded.kept_indices == spec.kept_indices
    assert loaded.original_categorical == spec.original_categorical
    assert loaded.block_ranges == spec.block_ranges
    assert loaded.new_categorical == spec.new_categorical
    assert loaded.source == spec.source


def test_json_includes_cached_new_categorical():
    """Layer B JSON must serialize new_categorical even though it's derived,
    so a debugging human can read it without recomputing."""
    spec = FeatureSpec(
        original_dim=10,
        kept_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9],
        original_categorical=[2, 3, 4, 5, 6],
        block_ranges=_toy_block_ranges(),
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "spec.json"
        spec.to_json(p)
        blob = json.loads(p.read_text())
    assert blob["new_categorical"] == [2, 3, 4, 5]


# ---- from_drop_columns -----------------------------------------------------

def test_from_drop_columns_with_index_map_file():
    """Use index_map_path to avoid PEFeatureExtractor in unit tests."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Note: feature_index_map.json uses [start, end_inclusive].
        index_map = {
            "dim": 10,
            "block_ranges": {"alpha": [0, 4], "beta": [5, 9]},
        }
        index_path = td / "feature_index_map.json"
        index_path.write_text(json.dumps(index_map))

        spec = FeatureSpec.from_drop_columns(
            [3, 7],
            original_categorical=[2, 5],   # provide one in scope; default 701/702 would OOR
            index_map_path=index_path,
        )

    assert spec.original_dim == 10
    assert spec.kept_indices == [0, 1, 2, 4, 5, 6, 8, 9]
    # block_ranges normalized to [start, end_exclusive)
    assert spec.block_ranges == {"alpha": (0, 5), "beta": (5, 10)}
    # original_categorical preserved verbatim; new_categorical computed
    assert spec.original_categorical == [2, 5]
    assert spec.new_categorical == [2, 4]    # 2→pos2; 5→pos4


def test_from_drop_columns_layer_a_json_format():
    """Selection scripts emit list-of-objects with at least an `index` field."""
    layer_a = [
        {"rank": 1, "index": 5, "block": "x", "field": "f1", "hashed": False, "mi_score": 0.0},
        {"rank": 2, "index": 7, "block": "x", "field": "f2", "hashed": False, "mi_score": 0.0},
        {"rank": 3, "index": 5, "block": "x", "field": "dup", "hashed": False, "mi_score": 0.0},  # dup tolerated
    ]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        drop_path = td / "dropped_features.json"
        drop_path.write_text(json.dumps(layer_a))
        index_path = td / "feature_index_map.json"
        index_path.write_text(json.dumps({"dim": 10, "block_ranges": {"x": [0, 9]}}))

        spec = FeatureSpec.from_drop_columns(
            drop_path,
            original_categorical=[2],
            index_map_path=index_path,
            source_note="unit test",
        )

    assert spec.kept_indices == [0, 1, 2, 3, 4, 6, 8, 9]   # 5 and 7 dropped; dup deduped
    assert spec.source["drop_count"] == 2
    assert spec.source["drop_file"].endswith("dropped_features.json")
    assert spec.source["note"] == "unit test"


def test_from_drop_columns_rejects_out_of_range():
    with tempfile.TemporaryDirectory() as td:
        index_path = Path(td) / "fim.json"
        index_path.write_text(json.dumps({"dim": 10, "block_ranges": {}}))
        try:
            FeatureSpec.from_drop_columns(
                [0, 99],   # 99 out of range
                original_categorical=[2],
                index_map_path=index_path,
            )
        except ValueError as e:
            assert "out of range" in str(e)
            return
    raise AssertionError("expected ValueError for out-of-range drop index")


# ---- defensive validation in from_json -------------------------------------

def test_from_json_rejects_unsorted_kept_indices():
    """Hand-edited spec.json with unsorted kept_indices must fail loudly —
    silent column reordering would produce wrong predictions."""
    bad_blob = {
        "original_dim": 10,
        "kept_indices": [0, 5, 2, 7, 9],   # 5 then 2 — out of order
        "original_categorical": [],
        "new_categorical": [],
        "block_ranges": {},
        "source": {},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad_spec.json"
        p.write_text(json.dumps(bad_blob))
        try:
            FeatureSpec.from_json(p)
        except ValueError as e:
            assert "sorted" in str(e).lower()
            return
    raise AssertionError("expected ValueError for unsorted kept_indices")


def test_from_json_accepts_sorted_kept_indices():
    """Sanity: well-formed sorted kept_indices loads fine."""
    blob = {
        "original_dim": 10,
        "kept_indices": [0, 2, 5, 7, 9],
        "original_categorical": [2],
        "new_categorical": [1],
        "block_ranges": {"x": [0, 10]},
        "source": {},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ok_spec.json"
        p.write_text(json.dumps(blob))
        spec = FeatureSpec.from_json(p)
    assert spec.kept_indices == [0, 2, 5, 7, 9]


def test_resolve_drop_indices_missing_index_field_message():
    """Layer-A entries missing the 'index' key produce a clear error."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "drop.json"
        p.write_text(json.dumps([{"rank": 1, "block": "x"}]))    # no 'index'
        try:
            _resolve_drop_indices(p)
        except ValueError as e:
            assert "'index'" in str(e)
            return
    raise AssertionError("expected ValueError mentioning the 'index' field")


# ---- _resolve_drop_indices direct tests ------------------------------------

def test_resolve_drop_indices_bare_list():
    assert _resolve_drop_indices([3, 1, 2, 1]) == [1, 2, 3]   # sorted, deduped


def test_resolve_drop_indices_layer_a_path():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "drop.json"
        p.write_text(json.dumps([{"index": 5}, {"index": 1}, {"index": 5}]))
        assert _resolve_drop_indices(p) == [1, 5]


def test_resolve_drop_indices_bare_int_list_in_json():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "drop.json"
        p.write_text(json.dumps([3, 1, 2]))
        assert _resolve_drop_indices(p) == [1, 2, 3]


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
