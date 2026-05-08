"""
Convert an embedded dropped-features JSON (output of train_custom_lgbm_fs.py)
into the same list format produced by mi_feature_selection.py.

Output fields per entry: rank, index, block, field, hashed, gain.
Entries are ordered rank 1 = lowest gain (worst feature) first.
"""

import json
from pathlib import Path

# --- User-editable paths ---
INPUT_PATH       = Path("output/Embedded/embedded_dropped_features.json")
OUTPUT_PATH      = Path("output/Embedded/updated_dropped_features.json")
FEATURE_MAP_PATH = Path("Documentation/feature_index_map.json")


def main():
    with open(INPUT_PATH) as f:
        embedded = json.load(f)
    with open(FEATURE_MAP_PATH) as f:
        fmap = json.load(f)

    index_map = {e["index"]: e for e in fmap["entries"]}
    gain_map  = {row["index"]: row["gain"] for row in embedded["feature_table"]}

    dropped_indices = embedded["dropped_indices"]
    dropped_with_gain = sorted(
        ((idx, gain_map.get(idx, 0.0)) for idx in dropped_indices),
        key=lambda x: (x[1], x[0]),
    )

    records = []
    for rank, (idx, gain) in enumerate(dropped_with_gain, start=1):
        entry = index_map.get(idx, {})
        records.append({
            "rank":   rank,
            "index":  idx,
            "block":  entry.get("block", "unknown"),
            "field":  entry.get("field", f"feature[{idx}]"),
            "hashed": entry.get("hashed", False),
            "gain":   gain,
        })

    out = OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Converted {len(records)} dropped features -> {out}")


if __name__ == "__main__":
    main()
