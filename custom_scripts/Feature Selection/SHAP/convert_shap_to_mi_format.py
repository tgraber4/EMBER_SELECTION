"""
Convert a SHAP dropped-features CSV (output of shap_cluster_feature_selection.py)
into the same list format produced by mi_feature_selection.py.

Output fields per entry: rank, index, block, field, hashed, mean_abs_shap, drop_reason.
Entries preserve the original rank order (rank 1 = dropped first).
"""

import csv
import json
from pathlib import Path

# --- User-editable paths ---
INPUT_PATH  = "output\SHAP\dropped_features.csv"
OUTPUT_PATH = "output/SHAP/shap_dropped_features.json"


def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    records = []
    with in_path.open(newline="") as f:
        for row in csv.DictReader(f):
            records.append({
                "rank":          int(row["rank"]),
                "index":         int(row["index"]),
                "block":         row["block"],
                "field":         row["field"],
                "hashed":        row["hashed"].strip().lower() == "true",
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "drop_reason":   row["drop_reason"],
            })

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Converted {len(records)} dropped features -> {out}")


if __name__ == "__main__":
    main()
