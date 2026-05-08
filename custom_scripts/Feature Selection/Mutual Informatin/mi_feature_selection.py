import json
import csv
import time
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from thrember.features import PEFeatureExtractor


def main():
    JSONL_FILE = r"ember_data\PE_train_sampled.jsonl"
    N_DROP = 257
    MAX_SAMPLES = 100_000
    OUT_DIR = Path("output/Mutual_Information")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT = str(OUT_DIR / "dropped_features.csv")
    FEATURE_MAP = "Documentation/feature_index_map.json"
    SEED = 42

    t0 = time.perf_counter()

    # Step 1: Load feature index map
    print("Loading feature index map...")
    with open(FEATURE_MAP, "r") as f:
        fmap_data = json.load(f)

    dim = fmap_data["dim"]
    index_map = {entry["index"]: entry for entry in fmap_data["entries"]}

    # Step 2: Stream JSONL, skip unlabeled rows, stop at MAX_SAMPLES
    print(f"Streaming {JSONL_FILE} (max {MAX_SAMPLES:,} labeled rows)...")
    extractor = PEFeatureExtractor()
    X_list = []
    y_list = []

    with open(JSONL_FILE, "r") as f:
        for line in f:
            if len(X_list) >= MAX_SAMPLES:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            label = raw.get("label", -1)
            if label == -1 or label is None:
                continue

            try:
                vec = extractor.process_raw_features(raw)
            except Exception:
                continue

            X_list.append(vec)
            y_list.append(int(label))

    print(f"  Collected {len(X_list):,} labeled samples.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    del X_list, y_list

    # Step 3: Build categorical mask — indices confirmed from model.py categorical_feature param
    cat_indices = {2, 3, 4, 5, 6, 701, 702}
    discrete_features = np.array([i in cat_indices for i in range(dim)], dtype=bool)

    # Step 4: Compute MI
    print(f"Computing mutual information ({X.shape[0]:,} samples x {X.shape[1]:,} features)...")
    print("  This may take 6-15 minutes...")
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=SEED)

    # Step 5: Rank and select bottom N_DROP (ascending = lowest MI first)
    sorted_indices = np.argsort(mi_scores)
    drop_indices = sorted_indices[:N_DROP]

    records = []
    for rank, feat_idx in enumerate(drop_indices, start=1):
        entry = index_map.get(int(feat_idx), {})
        records.append({
            "rank": rank,
            "index": int(feat_idx),
            "block": entry.get("block", "unknown"),
            "field": entry.get("field", f"feature[{feat_idx}]"),
            "hashed": entry.get("hashed", False),
            "mi_score": float(mi_scores[feat_idx]),
        })

    # Step 6: Console table
    print()
    print(f"{'Rank':>6}  {'Index':>6}  {'Block':>20}  {'MI Score':>12}  {'Hashed':>6}  Field")
    print("-" * 90)
    for r in records:
        print(f"{r['rank']:>6}  {r['index']:>6}  {r['block']:>20}  {r['mi_score']:.8f}  {str(r['hashed']):>6}  {r['field']}")

    print()
    print("Drop candidates by block:")
    block_counts = Counter(r["block"] for r in records)
    for block, count in sorted(block_counts.items(), key=lambda x: -x[1]):
        print(f"  {block:>20}: {count}")

    # Step 7: Save CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "index", "block", "field", "hashed", "mi_score"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved: {OUTPUT}")

    json_path = OUTPUT.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved: {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
