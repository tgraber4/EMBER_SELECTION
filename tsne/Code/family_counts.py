"""
Count samples per malware family in the sampled train .jsonl file.

Streams the file line-by-line and tallies the `family` field. Rows with an
empty/null family (e.g. benign samples) are counted under "(none)".

Prints a table sorted by count and writes
the same table to ember_data/family_counts.csv.

Usage (from the EMBER2024 root):
    python tsne/Code/family_counts.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("./ember_data")
INPUT_PATH = DATA_DIR / "PE_train_sampled_min75.jsonl"
OUTPUT_CSV = DATA_DIR / "family_counts_min75.csv"
NO_FAMILY = "(none)"


def count_families(path: Path) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            counts[obj.get("family") or NO_FAMILY] += 1
    return counts, total


def main() -> None:
    print(f"[read] {INPUT_PATH}")
    counts, total = count_families(INPUT_PATH)
    print(f"       {total:,} samples, {len(counts):,} distinct families")

    rows = sorted(counts.items(), key=lambda r: (-r[1], r[0]))

    width = max(len(f) for f, _ in rows)
    print()
    print(f"{'family':<{width}}  {'train':>9}")
    for f, n in rows:
        print(f"{f:<{width}}  {n:>9,}")
    print(f"{'TOTAL':<{width}}  {total:>9,}")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["family", "train"])
        writer.writerows(rows)
    print(f"\n[write] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
