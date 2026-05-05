"""Check whether every sample in PE_both_sampled.jsonl has a malware/benign label."""

import json
from collections import Counter
from pathlib import Path

JSONL_PATH = Path(__file__).resolve().parent.parent / "ember_data" / "PE_train_sampled.jsonl"


def main() -> None:
    total = 0
    missing_label = []
    label_counts: Counter = Counter()

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {lineno}: JSON decode error: {e}")
                continue

            if "label" not in sample:
                missing_label.append(lineno)
                continue

            label_counts[sample["label"]] += 1

    print(f"File: {JSONL_PATH}")
    print(f"Total samples: {total}")
    print(f"Samples missing 'label' key: {len(missing_label)}")
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda kv: (kv[0] is None, kv[0])):
        name = {0: "benign", 1: "malware", -1: "unlabeled"}.get(label, str(label))
        print(f"  {label!r} ({name}): {count}")

    labeled = label_counts.get(0, 0) + label_counts.get(1, 0)
    unlabeled = total - labeled
    print()
    if unlabeled == 0 and not missing_label:
        print("All samples have a malware/benign label.")
    else:
        print(f"{unlabeled} sample(s) are NOT labeled as malware/benign.")
        if missing_label:
            preview = missing_label[:10]
            print(f"  First missing-label line numbers: {preview}"
                  f"{' ...' if len(missing_label) > 10 else ''}")


if __name__ == "__main__":
    main()
