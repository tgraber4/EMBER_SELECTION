"""
Remove all samples belonging to rare malware families from a .jsonl file.

Two streaming passes over the input:
  1. count samples per `family`
  2. write only the lines whose family has at least `--threshold` samples

Benign samples (label 0) have no family and are always kept. Malware (label 1)
with an empty/null family is dropped, since it belongs to no family.

Lines are copied verbatim, so all features are preserved.

Usage (from the EMBER2024 root):
    python custom_scripts_2\\filter_families.py ember_data\\PE_train_sampled.jsonl
    python custom_scripts_2\\filter_families.py in.jsonl -t 100 -o out.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_THRESHOLD = 75


def parse(line: str) -> tuple[str | None, bool]:
    """Return (family or None, is_benign)."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None, False
    return obj.get("family") or None, obj.get("label") == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="input .jsonl file")
    parser.add_argument("-t", "--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"minimum samples per family to keep (default {DEFAULT_THRESHOLD})")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output path (default: <input>_min<threshold>.jsonl)")
    args = parser.parse_args()

    output = args.output or args.input.with_name(
        f"{args.input.stem}_min{args.threshold}{args.input.suffix}"
    )
    if output.resolve() == args.input.resolve():
        parser.error("output must differ from input")

    # Pass 1: family counts
    counts: Counter = Counter()
    total = 0
    with args.input.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            fam, _ = parse(line)
            if fam:
                counts[fam] += 1

    kept_families = {f for f, c in counts.items() if c >= args.threshold}
    print(f"[count] {total:,} samples, {len(counts):,} families")
    print(f"[count] {len(kept_families):,} families have >= {args.threshold} samples")

    # Pass 2: write
    kept = benign_kept = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, \
            output.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            if not line.strip():
                continue
            fam, benign = parse(line)
            if fam is None:
                if not benign:
                    continue  # malware with no family
                benign_kept += 1
            elif fam not in kept_families:
                continue
            if not line.endswith("\n"):
                line += "\n"
            fout.write(line)
            kept += 1

    print(f"[write] kept {kept:,} of {total:,} samples "
          f"({kept - benign_kept:,} family, {benign_kept:,} benign)")
    print(f"[write] {output}")


if __name__ == "__main__":
    main()
