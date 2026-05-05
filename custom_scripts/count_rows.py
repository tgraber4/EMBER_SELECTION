"""
Count rows (samples) in .jsonl files. Two modes:

  - "all":    count rows in every .jsonl file under ember_data/ and report a total.
  - "single": count rows in a single .jsonl file.

Configure via the MODE / TARGET constants below, or pass a path on the CLI:
    python count_rows.py                       # uses MODE
    python count_rows.py --all                 # all .jsonl in DATA_DIR
    python count_rows.py path/to/file.jsonl    # single file
Streams line-by-line so memory stays bounded.
"""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "ember_data"

# "all" -> count every .jsonl in DATA_DIR; "single" -> count just TARGET.
MODE = "single"
TARGET = DATA_DIR / "PE_train_sampled.jsonl"  # used when MODE == "single"


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_all(data_dir: Path) -> None:
    all_files = sorted(p for p in data_dir.glob("*.jsonl") if p.is_file())
    if not all_files:
        print(f"No .jsonl files found in {data_dir}")
        return

    total = 0
    for p in all_files:
        n = count_lines(p)
        total += n
        print(f"{p.name}: {n:,}")

    print("-" * 60)
    print(f"Total rows across all .jsonl files in {data_dir}: {total:,}")


def count_single(path: Path) -> None:
    if not path.is_file():
        print(f"File not found: {path}")
        return
    n = count_lines(path)
    print(f"{path}: {n:,}")


def main() -> None:
    args = sys.argv[1:]
    if args:
        if args[0] in ("--all", "-a"):
            count_all(DATA_DIR)
        else:
            count_single(Path(args[0]))
        return

    if MODE == "all":
        count_all(DATA_DIR)
    elif MODE == "single":
        count_single(TARGET)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (expected 'all' or 'single')")


if __name__ == "__main__":
    main()
