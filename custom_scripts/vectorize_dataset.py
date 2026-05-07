"""Vectorize a folder of EMBER JSONL files into .dat memmaps.

Wraps thrember.create_vectorized_features. The folder must contain at least
one JSONL with `train`, `test`, and `challenge` in the filename (substring
match). Produces six files: X_<subset>.dat / y_<subset>.dat for each subset.

Edit DATA_DIR, then run:

    python custom_scripts/vectorize_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from thrember import create_vectorized_features


_DAT_FILES = (
    "X_train.dat", "y_train.dat",
    "X_test.dat",  "y_test.dat",
    "X_challenge.dat", "y_challenge.dat",
)


def main() -> int:
    DATA_DIR = Path("ember_data/")
    OVERWRITE = False             # set True to re-vectorize when .dat files exist

    if not DATA_DIR.is_dir():
        print(f"error: DATA_DIR {DATA_DIR} does not exist", file=sys.stderr)
        return 2

    existing = [f for f in _DAT_FILES if (DATA_DIR / f).is_file()]
    if existing and not OVERWRITE:
        print(f"All .dat files already present in {DATA_DIR}; nothing to do.")
        print("Set OVERWRITE = True to force re-vectorization.")
        return 0

    print(f"Vectorizing JSONL -> .dat in {DATA_DIR} ...")
    create_vectorized_features(DATA_DIR)

    print("\nWrote:")
    for f in _DAT_FILES:
        path = DATA_DIR / f
        if path.is_file():
            size_mb = path.stat().st_size / 1_000_000
            print(f"  {f}  ({size_mb:,.1f} MB)")
        else:
            print(f"  {f}  (missing!)")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
