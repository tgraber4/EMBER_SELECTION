"""Download the EMBER2024 challenge dataset from HuggingFace.

Edit DEST_DIR, then run:

    python custom_scripts/download_challenge.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from thrember import download_dataset


def main() -> int:
    DEST_DIR = Path("ember_data/")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    abs_dest = DEST_DIR.resolve()
    before = {p.name for p in abs_dest.iterdir() if p.is_file()}

    # download_dataset chdirs into the target; restore CWD on exit.
    cwd = os.getcwd()
    try:
        print(f"Downloading challenge dataset into {abs_dest} ...")
        download_dataset(str(abs_dest), split="challenge", file_type="all")
    finally:
        os.chdir(cwd)

    new_files = sorted({p.name for p in abs_dest.iterdir() if p.is_file()} - before)
    if new_files:
        print("\nDownloaded:")
        for name in new_files:
            size_mb = (abs_dest / name).stat().st_size / 1_000_000
            print(f"  {name}  ({size_mb:,.1f} MB)")
    else:
        print("\nNo new files (already present).")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
