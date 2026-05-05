"""
Download the EMBER2024 PE dataset (Win32, Win64, Dot_Net) for both the train
(first 52 weeks) and test (last 12 weeks) splits and apply Bernoulli sampling
at independent rates per split, writing two separate output files so the
official temporal train/test boundary is preserved downstream.

The upstream joyce8/EMBER2024 shards contain each PE sample twice: once with
the original (empty) `caps` field and once with populated Capa analysis. We
dedupe by md5 within each split and keep the version whose `caps` field is
populated. The Bernoulli decision is computed deterministically from the md5
(plus split), so both copies of a sample always agree on whether to keep it.

Streams each input .jsonl line-by-line. Only selected rows are buffered in
memory. All 2,568 feature dimensions and the nested label dictionary are
preserved verbatim from the chosen source line.

Usage:
    python bernoulli_sample.py
"""

import hashlib
import json
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

PE_FILE_TYPES = ("Win32", "Win64", "Dot_Net")
SPLITS = ("train", "test")
SPLIT_PROBABILITIES = {
    "train": 0.0427,
    "test": 0.0427,
}
SEED = 42
HF_REPO_ID = "joyce8/EMBER2024"
DATA_DIR = Path("./ember_data")
OUTPUT_PATHS = {
    "train": DATA_DIR / "PE_train_sampled.jsonl",
    "test": DATA_DIR / "PE_test_sampled.jsonl",
}

_SAMPLE_SCALE = 2**64


def download_pe_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        for file_type in PE_FILE_TYPES:
            existing = sorted(data_dir.glob(f"*_{file_type}_{sp}.jsonl"))
            direct = data_dir / f"{file_type}_{sp}.jsonl"
            if direct.is_file() and direct not in existing:
                existing.append(direct)
            if existing:
                print(f"[skip] {file_type}_{sp}: found {len(existing)} existing .jsonl shard(s)")
                continue

            zip_name = f"{file_type}_{sp}.zip"
            print(f"[download] {zip_name}")
            zip_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=zip_name,
                repo_type="dataset",
            )
            print(f"[extract]  {zip_name} -> {data_dir}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            try:
                os.remove(zip_path)
            except OSError as e:
                print(f"[warn] could not remove cached zip {zip_path}: {e}")


def iter_pe_files_by_split(data_dir: Path) -> dict[str, list[Path]]:
    by_split: dict[str, list[Path]] = {sp: [] for sp in SPLITS}
    for sp in SPLITS:
        for file_type in PE_FILE_TYPES:
            matches = sorted(data_dir.glob(f"*_{file_type}_{sp}.jsonl"))
            direct = data_dir / f"{file_type}_{sp}.jsonl"
            if direct.is_file() and direct not in matches:
                matches.append(direct)
            if not matches:
                print(f"[warn] no files matched *_{file_type}_{sp}.jsonl in {data_dir}")
                continue
            by_split[sp].extend(matches)
    if not any(by_split.values()):
        raise FileNotFoundError(f"No PE .jsonl files found in {data_dir}")
    return by_split


def _sample_decision(md5: str, split: str, p: float, seed: int) -> bool:
    # Deterministic per-(split, md5) Bernoulli: both duplicate rows agree.
    digest = hashlib.blake2b(f"{seed}:{split}:{md5}".encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(digest, "big") / _SAMPLE_SCALE
    return u < p


def bernoulli_sample(
    input_paths: list[Path],
    output_path: Path,
    split: str,
    p: float,
    seed: int,
) -> tuple[int, int, int]:
    # md5 -> (has_caps, raw_line_with_newline)
    kept: dict[str, tuple[bool, str]] = {}
    total_read = 0
    replacements = 0

    for path in input_paths:
        file_read = 0
        file_new = 0
        file_replaced = 0
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                file_read += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                md5 = obj.get("md5")
                if not md5:
                    continue
                if not _sample_decision(md5, split, p, seed):
                    continue
                has_caps = bool(obj.get("caps"))
                if not line.endswith("\n"):
                    line = line + "\n"
                prev = kept.get(md5)
                if prev is None:
                    kept[md5] = (has_caps, line)
                    file_new += 1
                elif has_caps and not prev[0]:
                    kept[md5] = (has_caps, line)
                    file_replaced += 1
                    replacements += 1
        total_read += file_read
        print(
            f"[{path.name}] read={file_read:,} new={file_new:,} "
            f"replaced={file_replaced:,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as fout:
        for _, line in kept.values():
            fout.write(line)

    return total_read, len(kept), replacements


def main() -> None:
    download_pe_dataset(DATA_DIR)
    inputs_by_split = iter_pe_files_by_split(DATA_DIR)

    for split in SPLITS:
        inputs = inputs_by_split.get(split, [])
        if not inputs:
            print(f"[warn] no inputs for split={split}; skipping")
            continue
        p = SPLIT_PROBABILITIES[split]
        output_path = OUTPUT_PATHS[split]

        print("=" * 60)
        print(f"Split: {split}  p={p}  seed={SEED} (dedup by md5, prefer non-empty caps)")
        print("Inputs:\n  " + "\n  ".join(str(x) for x in inputs))
        print(f"Output: {output_path}")

        total_read, total_kept, replacements = bernoulli_sample(
            inputs, output_path, split, p, SEED
        )

        print("-" * 60)
        print(f"[{split}] Total rows read:           {total_read:,}")
        print(f"[{split}] Unique samples kept (md5): {total_kept:,}")
        print(f"[{split}] Rows replaced by caps-rich version: {replacements:,}")
        print(f"[{split}] Output size: {os.path.getsize(output_path) / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    main()
