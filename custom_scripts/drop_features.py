"""Vectorize raw EMBER JSONL samples and drop a list of feature indices.

Inputs:
  --dropped : path to dropped_features.csv or dropped_features.json
              (must contain an 'index' column/field)
  --in      : input JSONL with raw EMBER samples (one JSON object per line)
  --out     : output JSONL path

Output JSONL: one object per line with
  {"sha256": ..., "label": ..., "vector": [reduced floats...]}
plus any other top-level scalar fields preserved from the input
(md5, sha1, label, file_type, family, ...).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from thrember import PEFeatureExtractor


PASSTHROUGH_KEYS = (
    "md5", "sha1", "sha256", "tlsh",
    "first_submission_date", "last_analysis_date", "detection_ratio",
    "label", "file_type", "family", "family_confidence",
    "behavior", "file_property", "packer", "exploit", "group",
)


def load_drop_indices(path: Path) -> list[int]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return sorted({int(row["index"]) for row in data})
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sorted({int(row["index"]) for row in reader})
    raise ValueError(f"unsupported dropped-features format: {suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dropped", required=True, type=Path)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    drop_idx = load_drop_indices(args.dropped)
    extractor = PEFeatureExtractor()
    dim = extractor.dim

    bad = [i for i in drop_idx if i < 0 or i >= dim]
    if bad:
        print(f"error: drop indices out of range [0,{dim}): {bad[:10]}{'...' if len(bad) > 10 else ''}", file=sys.stderr)
        return 2

    keep_mask = np.ones(dim, dtype=bool)
    keep_mask[drop_idx] = False
    keep_idx = np.flatnonzero(keep_mask)
    print(f"dim {dim} -> {keep_idx.size} ({len(drop_idx)} dropped)")

    n = 0
    with args.inp.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            vec = extractor.process_raw_features(raw)
            reduced = vec[keep_idx].tolist()
            out = {k: raw[k] for k in PASSTHROUGH_KEYS if k in raw}
            out["vector"] = reduced
            fout.write(json.dumps(out) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  processed {n}")
    print(f"done: {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
