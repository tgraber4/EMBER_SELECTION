"""
Vectorize and scale the family-filtered training set.

Steps:
  1. Read every row of PE_train_sampled_min75.jsonl (no subsampling)
  2. Vectorize each row the way EMBER does (PEFeatureExtractor.process_raw_features,
     the same call thrember.model.vectorize makes) -> full 2,568-wide vector
  3. QuantileTransformer on all features, then StandardScaler on all features

Uses `thrember` rather than `thrember_lite`: thrember_lite only slices columns of an
already-vectorized .dat file by a drop list, and here all 2,568 features are kept.

Outputs (default custom/tsne/Scaling Output/):
  X_raw.npy, X_quantile.npy, X_scaled.npy, y.npy, quantile_transformer.joblib,
  standard_scaler.joblib

Usage (from the EMBER2024 root):
    python custom/tsne/Code/vectorize_scale.py
    python custom/tsne/Code/vectorize_scale.py -n 20000 --seed 42
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tqdm
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from thrember import PEFeatureExtractor

TSNE_DIR = Path(__file__).resolve().parent.parent
ROOT = TSNE_DIR.parent.parent
DEFAULT_INPUT = ROOT / "ember_data" / "PE_train_sampled_min75.jsonl"
DEFAULT_OUTPUT = TSNE_DIR / "Scaling Output"


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    print(f"[read] {len(lines):,} rows")
    return lines


def vectorize(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    extractor = PEFeatureExtractor()
    X = np.empty((len(lines), extractor.dim), dtype=np.float32)
    y = np.empty(len(lines), dtype=np.int32)
    for i, line in enumerate(tqdm.tqdm(lines, desc="vectorize")):
        raw = json.loads(line)
        X[i] = extractor.process_raw_features(raw)
        label = raw["label"]
        y[i] = -1 if label is None else label
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-quantiles", type=int, default=1000,
                        help="QuantileTransformer n_quantiles (default 1000, sklearn's default)")
    args = parser.parse_args()

    lines = read_lines(args.input)
    X, y = vectorize(lines)
    del lines
    print(f"[vectorize] X {X.shape}, y {y.shape} "
          f"(benign {(y == 0).sum():,}, malware {(y == 1).sum():,}, unlabeled {(y == -1).sum():,})")

    # QuantileTransformer first (all features), then StandardScaler (all features)
    qt = QuantileTransformer(n_quantiles=min(args.n_quantiles, len(X)),
                             random_state=args.seed, copy=True)
    X_q = qt.fit_transform(X).astype(np.float32)
    print("[scale] QuantileTransformer done")

    ss = StandardScaler()
    X_s = ss.fit_transform(X_q).astype(np.float32)
    print(f"[scale] StandardScaler done "
          f"({int((ss.scale_ == 1.0).sum())} columns had zero variance after quantile transform)")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_raw.npy", X)
    np.save(out / "X_quantile.npy", X_q)
    np.save(out / "X_scaled.npy", X_s)
    np.save(out / "y.npy", y)
    joblib.dump(qt, out / "quantile_transformer.joblib")
    joblib.dump(ss, out / "standard_scaler.joblib")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
