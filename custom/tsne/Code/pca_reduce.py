"""
Compress the scaled feature matrix with PCA for t-SNE.

Fits a full PCA on X_scaled.npy, then keeps the smallest number of components whose
cumulative explained variance is at least --min-variance (default 0.50).

Inputs  (default custom/tsne/Scaling Output/):  X_scaled.npy
Outputs (default custom/tsne/PCA Output/):  X_pca.npy, pca.joblib

Usage (from the EMBER2024 root):
    python custom/tsne/Code/pca_reduce.py
    python custom/tsne/Code/pca_reduce.py --min-variance 0.6
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

TSNE_DIR = Path(__file__).resolve().parent.parent
ROOT = TSNE_DIR.parent.parent
DEFAULT_DIR = TSNE_DIR / "Scaling Output"
OUT_DIR = TSNE_DIR / "PCA Output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--min-variance", type=float, default=0.50,
                        help="minimum cumulative explained variance to keep (default 0.50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.data_dir / "X_scaled.npy")
    print(f"[load] X {X.shape}")

    # Full fit (n_components=None) so the cumulative curve is exact
    pca = PCA(svd_solver="full", random_state=args.seed)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, args.min_variance) + 1)
    print(f"[pca] {k} components reach {cum[k - 1]:.2%} variance (target >= {args.min_variance:.0%})")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        if cum[-1] >= t:
            print(f"[pca]   {t:.0%} variance needs {int(np.searchsorted(cum, t) + 1)} components")

    # Keep only the first k components (PCA components are ordered, so this equals fitting k)
    X_pca = (pca.transform(X)[:, :k]).astype(np.float32)
    pca.components_ = pca.components_[:k]
    pca.explained_variance_ = pca.explained_variance_[:k]
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:k]
    pca.singular_values_ = pca.singular_values_[:k]
    pca.n_components_ = k

    np.save(OUT_DIR / "X_pca.npy", X_pca)
    joblib.dump(pca, OUT_DIR / "pca.joblib")
    print(f"[write] X_pca {X_pca.shape} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
