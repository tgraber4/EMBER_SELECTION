"""
Run t-SNE on the PCA-reduced features.

Hyperparameters (planned in tsne_plan):
  perplexity=80, learning_rate="auto", max_iter=3000, init="pca" (sklearn default)
Everything else is the sklearn default (Barnes-Hut, euclidean, 2 components).

Inputs  (default output/Scaling/):  X_pca.npy
Outputs (default output/Scaling/):  X_tsne.npy  (N x 2, row order matches X_pca / y)

Usage (from the EMBER2024 root):
    python custom_scripts_2/run_tsne.py
    python custom_scripts_2/run_tsne.py --perplexity 50 --max-iter 2000
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "output" / "Scaling"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--perplexity", type=float, default=80)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = np.load(args.data_dir / "X_pca.npy")
    print(f"[load] X_pca {X.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        max_iter=args.max_iter,
        init="pca",
        random_state=args.seed,
        n_jobs=-1,
        verbose=2,
    )
    t0 = time.time()
    emb = tsne.fit_transform(X).astype(np.float32)
    print(f"[tsne] done in {time.time() - t0:.0f}s, KL divergence {tsne.kl_divergence_:.4f}, "
          f"{tsne.n_iter_} iterations")

    np.save(args.data_dir / "X_tsne.npy", emb)
    print(f"[write] X_tsne {emb.shape} -> {args.data_dir}")


if __name__ == "__main__":
    main()
