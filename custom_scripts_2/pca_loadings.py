"""
Break down each retained PCA component into the original features.

Inputs : output/Scaling/pca.joblib, Documentation/feature_index_map.json
Outputs: output/Scaling/pca_top_features.csv    (pc, rank, index, block, field, loading, pc_variance_ratio)
         output/Scaling/pca_block_contrib.csv   (rows = PC, cols = block, values = share of squared loadings)
         output/Scaling/pca_feature_importance.csv (variance-weighted importance per feature)

Usage: python custom_scripts_2/pca_loadings.py [--top 15]
"""
import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "output" / "Scaling"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    pca = joblib.load(D / "pca.joblib")
    L = pca.components_                      # (k, 2568)
    evr = pca.explained_variance_ratio_
    meta = pd.DataFrame(json.load(open(ROOT / "Documentation" / "feature_index_map.json"))["entries"])
    assert len(meta) == L.shape[1]

    rows = []
    for i, (w, v) in enumerate(zip(L, evr), 1):
        for r, j in enumerate(np.argsort(-np.abs(w))[:a.top], 1):
            rows.append((f"PC{i}", r, j, meta.block[j], meta.field[j], w[j], v))
    pd.DataFrame(rows, columns=["pc", "rank", "index", "block", "field", "loading",
                                "pc_variance_ratio"]).to_csv(D / "pca_top_features.csv", index=False)

    sq = pd.DataFrame(L ** 2, index=[f"PC{i}" for i in range(1, len(L) + 1)])
    blk = sq.T.groupby(meta.block.values).sum().T
    blk.insert(0, "variance_ratio", evr)
    blk.round(4).to_csv(D / "pca_block_contrib.csv")

    imp = (L ** 2 * evr[:, None]).sum(0) / evr.sum()
    out = meta.assign(importance=imp).sort_values("importance", ascending=False)
    out.to_csv(D / "pca_feature_importance.csv", index=False)
    print(f"k={len(L)}, cum var={evr.sum():.2%}")


if __name__ == "__main__":
    main()
