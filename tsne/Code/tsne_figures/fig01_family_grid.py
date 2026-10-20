"""
Figure 01: small-multiples grid, one panel per malware family.

Every panel shows all points in light gray and highlights one family in color.
Panels are ordered from most compact to most scattered, using the fraction of
each point's 15 nearest t-SNE neighbors that share its family (shown in the
panel title). Axes are identical across panels so they can be compared.

Output: tsne/TSNE Output/tsne_figures/01_family_grid/family_grid.png
        tsne/TSNE Output/tsne_figures/01_family_grid/family_order.csv

Usage (from the EMBER2024 root):
    python tsne/Code/tsne_figures/fig01_family_grid.py
    python tsne/Code/tsne_figures/fig01_family_grid.py --cols 5
"""

import argparse
import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    c.add_common_args(parser)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("-k", type=int, default=15, help="neighbors for the compactness score")
    args = parser.parse_args()

    emb, groups = c.load(args)
    colors = c.family_colors(groups)
    counts = Counter(groups)
    purity = c.neighbor_purity(emb, groups, args.k)

    families = sorted((g for g in counts if g != c.BENIGN), key=lambda g: -purity[g])
    rows = math.ceil(len(families) / args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(3.4 * args.cols, 3.2 * rows))
    axes = axes.ravel()

    for ax, fam in zip(axes, families):
        m = groups == fam
        ax.scatter(emb[~m, 0], emb[~m, 1], s=1.5, color="#d9d9d9", alpha=0.5,
                   linewidths=0, rasterized=True)
        ax.scatter(emb[m, 0], emb[m, 1], s=7, color=colors[fam], linewidths=0)
        ax.set_title(f"{fam} (n={counts[fam]:,}, purity {purity[fam]:.2f})", fontsize=9)
        c.style_axes(ax, emb)
    for ax in axes[len(families):]:
        ax.axis("off")

    fig.suptitle(
        f"t-SNE by family, ordered from most compact to most scattered "
        f"(purity = share of {args.k} nearest t-SNE neighbors in the same family)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = c.new_figure_dir(args, "01_family_grid")
    fig.savefig(out / "family_grid.png", dpi=args.dpi, bbox_inches="tight")
    with (out / "family_order.csv").open("w", encoding="utf-8") as f:
        f.write("rank,family,n,neighbor_purity\n")
        for i, fam in enumerate(families, 1):
            f.write(f"{i},{fam},{counts[fam]},{purity[fam]:.4f}\n")
    print(f"[write] {out / 'family_grid.png'}")
    print(f"[write] {out / 'family_order.csv'}")


if __name__ == "__main__":
    main()
