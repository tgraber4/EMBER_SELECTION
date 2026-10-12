"""
Figures 03 and 04: family subsets.

  03_well_clustered     families that form compact clusters (WELL_CLUSTERED in common.py)
  04_poorly_clustered   families that overlap each other / benign (POORLY_CLUSTERED in common.py)

Each preset is written twice:
  *_no_benign.png     benign hidden
  *_with_benign.png   benign as a faint gray backdrop

You can also plot any custom list with --families (written to 05_custom/).

Output: output/Scaling/tsne_figures/03_well_clustered/, 04_poorly_clustered/

Usage (from the EMBER2024 root):
    python custom_scripts_2/tsne_figures/fig03_04_family_subsets.py
    python custom_scripts_2/tsne_figures/fig03_04_family_subsets.py --families wacatac njrat
"""

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as c


def plot_subset(emb, groups, colors, counts, families, title, path, with_benign, dpi):
    fig, ax = plt.subplots(figsize=(14, 9))
    if with_benign:
        c.draw_benign(ax, emb, groups, s=3, alpha=0.2)
    for fam in families:
        m = groups == fam
        ax.scatter(emb[m, 0], emb[m, 1], s=9, color=colors[fam], alpha=0.85, linewidths=0,
                   label=f"{fam} ({counts[fam]:,})")
    ax.set_title(title)
    c.style_axes(ax, emb)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=3, frameon=False)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    c.add_common_args(parser)
    parser.add_argument("--families", nargs="*", default=None,
                        help="custom family list (lowercase); writes only 05_custom/")
    args = parser.parse_args()

    emb, groups = c.load(args)
    colors = c.family_colors(groups)
    counts = Counter(groups)

    if args.families:
        jobs = [("05_custom", "custom", "Selected families", c.pick_existing(
            [f.lower() for f in args.families], groups))]
    else:
        jobs = [
            ("03_well_clustered", "well_clustered", "Well-clustered families",
             c.pick_existing(c.WELL_CLUSTERED, groups)),
            ("04_poorly_clustered", "poorly_clustered", "Poorly-clustered families",
             c.pick_existing(c.POORLY_CLUSTERED, groups)),
        ]

    for folder, stem, title, fams in jobs:
        if not fams:
            continue
        out = c.new_figure_dir(args, folder)
        plot_subset(emb, groups, colors, counts, fams, f"t-SNE: {title} (benign hidden)",
                    out / f"{stem}_no_benign.png", False, args.dpi)
        plot_subset(emb, groups, colors, counts, fams, f"t-SNE: {title} (benign in background)",
                    out / f"{stem}_with_benign.png", True, args.dpi)


if __name__ == "__main__":
    main()
