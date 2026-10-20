"""
Plot the t-SNE embedding, one color per malware family plus one for benign.

Family labels are re-read from the JSONL; its row order matches X_tsne.npy.

To hide groups, add their names to HIDE below (or pass --hide on the command line).
Use "benign" for benign samples; families are matched case-insensitively.
Example:  HIDE = ["benign", "berbew"]

Inputs  (default custom/tsne/TSNE Output/):  X_tsne.npy
Output  (default custom/tsne/TSNE Output/):  tsne_plot.png

Usage (from the EMBER2024 root):
    python custom/tsne/Code/plot_tsne.py
    python custom/tsne/Code/plot_tsne.py --hide benign berbew
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- edit this array to hide groups ----------------------------------------
HIDE: list[str] = []
# ----------------------------------------------------------------------------

TSNE_DIR = Path(__file__).resolve().parent.parent
ROOT = TSNE_DIR.parent.parent
DEFAULT_DIR = TSNE_DIR / "TSNE Output"
DEFAULT_JSONL = ROOT / "ember_data" / "PE_train_sampled_min75.jsonl"
BENIGN = "benign"


def read_groups(path: Path) -> np.ndarray:
    """Group name per row: 'benign' or the (lowercase) family."""
    groups = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            groups.append(BENIGN if obj.get("label") == 0 else str(obj.get("family")).lower())
    return np.array(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--hide", nargs="*", default=None,
                        help="groups to hide (overrides HIDE in the script)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    hide = {h.lower() for h in (args.hide if args.hide is not None else HIDE)}

    emb = np.load(args.data_dir / "X_tsne.npy")
    groups = read_groups(args.jsonl)
    if len(groups) != len(emb):
        raise SystemExit(f"row mismatch: {len(groups)} labels vs {len(emb)} embedding rows")

    counts = Counter(groups)
    unknown = hide - set(counts)
    if unknown:
        print(f"[warn] not found, ignoring: {sorted(unknown)}")
        print(f"[warn] available: {sorted(map(str, counts))}")

    # Fixed color per group (by size, so hiding one doesn't recolor the rest)
    families = [g for g, _ in counts.most_common() if g != BENIGN]
    palette = list(plt.cm.tab20.colors) + list(plt.cm.Dark2.colors)
    colors = {g: palette[i % len(palette)] for i, g in enumerate(families)}
    colors[BENIGN] = (0.75, 0.75, 0.75)

    fig, ax = plt.subplots(figsize=(14, 10))
    # Largest groups first so small families are drawn on top
    for g, n in counts.most_common():
        if g in hide:
            continue
        m = groups == g
        ax.scatter(emb[m, 0], emb[m, 1], s=4 if g == BENIGN else 7, color=colors[g],
                   alpha=0.4 if g == BENIGN else 0.8, linewidths=0, label=f"{g} ({n:,})")
    ax.set_title("t-SNE of PCA-reduced EMBER features" + (f" (hidden: {', '.join(sorted(hide))})" if hide else ""))
    ax.set_xticks([])
    ax.set_yticks([])
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=3, frameon=False)
    for h in leg.legend_handles:
        h.set_alpha(1)

    out = args.output or args.data_dir / "tsne_plot.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
