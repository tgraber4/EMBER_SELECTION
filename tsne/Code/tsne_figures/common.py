"""
Shared helpers for the t-SNE figure scripts in this folder.

Inputs  (default tsne/TSNE Output/):  X_tsne.npy
        (default ember_data/):      PE_train_sampled_min75.jsonl  (row order matches X_tsne.npy)
Outputs (default tsne/TSNE Output/tsne_figures/<figure folder>/)

Edit WELL_CLUSTERED / POORLY_CLUSTERED below to change the family presets
used by figures 03 and 04.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

TSNE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT = TSNE_DIR.parent
DEFAULT_DATA_DIR = TSNE_DIR / "TSNE Output"
DEFAULT_JSONL = ROOT / "ember_data" / "PE_train_sampled_min75.jsonl"
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "tsne_figures"
BENIGN = "benign"

# ---- family presets (lowercase), from neighbor purity in 01_family_grid/family_order.csv ----
# well: purity >= 0.80; poor: purity < 0.55; flystudio, blackmoon, sinowal sit in between and are left out.
WELL_CLUSTERED = [
    "sfone", "berbew", "grandoreiro", "cosmu", "xmrig", "tempedreve",
    "glupteba", "strelastealer", "expiro", "upatre",
]
POORLY_CLUSTERED = [
    "ipamor", "barys", "agenttesla", "wacatac", "redline", "floxif", "virut", "njrat",
]
# -----------------------------------------------------------------------------

BENIGN_COLOR = "#b8b8b8"
MALWARE_COLOR = "#d62728"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="parent folder; each script writes to its own subfolder")
    parser.add_argument("--dpi", type=int, default=200)


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


def load(args) -> tuple[np.ndarray, np.ndarray]:
    emb = np.load(args.data_dir / "X_tsne.npy")
    groups = read_groups(args.jsonl)
    if len(groups) != len(emb):
        raise SystemExit(f"row mismatch: {len(groups)} labels vs {len(emb)} embedding rows")
    return emb, groups


def family_colors(groups: np.ndarray) -> dict[str, tuple]:
    """Fixed color per family (ordered by size) so colors are stable across figures."""
    import matplotlib.pyplot as plt

    counts = Counter(groups)
    families = [g for g, _ in counts.most_common() if g != BENIGN]
    palette = list(plt.cm.tab20.colors) + list(plt.cm.Dark2.colors)
    colors = {g: palette[i % len(palette)] for i, g in enumerate(families)}
    colors[BENIGN] = (0.72, 0.72, 0.72)
    return colors


def neighbor_purity(emb: np.ndarray, groups: np.ndarray, k: int = 15) -> dict[str, float]:
    """Per family: mean fraction of the k nearest t-SNE neighbors in the same family.

    Only used to order panels from tightest to most scattered; neighbors are
    taken in the 2-D embedding, so this is a visual-compactness score.
    """
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]
    same = (groups[idx] == groups[:, None]).mean(axis=1)
    return {g: float(same[groups == g].mean()) for g in set(groups)}


def pick_existing(wanted: list[str], groups: np.ndarray) -> list[str]:
    present = set(groups)
    missing = [w for w in wanted if w not in present]
    if missing:
        print(f"[warn] families not found, skipped: {missing}")
    return [w for w in wanted if w in present]


def new_figure_dir(args, name: str) -> Path:
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def draw_benign(ax, emb, groups, s=3, alpha=0.25, color=BENIGN_COLOR) -> None:
    m = groups == BENIGN
    ax.scatter(emb[m, 0], emb[m, 1], s=s, color=color, alpha=alpha, linewidths=0, rasterized=True)


def style_axes(ax, emb: np.ndarray) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    pad = 0.02
    for lim, setter in ((emb[:, 0], ax.set_xlim), (emb[:, 1], ax.set_ylim)):
        span = lim.max() - lim.min()
        setter(lim.min() - pad * span, lim.max() + pad * span)
