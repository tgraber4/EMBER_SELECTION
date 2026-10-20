# t-SNE exploration

Checks whether the vectorized, scaled EMBER2024 features cluster by malware family and by benign vs. malware. This is separate from the feature-selection work: all 2,568 features are kept.

## Workflow

Run from the repo root, in this order.

| # | Script (`tsne/Code/`) | Input | Output |
|---|---|---|---|
| 1 | `family_counts.py` | `ember_data/PE_train_sampled.jsonl` | `ember_data/family_counts.csv` |
| 2 | `filter_families.py <input> -t 75` | `PE_train_sampled.jsonl` | `ember_data/PE_train_sampled_min75.jsonl` |
| 3 | `vectorize_scale.py` | `PE_train_sampled_min75.jsonl` | `Scaling Output/` (`X_raw`, `X_quantile`, `X_scaled`, `y`, fitted scalers) |
| 4 | `pca_reduce.py` | `Scaling Output/X_scaled.npy` | `PCA Output/X_pca.npy`, `pca.joblib` |
| 5 | `pca_loadings.py` | `PCA Output/pca.joblib` | `PCA Output/pca_*.csv` |
| 6 | `run_tsne.py` | `PCA Output/X_pca.npy` | `TSNE Output/X_tsne.npy` |
| 7 | `plot_tsne.py`, `tsne_figures/run_all.py` | `TSNE Output/X_tsne.npy` + the filtered JSONL | PNGs in `TSNE Output/` |

Step 1 is run on the unfiltered file to choose the threshold. Re-point its paths at the `_min75` file to inspect the filtered set.

## Key values

| Item | Value |
|---|---|
| Sampled train set | 30,000 rows (17,264 with no family, 12,736 with a family) |
| Distinct families before filtering | 1,189 (567 are singletons) |
| Family threshold | at least 75 samples |
| Families kept | 21 (smallest has 77 samples) |
| Filtered set | 21,762 rows: 14,981 benign, 6,781 malware |
| Features | 2,568 (full vector, none dropped) |
| Scaling | `QuantileTransformer` (uniform, 1000 quantiles), then `StandardScaler` |
| PCA | 53 components, 50.16% cumulative variance (target 0.50) |
| t-SNE | 2 components, perplexity 80, 3000 iterations, `init="pca"`, seed 42 |

## Folders

| Folder | Contents |
|---|---|
| `Code/` | scripts above; `tsne_figures/` has the plotting scripts and family presets |
| `Scaling Output/` | scaled matrices and fitted scalers (git-ignored, about 660 MB) |
| `PCA Output/` | `X_pca.npy`, `pca.joblib` (git-ignored) and the loadings CSVs |
| `TSNE Output/` | `X_tsne.npy`, overview plots, `tsne_figures/` |
| `Documentation/` | [family counts](Documentation/family_counts_breakdown.md), [PCA components](Documentation/pca_component_breakdown.md) |

## Notes

- **Row order** is identical across `X_raw`, `X_scaled`, `X_pca`, `X_tsne`, `y` and the filtered JSONL. The plots rely on this to look up family labels.
- **Filtering drops malware with no family.** Benign rows are always kept, which is why the filtered set has 14,981 benign rows against 17,264 "no family" rows before.
- **Vectorizing uses `thrember`, not `thrember_lite`**, because `thrember_lite` only slices columns by a drop list and nothing is dropped here.
- **Different threshold or variance target:** change `-t` in step 2 or `--min-variance` in step 4, then re-run every later step. The figure presets (`WELL_CLUSTERED`, `POORLY_CLUSTERED` in `tsne_figures/common.py`) name specific families and may need editing if the family set changes.
