# Family Counts Breakdown

Source: `ember_data/family_counts.csv` (from `custom/tsne/Code/family_counts.py`, run on `ember_data/PE_train_sampled.jsonl`).
## Summary

| Metric | Value |
|---|---|
| Total samples | 30,000 |
| No family (`(none)`, benign) | 17,264 (57.5%) |
| Samples with a family | 12,736 (42.5%) |
| Distinct named families | 1,189 |
| Singleton families (1 sample) | 567 |
| Median samples per family | 2 |
| Largest family | berbew (1,681) |

## Families by minimum size

| Min samples | Families | Samples covered |
|---|---|---|
| ≥ 1 | 1,189 | 12,736 |
| ≥ 2 | 622 | 12,169 |
| ≥ 5 | 290 | 11,279 |
| ≥ 10 | 156 | 10,402 |
| ≥ 25 | 80 | 9,234 |
| ≥ 50 | 38 | 7,831 |
| ≥ 75 | 21 | 6,781 |
| ≥ 100 | 15 | 6,258 |
| ≥ 500 | 4 | 3,885 |

The distribution is heavily long-tailed: about half of all families have 1–2 samples, while 21 families have ≥ 75.

## Families by sample-count range

| Samples per family | Families | % of families | Samples covered |
|---|---|---|---|
| 1 | 567 | 47.7% | 567 |
| 2 | 177 | 14.9% | 354 |
| 3–5 | 198 | 16.7% | 751 |
| 6–10 | 98 | 8.2% | 732 |
| 11–25 | 74 | 6.2% | 1,223 |
| 26–50 | 39 | 3.3% | 1,378 |
| 51–75 | 15 | 1.3% | 950 |
| 76–100 | 6 | 0.5% | 523 |
| 101–500 | 11 | 0.9% | 2,373 |
| 501+ | 4 | 0.3% | 3,885 |
| **Total** | **1,189** | 100% | **12,736** |

## `PE_train_sampled_min75.jsonl`

- Total samples: 21,762
- Distinct named families: **21** (plus `(none)` with 14,981 samples → 22 distinct values)
- Smallest named family: 77 samples
