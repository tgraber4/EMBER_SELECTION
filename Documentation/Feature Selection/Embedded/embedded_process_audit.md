# Audit: `train_custom_lgbm_fs.py` Embedded Feature-Selection Process

This document audits the embedded LightGBM feature-selection script
(`custom_scripts/train_custom_lgbm_fs.py`) against
`lightgbm_feature_selection_plan.md`, with attention to whether the
process actually identifies "no gain" features correctly and whether the
hyperparameters in `examples/lgbm_config.json` bias that judgement.

---

## TL;DR

- The **extraction** of zero-gain / zero-split features is correct.
  LightGBM's `feature_importance(...)` is being used the right way and
  the absolute-zero definition is exact.
- The **interpretation** ("these features are noise") is partially
  confounded by the training config. Several hyperparameters push
  marginal-but-real features into the zero-gain bucket.
- The script never retrains on the reduced feature set, so the
  reported "information cost = 0.0000%" is a training-fit metric, not a
  generalization result.

---

## Summary of Recommended Changes

### Hyperparameters (audit-time only — not necessarily for production training)

| Setting | Current | Suggested for audit run | Why |
|---|---|---|---|
| `min_data_in_leaf` | 100 | 20–50 | Lets rare-but-real signals split. Biggest single source of false zero-gain flags. |
| `feature_fraction` | 0.9 | 1.0 | Removes per-tree feature subsampling so every feature is offered every tree. |
| `feature_fraction_bynode` | 0.9 | 1.0 | Same reason at the node level. Stack with above gives ~0.81 effective exposure. |
| seeds | single (`0`) | 3–5 seeds | Run the audit multiple times and **intersect** the absolute-zero sets. Only features that are zero across all runs are confidently useless. |

### Script changes (`train_custom_lgbm_fs.py`)

1. **Add a total wall-clock timer.** Currently only `train_seconds` and
   `proc_seconds` are tracked; vectorization, test scoring, and JSON
   write are untimed.
2. **Skip vectorization when `.dat` files already exist** (or add a
   `--force-vectorize` flag). Right now `vectorize_split` always
   overwrites.
3. **Optional: retrain on the pruned feature set** and report a real
   test-AUC / test-accuracy delta. Without this, "0% information cost"
   only describes the training-set gain ledger.
4. **Document tie-breaking** when `absolute_zero_idx.size > n_drop`:
   the script truncates with `[:n_drop]`, which keeps the
   lowest-numbered indices. That biases the dropped subset toward early
   feature blocks (general file info, header, optional header).
   Consider documenting this or switching to a different deterministic
   tiebreaker (e.g., descending split count among the zeros, or
   index reversed).
5. **Record the run config in the report JSON** (`min_data_in_leaf`,
   `feature_fraction`, `feature_fraction_bynode`, seed) so dropped-
   feature lists from different runs are comparable.

### Plan changes (`lightgbm_feature_selection_plan.md`)

- The plan does not mandate a total-script wall-clock or a post-prune
  retrain. If those are wanted, the plan itself should be amended;
  otherwise the script is faithful to the spec and no script change is
  needed.

---

## Human-Readable Summary of the Conceptual Issues

The script answers the question **"Which features did this particular
trained model never use?"** very accurately. The trap is in treating
that question as if it were the same as **"Which features are useless
to malware classification?"** They are not the same question, and a
few of the LightGBM hyperparameters widen the gap.

### Concept 1 — "Unused by this model" ≠ "Uninformative"

A feature can score zero gain for several different reasons:

1. It really carries no signal (true noise).
2. It carries signal but **another feature carries the same signal**,
   and LightGBM picked that other feature first. The redundant one
   then never gets a chance.
3. It's **constant or near-constant** in the training sample.
4. It's **dominated by the current hyperparameters** — under different
   regularization or subsampling settings, it would have been used.

The script cannot distinguish these cases. All four show up in the
exact same way: gain = 0, split = 0.

### Concept 2 — The leaf-size floor hides rare signals

LightGBM only creates a split if both child leaves keep at least
`min_data_in_leaf` rows. The current config uses **100**. If a feature
only matters for a niche family of malware that's rarer than ~100
samples in the training set, LightGBM is mathematically forbidden from
splitting on it — even when the split would be perfectly clean. That
feature gets reported as zero-gain not because it's noise, but because
the model wasn't allowed to look at it through this lens.

This is the single biggest reason a feature can be "unused but not
useless" on EMBER-style data, where many of the most discriminative PE
indicators are long-tail.

### Concept 3 — Subsampling adds jitter to "zero"

`feature_fraction = 0.9` means each tree only sees 90% of features.
`feature_fraction_bynode = 0.9` means each split candidate only sees
90% of *those*. So every node makes its decision from roughly 81% of
the feature pool. With 500 trees this still gives every feature plenty
of opportunities, but on the **margin** — features that LightGBM was
always going to lose against a stronger one anyway — subsampling pushes
some of them across the line into "0 gain" purely as a coin-flip
artifact. Different seeds will move different features in and out.

### Concept 4 — Class imbalance + leaf floor interact

`is_unbalance: true` reweights the loss to compensate for an
imbalanced class distribution, but `min_data_in_leaf` is enforced on
**raw row counts, not weighted counts**. So minority-class-specific
features face a stricter effective barrier than the reweighted loss
would imply. They're more likely than other features to fall into the
zero-gain bucket for purely procedural reasons.

### Concept 5 — Importance is taken at `best_iteration`

Early stopping picks the iteration where validation loss bottoms out.
Importance is then read at that iteration. This is the correct choice
*for the model that's saved* — but it means features that LightGBM
typically uses in late-stage residual-fitting trees never get counted.
A feature with no gain at iteration 80 might have had gain at iteration
300. The script shows the iteration-80 view because that's where the
model lives.

### Concept 6 — "Information cost = 0%" is a training metric

The information-cost figure compares dropped gain to total gain on the
**training fold**. Even if it's exactly zero, that doesn't prove the
pruned model generalizes equally well. To turn that 0% into a real
guarantee, the script would need to retrain on the reduced feature
matrix and re-score the held-out test set. Right now the printed test
accuracy belongs to the *full-feature* baseline, not to the pruned
candidate.

### Concept 7 — A single seed gives a single realization

The 1,034 absolute-zero features reported on this run are one draw.
With different `seed`, `bagging_seed`, or `feature_fraction_seed`
values, that count typically moves by a few dozen in either direction.
Features that are absolute-zero **across multiple seeded runs** are
the ones safe to call useless. Features that drift in and out are
marginal, not worthless.

---

## What This Means For The Current Run

For the specific report:

```
Total features                : 2568
Absolute zeros (Gain & Split) : 1034
Target drop (N_drop)          : 257
Signal features dropped       : 0
Information cost              : 0.0000%
```

- The 257 features actually dropped are drawn from a pool of 1,034
  unused-by-this-model features, so the prune is comfortably inside a
  4× safety margin even before any of the caveats above.
- The 1,034 figure overstates "true noise" by some unknown amount
  because of Concepts 2, 3, 4, and 7. The *real* count of genuinely
  dispensable features is probably smaller — but still very likely
  larger than 257.
- The current 10% prune is therefore safe to ship; the question of
  *how much further* it could be pushed cannot be answered without the
  multi-seed, relaxed-leaf-size audit described in the recommended
  changes section.
