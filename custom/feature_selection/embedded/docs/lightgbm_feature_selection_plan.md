# LightGBM Embedded Feature Selection Strategy (10% Reduction Plan)

This plan outlines a systematic, model-embedded approach to feature selection using LightGBM. The goal is to prune the bottom 10% of features based on their contribution to the model while tracking performance metrics and the "information cost" of the reduction.

> **Hard cap.** The 10% target is a strict ceiling, not a floor. The selection process drops *exactly* `N_drop = floor/round(T * 0.10)` features and never more — even if the absolute-zero set is larger than the quota.

---

## Phase 1: Training and Time Tracking
The objective of this phase is to establish a high-quality baseline and measure the computational overhead of the training process.

1.  **Hyperparameter Definition**: Configure the LightGBM model with a sufficient number of estimators (e.g., `n_estimators=1000`) and appropriate complexity (e.g., `num_leaves`, `learning_rate`) to ensure the model thoroughly explores the feature space.
2.  **Quota Calculation**:
    * Determine total feature count ($T$).
    * Calculate the target number of features to drop ($N_{drop} = T 	imes 0.10$).
3.  **Timing Start**: Record the exact system timestamp immediately prior to initiating the `.fit()` method.
4.  **Baseline Training**: Execute a single training pass. Ensure the model is trained until convergence (using early stopping if a validation set is available).
5.  **Timing End**: Record the timestamp upon completion to calculate **Total Training Duration**.

---

## Phase 2: Metric Extraction and Zero-Value Audit
LightGBM logs split frequency and information gain during training. This phase extracts those logs for analysis.

1.  **Extract Primary Metrics**:
    * **Split (Frequency)**: The number of times each feature was used to partition the data.
    * **Gain (Total)**: The cumulative reduction of the loss function attributed to each feature.
2.  **Threshold Identification**: Identify and count features that fall into the following "zero-utility" categories:
    * **Features with 0 Gain**: Features that never improved the model's objective.
    * **Features with 0 Splits**: Features that were never selected as a decision node.
    * **Absolute Zeros**: The intersection where both Gain and Split are exactly 0.
3.  **Data Structuring**: Create a dataframe or table mapping each Feature Name to its Gain score, Split count, and "Zero-Status" flags.

---

## Phase 3: The "Final Cut" Selection Logic
This phase applies a tiered priority system to reach the 10% removal target while minimizing the loss of predictive signal.

1.  **Step 1: Primary Pruning (Absolute Zeros)**:
    * Identify all features where `Gain == 0` AND `Split == 0`.
    * Move these features into the "Removed Features" list.
2.  **Step 2: Quota Assessment (Hard Cap)**:
    * The "Removed Features" list MUST end with exactly $N_{drop}$ entries — never more.
    * If the absolute-zero set is **larger** than $N_{drop}$, truncate it: take the first $N_{drop}$ absolute zeros (deterministic ordering, e.g. ascending feature index) and stop. The remaining absolute-zero features stay in the active set even though they contribute no signal.
    * If the absolute-zero set is **equal to** $N_{drop}$, the selection is complete.
    * If the absolute-zero set is **smaller than** $N_{drop}$, proceed to Step 3.
3.  **Step 3: Ranking by Cumulative Gain**:
    * For all features remaining in the active set, rank them in ascending order (lowest to highest) based strictly on their **Total Gain** score.
4.  **Step 4: Top-Up Removal**:
    * Iteratively move features from the bottom of the Gain-ranked list into the "Removed Features" list until the total count of removed features exactly equals $N_{drop}$.
    * The final removed-feature count is exactly $N_{drop}$. The 10% ceiling is non-negotiable.

---

## Phase 4: Impact Reporting and Performance Audit
This phase provides a post-analysis report to quantify exactly what was removed and the theoretical impact on the model.

1.  **Information Loss Calculation**:
    * Calculate the **Total Global Gain** (sum of Gain for all original features).
    * Calculate the **Sum of Dropped Gain** (sum of Gain for the 10% removed features).
    * Report the ratio: $rac{	ext{Sum of Dropped Gain}}{	ext{Total Global Gain}} 	imes 100$. This represents the percentage of model "intelligence" sacrificed.
2.  **Threshold Summary Table**:
    * **Total Features**: Count of original columns.
    * **Target Drop ($N_{drop}$)**: The calculated 10% goal.
    * **Count of 0-Gain Features**: Total features that provided zero loss reduction.
    * **Count of 0-Split Features**: Total features never used in a tree.
    * **Count of Absolute Zeros**: Total features that met both zero-criteria.
    * **Signal Features Dropped**: The count of features removed that had a Gain $> 0$ (to satisfy the 10% requirement).
3.  **Execution Time Report**:
    * **Training Time**: The duration recorded in Phase 1.
    * **Processing Time**: The time taken to extract metrics and execute the selection logic.
