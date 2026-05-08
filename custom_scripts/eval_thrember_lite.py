"""
Evaluate a thrember_lite ModelBundle on the test and challenge sets.

Mirrors examples/eval_lgbm.py but for sliced (thrember_lite) models — loads a
ModelBundle (model.txt + spec.json), reads .dat files through
read_vectorized_features so the column slice is applied automatically, and
computes ROC AUC, PR AUC, TPR @ 1% FPR, plus an ROC-curve PDF for each set.

Setup is assumed done: `thrember` and `thrember_lite` must be importable.

Edit MODEL_DIR / DATA_DIR, then run:

    python custom_scripts/eval_thrember_lite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: no display needed, saves PDF only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
from sklearn.metrics import (    # noqa: E402
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from thrember_lite import ModelBundle, read_vectorized_features  # noqa: E402


def _step(n: int, title: str) -> None:
    print(f"\n[step {n}] {title}")
    print("-" * 60)


def _eval(booster, X: np.ndarray, y: np.ndarray, *,
          label: str, fpr_target: float = 0.01) -> dict:
    """Compute ROC AUC, PR AUC, and TPR @ fpr_target. Caller filters y == -1."""
    y_pred = booster.predict(X)
    roc_auc_v = roc_auc_score(y, y_pred)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    pr_auc_v = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y, y_pred)
    idx = int(np.argmin(np.abs(fpr - fpr_target)))
    tpr_at_target = float(tpr[idx])

    print(
        f"{label:>10s}: ROC AUC = {roc_auc_v:.4f}   "
        f"PR AUC = {pr_auc_v:.4f}   "
        f"TPR @ {fpr_target:.0%} FPR = {tpr_at_target:.4f}"
    )
    return {
        "roc_auc": roc_auc_v,
        "pr_auc": pr_auc_v,
        "fpr": fpr,
        "tpr": tpr,
        "tpr_at_target": tpr_at_target,
        "fpr_target": fpr_target,
    }


def _save_roc_plot(metrics: dict, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fpr_target = metrics["fpr_target"]
    tpr_at_target = metrics["tpr_at_target"]

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.plot(metrics["fpr"], metrics["tpr"], color="black")
    plt.xlim(0.00005, 1.0)
    plt.ylim(0.65, 1.02)
    plt.xscale("log")
    plt.plot(
        [fpr_target, fpr_target, 0],
        [0, tpr_at_target, tpr_at_target],
        color="red", linestyle="--",
        label=f"TPR @ {fpr_target:.0%} FPR = {tpr_at_target:.4f}",
    )
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"saved ROC curve to {out_path}")


def main() -> int:
    # ----- inputs -- edit these to point at your model + data ---------------
    MODEL_DIR      = Path("runs/SHAP01/")
    DATA_DIR       = Path("ember_data/")
    EVAL_CHALLENGE = False            # set False to skip the challenge set
    OUT_PLOT_TEST  = MODEL_DIR / "roc_test.pdf"
    OUT_PLOT_CHAL  = MODEL_DIR / "roc_challenge.pdf"
    # -----------------------------------------------------------------------

    if not MODEL_DIR.is_dir():
        print(f"error: MODEL_DIR {MODEL_DIR} does not exist", file=sys.stderr)
        return 2
    if not DATA_DIR.is_dir():
        print(f"error: DATA_DIR {DATA_DIR} does not exist", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ step 1
    _step(1, f"Load ModelBundle from {MODEL_DIR}")
    bundle = ModelBundle.load(MODEL_DIR)
    print(
        f"booster: {bundle.booster.num_feature()} features, "
        f"{bundle.booster.num_trees()} trees"
    )
    print(
        f"spec:    original_dim={bundle.spec.original_dim}, "
        f"kept={len(bundle.spec.kept_indices)}"
    )

    # ------------------------------------------------------------------ step 2
    _step(2, f"Evaluate on test set ({DATA_DIR / 'X_test.dat'})")
    X_test, y_test = read_vectorized_features(DATA_DIR, "test", bundle.spec)
    test_labeled = y_test != -1
    print(f"{y_test.shape[0]} test rows ({int(test_labeled.sum())} labeled)")
    test_metrics = _eval(
        bundle.booster, X_test[test_labeled], y_test[test_labeled],
        label="test",
    )
    _save_roc_plot(test_metrics, OUT_PLOT_TEST, "Test set ROC")

    # ------------------------------------------------------------------ step 3
    chal_x_path = DATA_DIR / "X_challenge.dat"
    if not EVAL_CHALLENGE:
        print("\n(skip step 3 -- EVAL_CHALLENGE is False)")
    elif not chal_x_path.is_file():
        print(f"\n(skip step 3 -- {chal_x_path} not found)")
    else:
        _step(3, f"Evaluate on challenge set ({chal_x_path})")
        X_chal, y_chal = read_vectorized_features(DATA_DIR, "challenge", bundle.spec)
        chal_labeled = y_chal != -1

        # EMBER2024 protocol: challenge contains only malicious samples; we
        # concatenate the test-set benigns to give AUC a realistic decision
        # boundary against new malware.
        test_benign_mask = (y_test == 0)
        X_full = np.concatenate(
            [X_test[test_benign_mask], X_chal[chal_labeled]], axis=0,
        )
        y_full = np.concatenate(
            [y_test[test_benign_mask], y_chal[chal_labeled]], axis=0,
        )
        print(
            f"{y_chal.shape[0]} challenge rows ({int(chal_labeled.sum())} labeled), "
            f"+ {int(test_benign_mask.sum())} test-benign for AUC"
        )
        chal_metrics = _eval(
            bundle.booster, X_full, y_full, label="challenge",
        )
        _save_roc_plot(chal_metrics, OUT_PLOT_CHAL, "Challenge set ROC")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
