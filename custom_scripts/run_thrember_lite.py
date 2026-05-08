"""
Run the full thrember_lite pipeline end-to-end:

  1. Vectorize JSONL -> .dat       (thrember.create_vectorized_features)
  2. Build spec.json from a drop  (thrember_lite.FeatureSpec.from_drop_columns)
  3. Train binary LightGBM        (thrember_lite.train_binary)
  4. (optional) Predict one file  (thrember_lite.predict_file)

Setup is assumed done: `thrember` and `thrember_lite` must be importable
(either `pip install -e .` from the repo root, or run with `PYTHONPATH=src`).

Edit the variables at the top of `main()` to point at your data, drop list,
and output directory, then run:

    python custom_scripts/run_thrember_lite.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from thrember import create_vectorized_features
from thrember_lite import FeatureSpec, ModelBundle, predict_file, train_binary


def _step(n: int, title: str) -> None:
    print(f"\n[step {n}] {title}")
    print("-" * 60)


def _dat_files_present(data_dir: Path) -> bool:
    """Cheap check: did `create_vectorized_features` already run here?"""
    needed = [
        "X_train.dat", "y_train.dat",
        "X_test.dat",  "y_test.dat",
        "X_challenge.dat", "y_challenge.dat",
    ]
    return all((data_dir / f).is_file() for f in needed)


def main() -> int:
    # ----- inputs -- edit these to point at your data ----------------------
    DATA_DIR    = Path("ember_data/")
    DROP        = Path("output/Embedded/updated_dropped_features.json")
    OUT         = Path("runs/EMBED01/")
    CONFIG      = Path("examples/lgbm_config.json")
    SEED        = 42
    SOURCE_NOTE = ""                   # free-form provenance string
    REVECTORIZE = True                # force step 1 even if .dat files exist
    PREDICT     = None                 # Path("suspicious.exe") or None
    # -----------------------------------------------------------------------

    # Validate inputs up front so failures point at the right input rather than
    # surfacing as cryptic errors deep inside thrember.
    if not DATA_DIR.is_dir():
        print(f"error: DATA_DIR {DATA_DIR} does not exist or is not a directory", file=sys.stderr)
        return 2
    if not DROP.is_file():
        print(f"error: DROP {DROP} does not exist", file=sys.stderr)
        return 2
    if CONFIG is not None and not CONFIG.is_file():
        print(f"error: CONFIG {CONFIG} does not exist", file=sys.stderr)
        return 2
    if PREDICT is not None and not PREDICT.is_file():
        print(f"error: PREDICT {PREDICT} does not exist", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ step 1
    _step(1, f"Vectorize JSONL -> .dat in {DATA_DIR}")
    if _dat_files_present(DATA_DIR) and not REVECTORIZE:
        print("X_*.dat / y_*.dat already present -- skipping. Set REVECTORIZE = True to force.")
    else:
        create_vectorized_features(DATA_DIR)

    # ------------------------------------------------------------------ step 2
    # spec.json is written by ModelBundle.save in step 3; we only build the
    # FeatureSpec here so train_binary has it.
    _step(2, f"Build spec from {DROP}")
    spec = FeatureSpec.from_drop_columns(DROP, source_note=SOURCE_NOTE)
    print(
        f"original_dim={spec.original_dim}, kept={len(spec.kept_indices)}, "
        f"dropped={spec.original_dim - len(spec.kept_indices)}"
    )

    # ------------------------------------------------------------------ step 3
    _step(3, "Train binary LightGBM")
    params: dict = {}
    if CONFIG is not None:
        with CONFIG.open(encoding="utf-8") as f:
            params = json.load(f)
        print(f"loaded LightGBM params from {CONFIG}")
    else:
        print("no CONFIG; using LightGBM defaults")

    booster = train_binary(DATA_DIR, spec, params, seed=SEED)
    ModelBundle.save(booster, spec, OUT)
    print(f"booster has {booster.num_feature()} features, {booster.num_trees()} trees")

    # ------------------------------------------------------------------ step 4
    if PREDICT is not None:
        _step(4, f"Predict on {PREDICT}")
        bundle = ModelBundle.load(OUT)
        score = predict_file(bundle, PREDICT)
        print(f"{PREDICT}\t{score:.6f}")
    else:
        print("\n(skip step 4 -- PREDICT is None)")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
