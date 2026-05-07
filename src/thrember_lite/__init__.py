"""thrember_lite — train and predict on stripped EMBER vectors.

Companion library to `thrember`. Reuses thrember's vectorized .dat files unchanged
and slices columns at load time so a single full-width dataset supports any number
of feature-selection experiments.

See `Documentation/thrember_lite_plan.md` for the design.
"""

from .spec import FeatureSpec, DEFAULT_CATEGORICAL
from .data import read_vectorized_features, apply_spec
from .train import train_binary
from .predict import ModelBundle, predict_file, predict_batch

__all__ = [
    "FeatureSpec",
    "DEFAULT_CATEGORICAL",
    "read_vectorized_features",
    "apply_spec",
    "train_binary",
    "ModelBundle",
    "predict_file",
    "predict_batch",
]
