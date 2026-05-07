"""Inference on sliced EMBER vectors.

Fills the gap noted in the plan: `thrember.predict_sample` (model.py:409-416) always
extracts a full-width vector. Here we extract full width via the same extractor and
then slice to `spec.kept_indices` before handing to the booster.

`ModelBundle` is a save/load utility (Decision #12) — `train_binary` returns a
plain `lgb.Booster`; the user calls `ModelBundle.save(booster, spec, dir)` to
package the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import lightgbm as lgb

from .spec import FeatureSpec
from .data import apply_spec


_MODEL_FILENAME = "model.txt"
_SPEC_FILENAME = "spec.json"


@dataclass
class ModelBundle:
    """Pairs a trained booster with its FeatureSpec for reproducible inference."""

    booster: lgb.Booster
    spec: FeatureSpec

    # ----- save/load ---------------------------------------------------------

    @staticmethod
    def save(booster: lgb.Booster, spec: FeatureSpec, out_dir: Path | str) -> None:
        """Write `model.txt` and `spec.json` into `out_dir`. Creates the dir."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out / _MODEL_FILENAME))
        spec.to_json(out / _SPEC_FILENAME)
        print(f"thrember_lite: wrote {out / _MODEL_FILENAME} and {out / _SPEC_FILENAME}")

    @classmethod
    def load(cls, in_dir: Path | str) -> "ModelBundle":
        """Load both files. Asserts extractor width matches spec.original_dim."""
        in_ = Path(in_dir)
        model_path = in_ / _MODEL_FILENAME
        spec_path = in_ / _SPEC_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not spec_path.is_file():
            raise FileNotFoundError(spec_path)

        booster = lgb.Booster(model_file=str(model_path))
        spec = FeatureSpec.from_json(spec_path)

        # Width-safety check — fail loudly if thrember's feature width changed since training
        from thrember import PEFeatureExtractor
        extractor_dim = PEFeatureExtractor().dim
        if extractor_dim != spec.original_dim:
            raise RuntimeError(
                f"PEFeatureExtractor().dim = {extractor_dim} but spec.original_dim = "
                f"{spec.original_dim}; thrember version mismatch since model was trained?"
            )

        return cls(booster=booster, spec=spec)


def predict_file(bundle: ModelBundle, path_or_bytes: Path | str | bytes) -> float:
    """Score a single PE sample. Accepts a path or raw bytes.

    Pipeline: read bytes → extract full 2568-dim vector → slice to
    `spec.kept_indices` → `booster.predict`.
    """
    from thrember import PEFeatureExtractor
    extractor = PEFeatureExtractor()
    bytez = _coerce_bytes(path_or_bytes)
    vec = np.asarray(extractor.feature_vector(bytez), dtype=np.float32)
    vec_kept = apply_spec(vec, bundle.spec)
    return float(bundle.booster.predict([vec_kept])[0])


def predict_batch(
    bundle: ModelBundle,
    paths_or_bytes: Iterable[Path | str | bytes],
) -> np.ndarray:
    """Score many samples with one extractor instance and one booster call.

    Returns a 1-D float array of scores in `[0, 1]`, in input order.
    """
    from thrember import PEFeatureExtractor
    extractor = PEFeatureExtractor()
    kept = bundle.spec.kept_indices

    paths_list = list(paths_or_bytes)
    out = np.empty((len(paths_list), len(kept)), dtype=np.float32)
    for i, item in enumerate(paths_list):
        bytez = _coerce_bytes(item)
        full = np.asarray(extractor.feature_vector(bytez), dtype=np.float32)
        out[i] = full[kept]

    return np.asarray(bundle.booster.predict(out), dtype=np.float32)


def _coerce_bytes(path_or_bytes: Path | str | bytes) -> bytes:
    """Read a file or pass through raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return bytes(path_or_bytes)
    p = Path(path_or_bytes)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.read_bytes()
