"""Argparse glue for thrember_lite. Run as `python -m thrember_lite.cli ...`.

Three subcommands matching the plan:
  build-spec  — bridge dropped_features.json into a FeatureSpec (spec.json)
  train       — train_binary on a data dir + spec, save ModelBundle to out_dir
  predict     — load a ModelBundle, score one file, print "{path}\t{score}"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .spec import FeatureSpec
from .train import train_binary
from .predict import ModelBundle, predict_file


def _cmd_build_spec(args: argparse.Namespace) -> int:
    spec = FeatureSpec.from_drop_columns(
        args.drop,
        index_map_path=args.index_map,
        source_note=args.source_note or "",
    )
    spec.to_json(args.out)
    print(
        f"thrember_lite: wrote {args.out} "
        f"(original_dim={spec.original_dim}, kept={len(spec.kept_indices)}, "
        f"dropped={spec.original_dim - len(spec.kept_indices)})"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    spec = FeatureSpec.from_json(args.spec)
    params: dict = {}
    if args.config is not None:
        with Path(args.config).open(encoding="utf-8") as f:
            params = json.load(f)
    booster = train_binary(args.data_dir, spec, params, seed=args.seed)
    ModelBundle.save(booster, spec, args.out_dir)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    bundle = ModelBundle.load(args.model_dir)
    score = predict_file(bundle, args.file)
    print(f"{args.file}\t{score:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m thrember_lite.cli",
        description="thrember_lite: train and predict on stripped EMBER vectors.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build-spec --------------------------------------------------------------
    bs = sub.add_parser(
        "build-spec",
        help="Build spec.json from a dropped_features.json (Layer A).",
    )
    bs.add_argument("--drop", required=True, type=Path,
                    help="Layer-A JSON of indices to drop (e.g. dropped_features.json).")
    bs.add_argument("--out", required=True, type=Path,
                    help="Output spec.json path.")
    bs.add_argument("--index-map", type=Path, default=None,
                    help="(optional) feature_index_map.json; otherwise derive from "
                         "PEFeatureExtractor in-process.")
    bs.add_argument("--source-note", default="",
                    help="Free-form provenance string stored in spec.json's source block.")
    bs.set_defaults(func=_cmd_build_spec)

    # train -------------------------------------------------------------------
    tr = sub.add_parser(
        "train",
        help="Train a binary LightGBM model and save a ModelBundle.",
    )
    tr.add_argument("data_dir", type=Path, help="Directory containing X_train.dat / y_train.dat.")
    tr.add_argument("spec", type=Path, help="spec.json built by `build-spec`.")
    tr.add_argument("out_dir", type=Path, help="Output directory for model.txt + spec.json.")
    tr.add_argument("--config", type=Path, default=None,
                    help="LightGBM params JSON. Optional — empty params are valid.")
    tr.add_argument("--seed", type=int, default=None,
                    help="Pin val-split RNG and LightGBM's three RNG knobs. "
                         "Required for reproducible ablations. Overwrites any seed in --config.")
    tr.set_defaults(func=_cmd_train)

    # predict -----------------------------------------------------------------
    pr = sub.add_parser(
        "predict",
        help="Score a single PE file. Prints `<path>\\t<score>` to stdout.",
    )
    pr.add_argument("model_dir", type=Path, help="Directory with model.txt + spec.json.")
    pr.add_argument("file", type=Path, help="PE file to score.")
    pr.set_defaults(func=_cmd_predict)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
