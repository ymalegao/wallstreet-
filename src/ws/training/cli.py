"""Command-line workflows for point-in-time training and one-time evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ws.training.final_eval import final_eval
from ws.training.final_fit import fit_final
from ws.training.table import build_training_table, write_data_coverage
from ws.training.trainer import run_walk_forward

_ROOT = Path(__file__).parents[3]
_UNIVERSE = _ROOT / "configs" / "universe" / "sp100-current.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ws.training.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-table", help="build a pre-holdout px-v1 or pxf-v1 table")
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--universe", type=Path, default=_UNIVERSE)
    build.add_argument("--feature-set", choices=("px-v1", "pxf-v1"), default="px-v1")
    build.add_argument("--facts", type=Path)
    build.add_argument("--out", type=Path, required=True)

    walk = commands.add_parser(
        "walk-forward", help="train the fixed LightGBM grid and run 50 train-only placebo permutations"
    )
    walk.add_argument("--table", type=Path, required=True)
    walk.add_argument("--out", type=Path, default=_ROOT / "runs" / "training")
    walk.add_argument("--num-threads", type=int, default=4)

    fit = commands.add_parser("fit-final", help="fit the pinned five-seed pre-holdout model")
    fit.add_argument("--table", type=Path, required=True)
    fit.add_argument("--num-threads", type=int, default=4)
    fit.add_argument("--iteration-policy", choices=("median-fold", "fresh-validation"), default="median-fold")

    evaluate = commands.add_parser("final-eval", help="run the one-time locked holdout evaluation")
    evaluate.add_argument("--model", required=True)

    args = parser.parse_args(argv)
    if args.command == "build-table":
        result = build_training_table(
            args.store,
            args.universe,
            args.out,
            feature_set=args.feature_set,
            facts_path=args.facts,
        )
        manifest = json.loads((args.store / "snapshot.json").read_text(encoding="utf-8"))
        write_data_coverage(manifest, _ROOT / "docs" / "data-coverage.md")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "walk-forward":
        result = run_walk_forward(
            args.table,
            args.out,
            num_threads=args.num_threads,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2, sort_keys=True))
        return 0
    if args.command == "fit-final":
        result = fit_final(args.table, iteration_policy=args.iteration_policy, num_threads=args.num_threads)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = final_eval(args.model)
    print(json.dumps({"model_id": result["model_id"], "holdout_range": result["holdout_range"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
