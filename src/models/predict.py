"""CLI for generating a submission CSV from a trained PXR model."""

import argparse
import sys
from pathlib import Path

import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize
from models import REGISTRY


def _generate(
    train_path: Path,
    test_path: Path,
    model_name: str,
    input_name: str,
    target: str,
    output_path: Path,
    cache_dir: Path,
) -> pl.DataFrame:
    """Fit model on train data, predict on test data, write submission CSV."""
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(REGISTRY.keys())}")
    if input_name not in INPUT_REGISTRY:
        raise ValueError(f"Unknown input: {input_name!r}. Available: {list(INPUT_REGISTRY.keys())}")

    train_df = pl.read_csv(train_path)
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in train data. Columns: {list(train_df.columns)}")

    test_df = pl.read_csv(test_path)

    model = REGISTRY[model_name]()
    model.fit(featurize(train_df, input_name, cache_dir), train_df[target].to_numpy())
    preds = model.predict(featurize(test_df, input_name, cache_dir))

    result = test_df.select(["Molecule Name", "SMILES"]).with_columns(pl.Series(target, preds.tolist()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    print(f"Wrote {len(result)} rows -> {output_path}")
    return result


def main() -> None:
    """Entry point for the generate-results CLI."""
    parser = argparse.ArgumentParser(description="Generate a submission CSV from a trained PXR model.")
    parser.add_argument("--train-path", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test-path", default="data/test.csv", help="Path to test CSV")
    parser.add_argument(
        "--model",
        default="mean_baseline",
        help=f"Model name to use. Available: {list(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--input",
        default="morgan",
        help=f"Input featurization to use. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--target", default="pEC50", help="Target column in training data")
    parser.add_argument("--output", default=None, help="Output CSV path (default: results/<model>.csv)")
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path("results") / f"{args.model}_submission.csv"

    try:
        _generate(
            Path(args.train_path),
            Path(args.test_path),
            args.model,
            args.input,
            args.target,
            output,
            Path(args.cache_dir),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
