"""CLI for generating a submission CSV from a trained PXR model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize
from models import REGISTRY
from models.utils import drop_nan_rows


def _generate(
    train_path: Path,
    test_path: Path,
    model_name: str,
    input_names: list[str],
    target: str,
    output_path: Path,
    cache_dir: Path,
) -> pl.DataFrame:
    """Fit model on train data, predict on test data, write submission CSV."""
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(REGISTRY.keys())}")
    unknown = [n for n in input_names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")

    train_df = pl.read_csv(train_path)
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in train data. Columns: {list(train_df.columns)}")

    test_df = pl.read_csv(test_path)

    X_train, y_train = drop_nan_rows(featurize(train_df, input_names, cache_dir), train_df[target].to_numpy(), label="train")

    X_test_raw = featurize(test_df, input_names, cache_dir)
    valid_mask = ~np.isnan(X_test_raw).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} of {len(test_df)} test molecules with NaN features")
    X_test = X_test_raw[valid_mask]
    test_df_clean = test_df.filter(pl.Series(valid_mask.tolist()))

    model = REGISTRY[model_name]()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = test_df_clean.select(["Molecule Name", "SMILES"]).with_columns(pl.Series(target, preds.tolist()))

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
        nargs="+",
        default=["morgan"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
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
            args.input,  # now a list[str]
            args.target,
            output,
            Path(args.cache_dir),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
