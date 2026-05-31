"""CLI for generating a submission CSV from a trained PXR model."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_tools.filters import FILTER_REGISTRY, apply_filter
from data_tools.inputs import INPUT_REGISTRY, featurize
from data_tools.load import _assemble_pool
from models import REGISTRY, PRECONFIG_REGISTRY
from models.utils import drop_nan_rows


def _generate(
    train_path: Path,
    test_path: Path,
    model_name: str,
    input_names: list[str],
    target: str,
    output_path: Path,
    cache_dir: Path,
    filter_fn=None,
    train_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fit model on train data, predict on test data, write submission CSV."""
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(REGISTRY.keys())}")
    unknown = [n for n in input_names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")

    if train_df is None:
        train_df = pl.read_csv(train_path)
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in train data. Columns: {list(train_df.columns)}")

    test_df = pl.read_csv(test_path)
    if filter_fn is not None:
        train_df = filter_fn(train_df, test_df)

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
    _print_prediction_summary(preds, target)
    _plot_prediction_distribution(preds, y_train, target, output_path)
    return result


def _generate_preconfig(
    train_path: Path,
    test_path: Path,
    model_name: str,
    target: str,
    output_path: Path,
    cache_dir: Path,
    filter_fn=None,
    train_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Fit a preconfigured model on train data, predict on test data, write submission CSV."""
    if model_name not in PRECONFIG_REGISTRY:
        raise ValueError(f"Unknown preconfig model: {model_name!r}. Available: {list(PRECONFIG_REGISTRY.keys())}")

    cls = PRECONFIG_REGISTRY[model_name]
    input_names = cls.required_repersentations

    if train_df is None:
        train_df = pl.read_csv(train_path)
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in train data. Columns: {list(train_df.columns)}")

    test_df = pl.read_csv(test_path)
    if filter_fn is not None:
        train_df = filter_fn(train_df, test_df)

    X_train, y_train = drop_nan_rows(featurize(train_df, input_names, cache_dir), train_df[target].to_numpy(), label="train")

    X_test_raw = featurize(test_df, input_names, cache_dir)
    valid_mask = ~np.isnan(X_test_raw).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"Dropped {n_dropped} of {len(test_df)} test molecules with NaN features")
    X_test = X_test_raw[valid_mask]
    test_df_clean = test_df.filter(pl.Series(valid_mask.tolist()))

    model = cls()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result = test_df_clean.select(["Molecule Name", "SMILES"]).with_columns(pl.Series(target, preds.tolist()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    print(f"Wrote {len(result)} rows -> {output_path}")
    _print_prediction_summary(preds, target)
    _plot_prediction_distribution(preds, y_train, target, output_path)
    return result


def _print_prediction_summary(preds: np.ndarray, target: str) -> None:
    """Print distribution statistics for the generated predictions."""
    p = preds.astype(float)
    percentiles = np.percentile(p, [5, 25, 50, 75, 95])
    print(f"\nPrediction summary ({target}, n={len(p)}):")
    print(f"  mean   : {p.mean():.4f}")
    print(f"  std    : {p.std():.4f}")
    print(f"  min    : {p.min():.4f}")
    print(f"  p5     : {percentiles[0]:.4f}")
    print(f"  p25    : {percentiles[1]:.4f}")
    print(f"  median : {percentiles[2]:.4f}")
    print(f"  p75    : {percentiles[3]:.4f}")
    print(f"  p95    : {percentiles[4]:.4f}")
    print(f"  max    : {p.max():.4f}")


def _plot_prediction_distribution(
    preds: np.ndarray, y_train: np.ndarray, target: str, output_path: Path
) -> None:
    """Save an overlaid density histogram comparing predictions to the training distribution."""
    p = preds.astype(float)
    t = y_train.astype(float)
    all_vals = np.concatenate([p, t])
    bins = np.linspace(all_vals.min(), all_vals.max(), 31)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t, bins=bins, density=True, alpha=0.5, label=f"train (n={len(t)}, mean={t.mean():.2f}, std={t.std():.2f})")
    ax.hist(p, bins=bins, density=True, alpha=0.5, label=f"predicted (n={len(p)}, mean={p.mean():.2f}, std={p.std():.2f})")
    ax.axvline(t.mean(), color="C0", linestyle="--", linewidth=1.2)
    ax.axvline(p.mean(), color="C1", linestyle="--", linewidth=1.2)
    ax.set_xlabel(target)
    ax.set_ylabel("Density")
    ax.set_title(f"{target}: predicted vs training distribution")
    ax.legend()
    fig.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot -> {plot_path}")


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
        "--preconfig",
        default=None,
        help=f"Preconfigured model name (uses its own representations). Available: {list(PRECONFIG_REGISTRY.keys())}",
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
    parser.add_argument(
        "--filter",
        default=None,
        choices=list(FILTER_REGISTRY.keys()),
        help="Training data filter to apply before fitting. Available: " + str(list(FILTER_REGISTRY.keys())),
    )
    parser.add_argument(
        "--include-unblinded",
        action="store_true",
        help=(
            "Add the 253 phase-1 unblinded test molecules to the training pool. "
            "The submission CSV still contains all 513 test predictions. "
            "Use load_test_holdout() for unbiased local evaluation on the remaining 260 molecules."
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    filter_fn = (lambda tr, te: apply_filter(tr, te, args.filter, cache_dir)) if args.filter else None

    train_override: pl.DataFrame | None = None
    if args.include_unblinded:
        train_override = _assemble_pool(include_unblinded=True, data_dir=Path(args.train_path).parent)
        unblinded_smiles = set(
            pl.read_csv(Path(args.train_path).parent / "test_unblinded.csv")["SMILES"].to_list()
        )
        test_smiles = set(pl.read_csv(args.test_path)["SMILES"].to_list())
        n_contaminated = len(unblinded_smiles & test_smiles)
        print(
            f"--include-unblinded: training on {len(train_override)} molecules "
            f"({n_contaminated} of {len(test_smiles)} test molecules were used in training — "
            f"use load_test_holdout() for unbiased local evaluation)",
            file=sys.stderr,
        )

    filter_tag = f"_{args.filter}" if args.filter else ""

    try:
        if args.preconfig is not None:
            default_name = f"{args.preconfig}{filter_tag}_submission.csv"
            output = Path(args.output) if args.output else Path("results") / default_name
            _generate_preconfig(
                Path(args.train_path),
                Path(args.test_path),
                args.preconfig,
                args.target,
                output,
                cache_dir,
                filter_fn,
                train_df=train_override,
            )
        else:
            input_tag = "+".join(args.input)
            default_name = f"{args.model}_{input_tag}{filter_tag}_submission.csv"
            output = Path(args.output) if args.output else Path("results") / default_name
            _generate(
                Path(args.train_path),
                Path(args.test_path),
                args.model,
                args.input,
                args.target,
                output,
                cache_dir,
                filter_fn,
                train_df=train_override,
            )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
