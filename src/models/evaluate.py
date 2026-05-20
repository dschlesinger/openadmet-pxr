"""CLI for evaluating PXR challenge models on a train/validation split."""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.filters import FILTER_REGISTRY, apply_filter
from data_tools.inputs import INPUT_REGISTRY, featurize
from models import REGISTRY, PXRModel, PRECONFIG_REGISTRY, PXRPreConfigModel
from models.utils import drop_nan_rows


def _compute_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 from actual and predicted Series."""
    a = [float(v) for v in actual]
    p = [float(v) for v in predicted]
    n = len(a)
    mean_a = sum(a) / n
    ss_res = sum((ai - pi) ** 2 for ai, pi in zip(a, p))
    ss_tot = sum((ai - mean_a) ** 2 for ai in a)
    mae = sum(abs(ai - pi) for ai, pi in zip(a, p)) / n
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"RMSE": (ss_res / n) ** 0.5, "MAE": mae, "R2": r2}


def _evaluate_model(
    model: PXRModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Fit model on train, evaluate on val, return metrics."""
    model.fit(X_train, y_train)
    preds = pl.Series("prediction", model.predict(X_val).tolist())
    return _compute_metrics(pl.Series("actual", y_val.tolist()), preds)


def _resolve_models(names: list[str]) -> list[type[PXRModel]]:
    """Return model classes for the given names, or all registered models if names == ['all']."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(REGISTRY.keys())}")
    return [REGISTRY[n] for n in names]

def _resolve_preconfigs(names: list[str]) -> list[type[PXRPreConfigModel]]:
    """Return preconfigured model classes for the given names, or all registered preconfigured models if names == ['all']."""
    if names == ["all"]:
        return list(PRECONFIG_REGISTRY.values())
    unknown = [n for n in names if n not in PRECONFIG_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(PRECONFIG_REGISTRY.keys())}")
    return [PRECONFIG_REGISTRY[n] for n in names]


def _resolve_inputs(names: list[str]) -> None:
    """Raise ValueError if any name is not in INPUT_REGISTRY."""
    unknown = [n for n in names if n not in INPUT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}")


def _print_results(results: list[tuple[str, dict[str, float]]]) -> None:
    """Print a formatted metrics table to stdout."""
    w = max(len(name) for name, _ in results)
    print(f"{'Model':{w}}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}")
    print(f"{'-' * w}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for name, m in results:
        print(f"{name:{w}}  {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['R2']:>8.4f}")


def main() -> None:
    """Entry point for the evaluate-models CLI."""
    parser = argparse.ArgumentParser(description="Evaluate PXR models on a train/val split.")
    parser.add_argument("--train-path", default="data/train_split.csv", help="Path to training CSV")
    parser.add_argument("--val-path", type=Path, default=Path("data/val_split.csv"), help="Path to validation CSV")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fallback val fraction if --val-path not found")
    parser.add_argument("--seed", type=int, default=42, help="Fallback random seed if --val-path not found")
    parser.add_argument("--preconfigs", nargs="+", default=[], help="Names of preconfigured models, default is none")
    parser.add_argument("--models", nargs="+", default=[], help="Model names to evaluate, or omit for all")
    parser.add_argument("--target", default="pEC50", help="Target column to predict")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["morgan"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--sort-by", default="MAE", choices=["MAE", "RMSE", "R2"], help="Metric to sort results by")
    parser.add_argument(
        "--filter",
        default=None,
        choices=list(FILTER_REGISTRY.keys()),
        help="Training data filter to apply before fitting. Available: " + str(list(FILTER_REGISTRY.keys())),
    )
    parser.add_argument("--test-path", default="data/test.csv", help="Test CSV used when --filter is set")
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    # If both models and preconfigs empty then assume all models and preconfigs
    if not args.models and not args.preconfigs:
        args.models = ["all"]
        args.preconfigs = ["all"]

    try:
        model_classes = _resolve_models(args.models)
        preconfig_classes = _resolve_preconfigs(args.preconfigs)
        _resolve_inputs(args.input)
        for cls in preconfig_classes:
            _resolve_inputs(cls.required_repersentations)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)

    train_df = pl.read_csv(args.train_path)
    if args.target not in train_df.columns:
        print(f"Target '{args.target}' not found. Columns: {list(train_df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        test_df = pl.read_csv(args.test_path)
        train_df = apply_filter(train_df, test_df, args.filter, cache_dir)

    if args.val_path.exists():
        val_df = pl.read_csv(args.val_path)
        print(f"Using {args.val_path}: {len(val_df)} val / {len(train_df)} train molecules", file=sys.stderr)
    else:
        print(
            f"WARNING: {args.val_path} not found — falling back to random split "
            f"(val_split={args.val_split}, seed={args.seed}). Run download-data first.",
            file=sys.stderr,
        )
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_df))
        n_val = int(len(train_df) * args.val_split)
        val_df = train_df[idx[:n_val].tolist()]
        train_df = train_df[idx[n_val:].tolist()]

    X_train, y_train = drop_nan_rows(featurize(train_df, args.input, cache_dir), train_df[args.target].to_numpy())
    X_val, y_val = drop_nan_rows(featurize(val_df, args.input, cache_dir), val_df[args.target].to_numpy())

    results = [(cls.name, _evaluate_model(cls(), X_train, y_train, X_val, y_val)) for cls in model_classes]

    for cls in preconfig_classes:
        X_train_pc, y_train_pc = drop_nan_rows(
            featurize(train_df, cls.required_repersentations, cache_dir), train_df[args.target].to_numpy()
        )
        X_val_pc, y_val_pc = drop_nan_rows(
            featurize(val_df, cls.required_repersentations, cache_dir), val_df[args.target].to_numpy()
        )
        results.append((cls.name, _evaluate_model(cls(), X_train_pc, y_train_pc, X_val_pc, y_val_pc)))
    reverse = args.sort_by == "R2"
    results.sort(key=lambda r: r[1][args.sort_by], reverse=reverse)
    filter_info = f"  filter={args.filter}" if args.filter else ""
    print(f"input={' + '.join(args.input)}{filter_info}")
    _print_results(results)
