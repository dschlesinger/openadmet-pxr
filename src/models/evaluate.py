"""CLI for evaluating PXR challenge models on a train/validation split."""

import argparse
import sys

import polars as pl

from models import REGISTRY, PXRModel


def _split_data(df: pl.DataFrame, val_fraction: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, val) after a reproducible shuffle-split."""
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


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


def _evaluate_model(model: PXRModel, train: pl.DataFrame, val: pl.DataFrame, target: str) -> dict[str, float]:
    """Fit model on train, evaluate on val, return metrics."""
    model.fit(train, target)
    return _compute_metrics(val[target], model.predict(val))


def _resolve_models(names: list[str]) -> list[type[PXRModel]]:
    """Return model classes for the given names, or all registered models if names == ['all']."""
    if names == ["all"]:
        return list(REGISTRY.values())
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {list(REGISTRY.keys())}")
    return [REGISTRY[n] for n in names]


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
    parser.add_argument("--train-path", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of training data held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the shuffle-split")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names to evaluate, or omit for all")
    parser.add_argument("--target", default="pEC50", help="Target column to predict")
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    try:
        model_classes = _resolve_models(args.models)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    df = pl.read_csv(args.train_path)
    if args.target not in df.columns:
        print(f"Target '{args.target}' not found. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    train, val = _split_data(df, args.val_split, args.seed)
    results = [(cls.name, _evaluate_model(cls(), train, val, args.target)) for cls in model_classes]
    _print_results(results)
