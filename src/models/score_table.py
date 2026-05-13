"""CLI to generate an input-by-model score table for the PXR challenge."""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize
from models import REGISTRY
from models.utils import drop_nan_rows

METRICS = ("RMSE", "MAE", "R2")


def _compute_score(actual: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    n = len(actual)
    res = actual - predicted
    ss_res = float((res**2).sum())
    if metric == "RMSE":
        return (ss_res / n) ** 0.5
    if metric == "MAE":
        return float(np.abs(res).sum()) / n
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def _load_split(
    train_path: Path,
    val_path: Path,
    val_split: float,
    seed: int,
    target: str,
    input_names: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pl.read_csv(train_path)
    if target not in train_df.columns:
        print(f"Target '{target}' not found. Columns: {list(train_df.columns)}", file=sys.stderr)
        sys.exit(1)

    X_train_raw = featurize(train_df, input_names, cache_dir)
    y_train_raw = train_df[target].to_numpy()
    X_train, y_train = drop_nan_rows(X_train_raw, y_train_raw)

    if val_path.exists():
        val_df = pl.read_csv(val_path)
        X_val_raw = featurize(val_df, input_names, cache_dir)
        y_val_raw = val_df[target].to_numpy()
        X_val, y_val = drop_nan_rows(X_val_raw, y_val_raw)
        print(f"Using {val_path}: {len(y_val)} val / {len(y_train)} train molecules", file=sys.stderr)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_train))
        n_val = int(len(X_train) * val_split)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, y_train, X_val, y_val = (
            X_train[train_idx], y_train[train_idx],
            X_train[val_idx], y_train[val_idx],
        )
    return X_train, y_train, X_val, y_val


def _print_table(
    scores: dict[str, dict[str, float]],
    inputs: list[str],
    models: list[str],
    metric: str,
) -> None:
    col_w = max(len(m) for m in models)
    row_w = max(len(inp) for inp in inputs)
    header = f"{'Input':{row_w}}  " + "  ".join(f"{m:{col_w}}" for m in models)
    print(header)
    print("-" * len(header))
    for inp in inputs:
        row = f"{inp:{row_w}}  "
        row += "  ".join(f"{scores[inp][m]:>{col_w}.4f}" for m in models)
        print(row)
    print(f"\n(score: {metric})")


def main() -> None:
    """Entry point for the score-table CLI."""
    parser = argparse.ArgumentParser(
        description="Generate an input-by-model score table for the PXR challenge."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument("--val-path", type=Path, default=Path("data/val_split.csv"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="pEC50")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(INPUT_REGISTRY.keys()),
        help=f"Input featurizations. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(REGISTRY.keys()),
        help=f"Models to evaluate. Available: {list(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--score",
        default="MAE",
        choices=list(METRICS),
        help="Metric to display in the table (default: MAE)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features"))
    args = parser.parse_args()

    unknown_inputs = [n for n in args.inputs if n not in INPUT_REGISTRY]
    unknown_models = [n for n in args.models if n not in REGISTRY]
    if unknown_inputs or unknown_models:
        if unknown_inputs:
            print(f"Unknown input(s): {unknown_inputs}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        if unknown_models:
            print(f"Unknown model(s): {unknown_models}. Available: {list(REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    model_classes = [REGISTRY[m] for m in args.models]
    scores: dict[str, dict[str, float]] = {}

    for inp in args.inputs:
        print(f"\n=== Input: {inp} ===", file=sys.stderr)
        X_train, y_train, X_val, y_val = _load_split(
            args.train_path, args.val_path, args.val_split,
            args.seed, args.target, [inp], args.cache_dir,
        )
        scores[inp] = {}
        for cls in model_classes:
            model = cls()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores[inp][cls.name] = _compute_score(y_val, preds, args.score)

    _print_table(scores, args.inputs, args.models, args.score)
