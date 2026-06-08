"""CLI for Optuna hyperparameter tuning of PXR challenge models."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import optuna
import polars as pl
from optuna.trial import FixedTrial

from data_tools.inputs import INPUT_REGISTRY, featurize
from models import REGISTRY
from models.evaluate import _evaluate_model
from models.optimize import SEARCH_SPACES
from models.utils import drop_nan_rows

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_callback(metric: str) -> Callable:
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number:4d}  {metric}={trial.value:.4f}  best={study.best_value:.4f}", flush=True)

    return _cb


def _build_split(
    train_df: pl.DataFrame, args: argparse.Namespace
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train_df, val_df), falling back to a random split if val CSV is missing."""
    if args.val_path.exists():
        return train_df, pl.read_csv(args.val_path)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_df))
    n_val = int(len(train_df) * args.val_split)
    return train_df[idx[n_val:].tolist()], train_df[idx[:n_val].tolist()]


def main() -> None:
    """Entry point for the tune-model CLI."""
    parser = argparse.ArgumentParser(description="Tune a PXR model with Optuna TPE search.")
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()), help="Model to tune")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100)")
    parser.add_argument("--timeout", type=float, default=None, help="Stop after N seconds regardless of trial count")
    parser.add_argument("--metric", default="RMSE", choices=["RMSE", "MAE", "R2"], help="Metric to optimize")
    parser.add_argument("--output", type=Path, default=None, help="Write best params + metrics to JSON")
    parser.add_argument("--train-path", default="data/train_split.csv", help="Training CSV path")
    parser.add_argument("--val-path", type=Path, default=Path("data/val_split.csv"), help="Validation CSV path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fallback val fraction if --val-path missing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler and fallback split")
    parser.add_argument("--target", default="pEC50", help="Target column")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["morgan"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Feature cache directory")
    args = parser.parse_args()

    unknown_inputs = [n for n in args.input if n not in INPUT_REGISTRY]
    if unknown_inputs:
        print(f"Unknown input(s): {unknown_inputs}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    model_cls = REGISTRY[args.model]
    cache_dir = Path(args.cache_dir)
    raw_train = pl.read_csv(args.train_path)
    train_df, val_df = _build_split(raw_train, args)

    X_train, y_train = drop_nan_rows(featurize(train_df, args.input, cache_dir), train_df[args.target].to_numpy())
    X_val, y_val = drop_nan_rows(featurize(val_df, args.input, cache_dir), val_df[args.target].to_numpy())

    direction = "maximize" if args.metric == "R2" else "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=args.seed))

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = SEARCH_SPACES[args.model](trial)
        metrics = _evaluate_model(model_cls(**params), X_train, y_train, X_val, y_val)
        return float(metrics[args.metric])

    print(
        f"Tuning {args.model}  trials={args.n_trials}  metric={args.metric}"
        f"  input={'+'.join(args.input)}  n_train={len(X_train)}  n_val={len(X_val)}"
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[_make_callback(args.metric)])

    best_params = study.best_params
    best_value = study.best_value

    best_model_params: Dict[str, Any] = SEARCH_SPACES[args.model](FixedTrial(best_params))
    final_metrics = _evaluate_model(model_cls(**best_model_params), X_train, y_train, X_val, y_val)

    print(f"\nBest {args.metric}: {best_value:.4f}  (after {len(study.trials)} trials)")
    print("Best params:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")
    print(
        f"\nFinal validation — RMSE={final_metrics['RMSE']:.4f}"
        f"  MAE={final_metrics['MAE']:.4f}  R2={final_metrics['R2']:.4f}"
    )

    if args.output:
        result: Dict[str, Any] = {
            "model": args.model,
            "metric": args.metric,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "params": best_params,
            "final_metrics": final_metrics,
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output}")
