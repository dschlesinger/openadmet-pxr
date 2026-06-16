"""CLI for Optuna hyperparameter tuning of PXR challenge models."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import optuna
import polars as pl
from optuna.trial import FixedTrial

from data_tools.filters import FILTER_REGISTRY, apply_filter
from data_tools.inputs import INPUT_REGISTRY, featurize
from data_tools.load import load_data
from models import REGISTRY
from models.evaluate import _evaluate_model
from models.optimize import SEARCH_SPACES
from models.utils import drop_nan_rows

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_callback(metric: str) -> Callable:
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number:4d}  {metric}={trial.value:.4f}  best={study.best_value:.4f}", flush=True)

    return _cb


def main() -> None:
    """Entry point for the tune-model CLI."""
    parser = argparse.ArgumentParser(description="Tune a PXR model with Optuna TPE search.")
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()), help="Model to tune")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100)")
    parser.add_argument("--timeout", type=float, default=None, help="Stop after N seconds regardless of trial count")
    parser.add_argument("--metric", default="RMSE", choices=["RMSE", "MAE", "R2"], help="Metric to optimize")
    parser.add_argument("--output", type=Path, default=None, help="Write best params + metrics to JSON")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing data CSVs")
    parser.add_argument("--val-split", type=float, default=0.2, help="Val fraction for scaffold/butina split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler and data split")
    parser.add_argument("--target", default="pEC50", help="Target column")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["morgan"],
        help=f"Input featurization(s) to use (hstacked if multiple). Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--cache-dir", default="data/features", help="Feature cache directory")
    parser.add_argument(
        "--scaffold-split",
        action="store_true",
        help="Use scaffold-based train/val split (no scaffold leakage).",
    )
    parser.add_argument(
        "--butina-split",
        action="store_true",
        help="Use Butina cluster split on Morgan fps (no similar molecules span both splits).",
    )
    parser.add_argument(
        "--butina-cutoff",
        type=float,
        default=0.4,
        help="Tanimoto distance cutoff for Butina clustering (default: 0.4).",
    )
    parser.add_argument(
        "--include-unblinded",
        action="store_true",
        help="Add phase-1 unblinded test molecules to the training pool.",
    )
    parser.add_argument(
        "--include-counter-assay",
        action="store_true",
        help="Add pEC50_counter column from counter-assay data.",
    )
    parser.add_argument(
        "--include-single-conc",
        action="store_true",
        help="Add log2_fc_single column from single-concentration screen.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        choices=list(FILTER_REGISTRY.keys()),
        help="Training data filter to apply before fitting. Available: " + str(list(FILTER_REGISTRY.keys())),
    )
    parser.add_argument("--test-path", default="data/test.csv", help="Test CSV used when --filter is set")
    args = parser.parse_args()

    if (args.scaffold_split or args.butina_split) and not 0.0 < args.val_split < 1.0:
        print(f"--val-split must be in (0, 1), got {args.val_split}", file=sys.stderr)
        sys.exit(1)

    unknown_inputs = [n for n in args.input if n not in INPUT_REGISTRY]
    if unknown_inputs:
        print(f"Unknown input(s): {unknown_inputs}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    model_cls = REGISTRY[args.model]
    cache_dir = Path(args.cache_dir)

    if args.scaffold_split:
        split_type = "scaffold"
    elif args.butina_split:
        split_type = "butina"
    else:
        split_type = "unblinded"
    train_df, val_df = load_data(
        include_unblinded=args.include_unblinded,
        include_counter_assay=args.include_counter_assay,
        include_single_concentration=args.include_single_conc,
        split_type=split_type,
        val_fraction=args.val_split,
        seed=args.seed,
        butina_cutoff=args.butina_cutoff,
        data_dir=args.data_dir,
    )
    print(
        f"load_data(split_type={split_type!r}): {len(val_df)} val / {len(train_df)} train molecules",
        file=sys.stderr,
    )

    if args.target not in train_df.columns:
        print(f"Target '{args.target}' not found. Columns: {list(train_df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.filter:
        test_df = pl.read_csv(args.test_path)
        train_df = apply_filter(train_df, test_df, args.filter, cache_dir)

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
