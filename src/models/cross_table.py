"""CLI to generate a representation × representation cross-score table."""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

from data_tools.inputs import INPUT_REGISTRY, featurize
from data_tools.load import load_data
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


def _load_data_splits(
    val_split: float,
    seed: int,
    target: str,
    cache_dir: Path,
    scaffold_split: bool = False,
    butina_split: bool = False,
    butina_cutoff: float = 0.4,
    include_unblinded: bool = False,
    include_counter_assay: bool = False,
    include_single_conc: bool = False,
    data_dir: Path = Path("data"),
) -> tuple[np.ndarray, np.ndarray, object, object]:
    if scaffold_split:
        split_type = "scaffold"
    elif butina_split:
        split_type = "butina"
    else:
        split_type = "unblinded"
    train_df, val_df = load_data(
        include_unblinded=include_unblinded,
        include_counter_assay=include_counter_assay,
        include_single_concentration=include_single_conc,
        split_type=split_type,
        val_fraction=val_split,
        seed=seed,
        butina_cutoff=butina_cutoff,
        data_dir=data_dir,
    )
    print(f"load_data(split_type={split_type!r}): {len(val_df)} val / {len(train_df)} train molecules", file=sys.stderr)

    if target not in train_df.columns:
        print(f"Target '{target}' not found. Columns: {list(train_df.columns)}", file=sys.stderr)
        sys.exit(1)

    y_train = train_df[target].to_numpy()
    y_val = val_df[target].to_numpy()
    return y_train, y_val, train_df, val_df


def _print_matrix(
    scores: dict[tuple[str, str], float],
    reps: list[str],
    model_name: str,
    metric: str,
) -> None:
    cell_w = 8
    row_w = max(len(r) for r in reps)
    header_pad = " " * (row_w + 2)
    col_header = "  ".join(f"{r:>{cell_w}}" for r in reps)
    header = f"{header_pad}{col_header}"
    print(header)
    print("-" * len(header))
    for rep_i in reps:
        row = f"{rep_i:{row_w}}  "
        row += "  ".join(f"{scores[(rep_i, rep_j)]:>{cell_w}.4f}" for rep_j in reps)
        print(row)
    print(f"\n(model: {model_name}, score: {metric})")


def _write_csv(
    scores: dict[tuple[str, str], float],
    reps: list[str],
    path: Path,
) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rep_row \\ rep_col"] + reps)
        for rep_i in reps:
            writer.writerow([rep_i] + [f"{scores[(rep_i, rep_j)]:.6f}" for rep_j in reps])


def main() -> None:
    """Entry point for the cross-table CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a representation×representation cross-score matrix."
    )
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="pEC50")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=list(REGISTRY.keys()),
        help=f"Model to use. Default: xgboost. Available: {list(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(INPUT_REGISTRY.keys()),
        help=f"Representations to cross. Default: all. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--score",
        default="R2",
        choices=list(METRICS),
        help="Metric to display (default: R2)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--output-csv", type=Path, default=None, help="Write matrix to this CSV path")
    parser.add_argument("--scaffold-split", action="store_true", help="Use scaffold-based train/val split")
    parser.add_argument("--butina-split", action="store_true", help="Use Butina cluster split on Morgan fps")
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Tanimoto distance cutoff for Butina clustering")
    parser.add_argument("--include-unblinded", action="store_true", help="Add phase-1 unblinded molecules to training pool")
    parser.add_argument("--include-counter-assay", action="store_true", help="Add counter-assay data to training pool")
    parser.add_argument("--include-single-conc", action="store_true", help="Add single-concentration screen data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory containing data CSVs")
    args = parser.parse_args()

    unknown = [n for n in args.inputs if n not in INPUT_REGISTRY]
    if unknown:
        print(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    model_cls = REGISTRY[args.model]
    reps = args.inputs

    y_train, y_val, train_df, val_df = _load_data_splits(
        args.val_split, args.seed, args.target, args.cache_dir,
        scaffold_split=args.scaffold_split,
        butina_split=args.butina_split,
        butina_cutoff=args.butina_cutoff,
        include_unblinded=args.include_unblinded,
        include_counter_assay=args.include_counter_assay,
        include_single_conc=args.include_single_conc,
        data_dir=args.data_dir,
    )

    print(f"Pre-loading {len(reps)} representations...", file=sys.stderr)
    train_feats: dict[str, np.ndarray] = {r: featurize(train_df, r, args.cache_dir) for r in reps}
    val_feats: dict[str, np.ndarray] = {r: featurize(val_df, r, args.cache_dir) for r in reps}

    pairs = list(itertools.product(reps, repeat=2))
    scores: dict[tuple[str, str], float] = {}

    for idx, (rep_i, rep_j) in enumerate(pairs):
        label = rep_i if rep_i == rep_j else f"{rep_i} + {rep_j}"
        print(f"[{idx + 1}/{len(pairs)}] {label}", file=sys.stderr)

        if rep_i == rep_j:
            X_train_raw = train_feats[rep_i]
            X_val_raw = val_feats[rep_i]
        else:
            X_train_raw = np.hstack([train_feats[rep_i], train_feats[rep_j]])
            X_val_raw = np.hstack([val_feats[rep_i], val_feats[rep_j]])

        X_train, y_train_c = drop_nan_rows(X_train_raw, y_train)
        X_val, y_val_c = drop_nan_rows(X_val_raw, y_val)

        model = model_cls()
        model.fit(X_train, y_train_c)
        preds = model.predict(X_val)
        scores[(rep_i, rep_j)] = _compute_score(y_val_c, preds, args.score)

    _print_matrix(scores, reps, args.model, args.score)

    if args.output_csv is not None:
        _write_csv(scores, reps, args.output_csv)
        print(f"\nMatrix written to {args.output_csv}", file=sys.stderr)
