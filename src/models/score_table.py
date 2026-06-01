"""CLI to generate an input-by-model score table for the PXR challenge."""

import argparse
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


def _load_split(  # pylint: disable=too-many-arguments
    val_split: float,
    seed: int,
    target: str,
    input_names: list[str],
    cache_dir: Path,
    scaffold_split: bool = False,
    butina_split: bool = False,
    butina_cutoff: float = 0.4,
    include_unblinded: bool = False,
    include_counter_assay: bool = False,
    include_single_conc: bool = False,
    data_dir: Path = Path("data"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    X_train, y_train = drop_nan_rows(featurize(train_df, input_names, cache_dir), train_df[target].to_numpy())
    X_val, y_val = drop_nan_rows(featurize(val_df, input_names, cache_dir), val_df[target].to_numpy())
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
    parser.add_argument("--val-split", type=float, default=0.1)
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
    parser.add_argument("--scaffold-split", action="store_true", help="Use scaffold-based train/val split")
    parser.add_argument("--butina-split", action="store_true", help="Use Butina cluster split on Morgan fps")
    parser.add_argument("--butina-cutoff", type=float, default=0.4, help="Tanimoto distance cutoff for Butina clustering (default: 0.4)")
    parser.add_argument("--include-unblinded", action="store_true", help="Add phase-1 unblinded molecules to training pool")
    parser.add_argument("--include-counter-assay", action="store_true", help="Add pEC50_counter column from counter-assay data")
    parser.add_argument("--include-single-conc", action="store_true", help="Add log2_fc_single column from single-concentration screen")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory containing data CSVs")
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
            args.val_split, args.seed, args.target, [inp], args.cache_dir,
            scaffold_split=args.scaffold_split,
            butina_split=args.butina_split,
            butina_cutoff=args.butina_cutoff,
            include_unblinded=args.include_unblinded,
            include_counter_assay=args.include_counter_assay,
            include_single_conc=args.include_single_conc,
            data_dir=args.data_dir,
        )
        scores[inp] = {}
        for cls in model_classes:
            model = cls()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores[inp][cls.name] = _compute_score(y_val, preds, args.score)

    _print_table(scores, args.inputs, args.models, args.score)
