"""CLI for visualizing train/test molecular space using dimensionality reduction."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from data_tools.filters import FILTER_REGISTRY, apply_filter
from data_tools.inputs import INPUT_REGISTRY, featurize

_FIG_DPI = 150
sns.set_theme(style="whitegrid")


def _adversarial_auc(X_train: np.ndarray, X_test: np.ndarray, seed: int = 42) -> float:
    """Train an XGBoost classifier to distinguish train (0) from test (1) examples.

    A high AUC indicates covariate shift — train and test occupy different
    chemical space regions. See: https://github.com/jeremycheminf/openadmet_scripts/blob/main/PXR/2D/TRAIN_TEST_DISTANCE.md
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: PLC0415
    from xgboost import XGBClassifier  # noqa: PLC0415

    X = np.vstack([X_train, X_test])
    y = np.array([0] * len(X_train) + [1] * len(X_test))

    # Replace any remaining NaNs with 0 (XGBoost can handle NaN natively but
    # sklearn CV wrapper may not always propagate the flag correctly)
    X = np.nan_to_num(X, nan=0.0)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def _drop_nan_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_clean, valid_mask) dropping rows with any NaN."""
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], valid_mask


def _preprocess(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA-reduce to n_components when dimensionality exceeds that threshold."""
    if X.shape[1] <= n_components:
        return X
    from sklearn.decomposition import PCA  # noqa: PLC0415

    return PCA(n_components=n_components, random_state=0).fit_transform(X)


def _reduce(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Apply dimensionality reduction and return 2D coordinates."""
    if method == "PCA":
        from sklearn.decomposition import PCA  # noqa: PLC0415

        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "UMAP":
        import umap  # noqa: PLC0415

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(_preprocess(X))
    if method == "t-SNE":
        from sklearn.manifold import TSNE  # noqa: PLC0415

        return TSNE(n_components=2, random_state=seed).fit_transform(_preprocess(X))
    raise ValueError(f"Unknown method: {method!r}")


def _plot_panel(ax: plt.Axes, coords: np.ndarray, n_train: int, title: str) -> None:
    """Scatter train (circles) and test (triangles) on ax."""
    ax.scatter(coords[:n_train, 0], coords[:n_train, 1], s=8, alpha=0.5, label="train")
    ax.scatter(coords[n_train:, 0], coords[n_train:, 1], s=8, alpha=0.6, label="test", marker="^")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2, fontsize=8)


def _run_and_plot(X_train: np.ndarray, X_test: np.ndarray, seed: int, input_name: str, out: Path) -> None:
    """Run adversarial AUC, PCA, UMAP, and t-SNE, then save a 3-panel figure."""
    print("Computing adversarial XGBoost AUC...", file=sys.stderr)
    auc = _adversarial_auc(X_train, X_test, seed)
    print(f"Adversarial AUC (train vs test): {auc:.3f}", file=sys.stderr)

    X = np.vstack([X_train, X_test])
    n_train = X_train.shape[0]

    methods = ["PCA", "UMAP", "t-SNE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Molecular space — {input_name}\nAdversarial XGBoost AUC (train vs test): {auc:.3f}",
        fontsize=13,
    )

    for ax, method in zip(axes, methods):
        print(f"Running {method}...", file=sys.stderr)
        coords = _reduce(X, method, seed)
        _plot_panel(ax, coords, n_train, method)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main() -> None:
    """Entry point for the visualize-space CLI."""
    parser = argparse.ArgumentParser(description="Visualize train/test molecular space.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["morgan"],
        help=f"Representation(s) to use, or 'all'. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test-path", default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (single input only)")
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP and t-SNE")
    parser.add_argument(
        "--filter",
        default=None,
        choices=list(FILTER_REGISTRY.keys()),
        help="Training data filter to apply before visualization. Available: " + str(list(FILTER_REGISTRY.keys())),
    )
    args = parser.parse_args()

    inputs = list(INPUT_REGISTRY.keys()) if args.input == ["all"] else args.input
    unknown = [n for n in inputs if n not in INPUT_REGISTRY]
    if unknown:
        print(f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    filter_tag = f"_{args.filter}" if args.filter else ""

    train_df = pl.read_csv(args.train_path)
    test_df = pl.read_csv(args.test_path)
    print(f"Loaded {len(train_df)} train, {len(test_df)} test molecules", file=sys.stderr)

    if args.filter:
        train_df = apply_filter(train_df, test_df, args.filter, cache_dir)

    for input_name in inputs:
        print(f"\n--- {input_name} ---", file=sys.stderr)
        X_train_raw = featurize(train_df, input_name, cache_dir)
        X_test_raw = featurize(test_df, input_name, cache_dir)

        X_train, _ = _drop_nan_rows(X_train_raw)
        X_test, _ = _drop_nan_rows(X_test_raw)
        print(f"After NaN drop: {X_train.shape[0]} train, {X_test.shape[0]} test", file=sys.stderr)

        if args.output is not None and len(inputs) == 1:
            out = args.output
        else:
            out = Path(f"viz/space/{input_name}{filter_tag}.png")

        display_name = input_name if not args.filter else f"{input_name} | filter: {args.filter}"
        _run_and_plot(X_train, X_test, args.seed, display_name, out)
