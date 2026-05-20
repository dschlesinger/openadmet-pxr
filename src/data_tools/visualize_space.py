"""CLI for visualizing train/test molecular space using dimensionality reduction."""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from data_tools.inputs import INPUT_REGISTRY, featurize

_FIG_DPI = 150
sns.set_theme(style="whitegrid")


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


def _run_and_plot(X: np.ndarray, n_train: int, seed: int, input_name: str, out: Path) -> None:
    """Run PCA, UMAP, and t-SNE, then save a 3-panel figure."""
    methods = ["PCA", "UMAP", "t-SNE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Molecular space — {input_name}", fontsize=14)

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
        default="morgan",
        help=f"Representation to use. Available: {list(INPUT_REGISTRY.keys())}",
    )
    parser.add_argument("--train-path", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--test-path", default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (default: viz/space/<input>.png)")
    parser.add_argument("--cache-dir", default="data/features", help="Directory for cached feature matrices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP and t-SNE")
    args = parser.parse_args()

    if args.input not in INPUT_REGISTRY:
        print(f"Unknown input: {args.input!r}. Available: {list(INPUT_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    output_path: Path = args.output if args.output is not None else Path(f"viz/space/{args.input}.png")
    cache_dir = Path(args.cache_dir)

    train_df = pl.read_csv(args.train_path)
    test_df = pl.read_csv(args.test_path)
    print(f"Loaded {len(train_df)} train, {len(test_df)} test molecules", file=sys.stderr)

    X_train_raw = featurize(train_df, args.input, cache_dir)
    X_test_raw = featurize(test_df, args.input, cache_dir)

    X_train, _ = _drop_nan_rows(X_train_raw)
    X_test, _ = _drop_nan_rows(X_test_raw)
    print(f"After NaN drop: {X_train.shape[0]} train, {X_test.shape[0]} test", file=sys.stderr)

    X_combined = np.vstack([X_train, X_test])
    _run_and_plot(X_combined, X_train.shape[0], args.seed, args.input, output_path)
