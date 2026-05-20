"""Training-data filters for the PXR challenge."""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from data_tools.inputs import featurize

FilterFn = Callable[[pl.DataFrame, pl.DataFrame, Path], pl.DataFrame]


def _bulk_tanimoto(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Return per-train-mol max Tanimoto similarity to any test molecule.

    Expects binary (0/1) fingerprint arrays. Result shape: (n_train,).
    """
    intersection = X_train @ X_test.T
    a_bits = X_train.sum(axis=1, keepdims=True)
    b_bits = X_test.sum(axis=1)
    union = a_bits + b_bits - intersection
    tanimoto = np.where(union > 0, intersection / union, 0.0)
    return tanimoto.max(axis=1)


def tanimoto_top30(train_df: pl.DataFrame, test_df: pl.DataFrame, cache_dir: Path) -> pl.DataFrame:
    """Keep the 30% of training molecules most similar to the test set by Tanimoto.

    Uses cached Morgan fingerprints (loaded via featurize) to avoid recomputation.
    """
    X_train = featurize(train_df, "morgan", cache_dir)
    X_test = featurize(test_df, "morgan", cache_dir)

    train_valid = np.isfinite(X_train).all(axis=1)
    test_valid = np.isfinite(X_test).all(axis=1)

    scores = np.zeros(len(train_df), dtype=np.float64)
    if train_valid.any() and test_valid.any():
        sims = _bulk_tanimoto(X_train[train_valid], X_test[test_valid])
        scores[train_valid] = sims

    n_keep = max(1, int(len(train_df) * 0.30))
    ranked = np.argsort(scores)[::-1]
    keep_idx = sorted(ranked[:n_keep].tolist())
    print(f"[tanimoto_top30] Kept {n_keep} of {len(train_df)} train molecules (top 30% by Tanimoto to test)",
          file=sys.stderr)
    return train_df[keep_idx]


FILTER_REGISTRY: dict[str, FilterFn] = {
    "tanimoto_top30": tanimoto_top30,
}


def apply_filter(train_df: pl.DataFrame, test_df: pl.DataFrame, name: str, cache_dir: Path) -> pl.DataFrame:
    """Apply named filter from FILTER_REGISTRY to train_df."""
    if name not in FILTER_REGISTRY:
        raise ValueError(f"Unknown filter: {name!r}. Available: {list(FILTER_REGISTRY.keys())}")
    return FILTER_REGISTRY[name](train_df, test_df, cache_dir)
