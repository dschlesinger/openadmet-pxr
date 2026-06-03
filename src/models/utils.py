"""Shared preprocessing utilities for PXR models."""

from typing import Optional, Tuple

import numpy as np
import torch


def make_pca(n_components: int):
    """Return a GPU-accelerated PCA (cuML) if CUDA is available, else sklearn randomized PCA."""
    if torch.cuda.is_available():
        try:
            from cuml.decomposition import PCA as CumlPCA

            return CumlPCA(n_components=n_components)
        except ImportError as e:
            print("cuML not found; falling back to CPU PCA.", str(e))
            pass
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, svd_solver="randomized")


def drop_nan_rows(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    label: str = "",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Drop rows that contain any non-finite value (NaN or inf) from X.

    Catches both NaN (invalid SMILES in RDKit descriptors) and inf (pathological
    descriptor values, e.g. Ipc or BalabanJ for certain ring systems).
    Prints a message if any rows are dropped.
    Returns (X_clean, y_clean) — y_clean is None when y is None.
    """
    mask = np.isfinite(X).all(axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        tag = f" ({label})" if label else ""
        print(f"Dropped {n_dropped} of {len(X)} examples with non-finite features{tag}")
    if y is not None:
        return X[mask], y[mask]
    return X[mask], None
