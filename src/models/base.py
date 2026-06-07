"""Abstract base classes for PXR challenge models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl


class PXRModel(ABC):
    """Base class for PXR challenge models.

    Operates on pre-computed feature matrices: fit(X, y) / predict(X).
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on feature matrix X and target vector y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for every row in X."""


class MetaModel(ABC):
    """Base class for DataFrame-aware models that manage their own featurization.

    Unlike PXRModel (which receives a single pre-computed feature matrix), a
    MetaModel receives the raw DataFrame plus the feature cache directory, so it
    can featurize multiple heterogeneous representations internally, run its own
    cross-validation, and compute SMILES-based meta-features. Stacking ensembles
    live here.
    """

    name: ClassVar[str] = "meta"

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, cache_dir: Path) -> None:
        """Fit on the training DataFrame, featurizing internally from cache_dir."""

    @abstractmethod
    def predict(self, df: pl.DataFrame, cache_dir: Path) -> np.ndarray:
        """Return predictions for every row of df."""
