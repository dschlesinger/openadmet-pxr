"""Abstract base class for PXR challenge models."""

from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl


class PXRModel(ABC):
    """Base class for PXR challenge models."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, train: pl.DataFrame, target: str) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, data: pl.DataFrame) -> pl.Series:
        """Return predictions for every row in data."""
