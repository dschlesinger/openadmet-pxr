"""Baseline models for the PXR challenge."""

from typing import ClassVar

import polars as pl

from models.base import PXRModel


class MeanBaseline(PXRModel):
    """Predicts the training-set mean for every input."""

    name: ClassVar[str] = "mean_baseline"

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, train: pl.DataFrame, target: str) -> None:
        val = train[target].mean()
        self._mean = float(val) if val is not None else 0.0

    def predict(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("prediction", [self._mean] * len(data))


class MedianBaseline(PXRModel):
    """Predicts the training-set median for every input."""

    name: ClassVar[str] = "median_baseline"

    def __init__(self) -> None:
        self._median: float = 0.0

    def fit(self, train: pl.DataFrame, target: str) -> None:
        val = train[target].median()
        self._median = float(val) if val is not None else 0.0

    def predict(self, data: pl.DataFrame) -> pl.Series:
        return pl.Series("prediction", [self._median] * len(data))
