"""Baseline models for the PXR challenge."""

from typing import ClassVar

import numpy as np

from models.base import PXRModel


class MeanBaseline(PXRModel):
    """Predicts the training-set mean for every input."""

    name: ClassVar[str] = "mean_baseline"

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(y.mean()) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)


class MedianBaseline(PXRModel):
    """Predicts the training-set median for every input."""

    name: ClassVar[str] = "median_baseline"

    def __init__(self) -> None:
        self._median: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._median = float(np.median(y)) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._median)
