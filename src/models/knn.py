"""K-Nearest Neighbors regressor for the PXR challenge."""

from typing import ClassVar

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from models.base import PXRModel

DEFAULT_N_NEIGHBORS = 5
DEFAULT_WEIGHTS = "uniform"
DEFAULT_METRIC = "euclidean"


class KNN(PXRModel):
    """KNN regressor wrapping sklearn KNeighborsRegressor."""

    name: ClassVar[str] = "knn"

    def __init__(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        weights: str = DEFAULT_WEIGHTS,
        metric: str = DEFAULT_METRIC,
    ) -> None:
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
