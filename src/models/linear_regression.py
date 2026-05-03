"""Linear regression model for the PXR challenge."""

from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from models.base import PXRModel

DEFAULT_ALPHA = 1.0
DEFAULT_FIT_INTERCEPT = True
DEFAULT_MAX_ITER = 10000
DEFAULT_N_COMPONENTS = 200


class LinearRegression(PXRModel):
    """Ridge regression with VarianceThreshold → StandardScaler → PCA preprocessing."""

    name: ClassVar[str] = "linear_regression"

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        fit_intercept: bool = DEFAULT_FIT_INTERCEPT,
        max_iter: int = DEFAULT_MAX_ITER,
        n_components: int = DEFAULT_N_COMPONENTS,
    ) -> None:
        self._model = Pipeline([
            ("variance_threshold", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("scale", FunctionTransformer(lambda x: x / n_components)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
