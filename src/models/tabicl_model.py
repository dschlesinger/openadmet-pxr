"""TabICL regressor for the PXR challenge."""

from typing import ClassVar

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

from models.base import PXRModel
from models.utils import make_pca

DEFAULT_N_ESTIMATORS = 8
DEFAULT_PCA_COMPONENTS = 200


class TabICL(PXRModel):
    """TabICL (Tabular In-Context Learning) regressor.

    In-context learner pre-trained on real tabular data; no gradient steps are
    taken at fit time — the training set is stored as context and used directly
    at inference.

    Input features are first reduced to PCA_COMPONENTS via PCA (with prior
    standardisation) to keep the dimensionality manageable.
    """

    name: ClassVar[str] = "tabicl"

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        pca_components: int = DEFAULT_PCA_COMPONENTS,
    ) -> None:
        self._preprocess = Pipeline([
            ("variance_threshold", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("pca", make_pca(pca_components)),
        ])
        self._model = TabICLRegressor(n_estimators=n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        print(f"Fitting {self.name} with {X.shape[0]} samples and {X.shape[1]} features...")
        X_r: np.ndarray = self._preprocess.fit_transform(X)
        print(f"Preprocessed features shape: {X_r.shape}")
        self._model.fit(X_r, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_r: np.ndarray = self._preprocess.transform(X)
        return self._model.predict(X_r)
