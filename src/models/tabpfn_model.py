"""TabPFN regressor for the PXR challenge."""

from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from models.base import PXRModel

DEFAULT_N_ESTIMATORS = 8
DEFAULT_PCA_COMPONENTS = 200
DEFAULT_RANDOM_STATE = 42


class TabPFN(PXRModel):
    """TabPFN (Tabular Prior-Data Fitted Networks) regressor.

    In-context learner pre-trained on synthetic tabular data; no gradient
    steps are taken at fit time — the training set is stored as context and
    used directly at inference.

    Input features are first reduced to PCA_COMPONENTS via PCA (with prior
    standardisation) to keep the dimensionality within TabPFN's sweet spot.
    """

    name: ClassVar[str] = "tabpfn"

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        pca_components: int = DEFAULT_PCA_COMPONENTS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._preprocess = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=random_state)),
        ])
        self._model = TabPFNRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_r: np.ndarray = self._preprocess.fit_transform(X)
        self._model.fit(X_r, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_r: np.ndarray = self._preprocess.transform(X)
        return self._model.predict(X_r)
