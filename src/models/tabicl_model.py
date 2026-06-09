"""TabICL regressor for the PXR challenge."""

from typing import ClassVar

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

from models.base import PXRModel
from models.utils import make_pca

# n_estimators is the ensemble size: each member holds the full training set as
# in-context memory, so peak RAM scales with it. The library default (8) OOMs on
# memory-constrained machines (~4 GB GPU / a few GB host) during predict; 2 keeps
# the footprint safe with negligible accuracy cost. Mixed precision (use_amp) and
# pinning to CUDA when available further bound host-RAM use.
DEFAULT_N_ESTIMATORS = 8
DEFAULT_PCA_COMPONENTS = 200
DEFAULT_BATCH_SIZE = 2


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
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._preprocess = Pipeline(
            [
                ("variance_threshold", VarianceThreshold(0.0)),
                ("scaler", StandardScaler()),
                ("pca", make_pca(pca_components)),
            ]
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = TabICLRegressor(
            n_estimators=n_estimators,
            batch_size=batch_size,
            use_amp=True,
            device=device,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        print(f"Fitting {self.name} with {X.shape[0]} samples and {X.shape[1]} features...")
        X_r: np.ndarray = self._preprocess.fit_transform(X)
        print(f"Preprocessed features shape: {X_r.shape}")
        self._model.fit(X_r, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_r: np.ndarray = self._preprocess.transform(X)
        return self._model.predict(X_r)
