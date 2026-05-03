"""XGBoost regressor for the PXR challenge."""

from typing import ClassVar, Optional

import numpy as np
from xgboost import XGBRegressor

from models.base import PXRModel

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLSAMPLE_BYTREE = 0.8
DEFAULT_RANDOM_STATE = 42


class XGBoost(PXRModel):
    """XGBoost gradient-boosted tree regressor."""

    name: ClassVar[str] = "xgboost"

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        subsample: float = DEFAULT_SUBSAMPLE,
        colsample_bytree: float = DEFAULT_COLSAMPLE_BYTREE,
        random_state: Optional[int] = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
