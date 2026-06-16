"""ElasticNetCV regressor for the PXR challenge.

Thin PXRModel wrapper around ``StandardScaler -> ElasticNetCV``. ElasticNetCV
selects its own ``alpha`` by internal cross-validation, so Optuna only needs to
tune the elastic-net mixing (``l1_ratio``) and the CV/path settings. This mirrors
the level-1 meta-learner inside ``StackedEnsemble`` and is the intended model to
pass to ``tune-stack --model elasticnet_cv`` when optimizing on OOF predictions.
"""

from typing import ClassVar, Sequence, Union

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import PXRModel

# Default l1_ratio grid matches StackedEnsemble's meta-learner.
DEFAULT_L1_RATIO: Sequence[float] = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
DEFAULT_N_ALPHAS = 100
DEFAULT_CV = 5
DEFAULT_MAX_ITER = 5000
DEFAULT_RANDOM_STATE = 42


class ElasticNetCVModel(PXRModel):
    """StandardScaler -> ElasticNetCV with internally cross-validated alpha."""

    name: ClassVar[str] = "elasticnet_cv"

    def __init__(
        self,
        l1_ratio: Union[float, Sequence[float]] = DEFAULT_L1_RATIO,
        n_alphas: int = DEFAULT_N_ALPHAS,
        cv: int = DEFAULT_CV,
        max_iter: int = DEFAULT_MAX_ITER,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNetCV(
                        l1_ratio=list(l1_ratio) if isinstance(l1_ratio, Sequence) else l1_ratio,
                        alphas=n_alphas,  # sklearn>=1.9: an int = number of alphas along the path
                        cv=cv,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
