"""Symbolic regression model for the PXR challenge (via PySR)."""

import os
import warnings
from typing import ClassVar, Optional

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="torch was imported before juliacall")
    warnings.filterwarnings("ignore", message="juliacall module already imported")
    from pysr import PySRRegressor  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import PXRModel

DEFAULT_NITERATIONS = 40
DEFAULT_POPULATIONS = 15
DEFAULT_RANDOM_STATE = 42


class SymbolicRegression(PXRModel):
    """Genetic-programming symbolic regressor with variance-threshold + scaling preprocessing."""

    name: ClassVar[str] = "symbolic_regression"

    def __init__(
        self,
        niterations: int = DEFAULT_NITERATIONS,
        populations: int = DEFAULT_POPULATIONS,
        random_state: Optional[int] = DEFAULT_RANDOM_STATE,
    ) -> None:
        self._preprocess = Pipeline([
            ("vt", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
        ])
        self._model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "log", "abs", "exp"],
            random_state=random_state,
            deterministic=True,
            parallelism="serial",
            temp_equation_file=True,
            delete_tempfiles=True,
            verbosity=0,
            progress=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s: np.ndarray = self._preprocess.fit_transform(X)
        self._model.fit(X_s, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s: np.ndarray = self._preprocess.transform(X)
        return self._model.predict(X_s)
