"""Abstract base class for PXR challenge models."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class PXRModel(ABC):
    """Base class for PXR challenge models."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on feature matrix X and target vector y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for every row in X."""
