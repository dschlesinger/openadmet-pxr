"""Abstract base class for PXR challenge models."""

from abc import ABC, abstractmethod
from typing import ClassVar, List
from representations import Representation

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

class PXRPreConfigModel(PXRModel):
    """Class for models with preconfigured input, allows for more specific model pipelines

    Does not dynamically take different reperesentions!
    """
    required_repersentations: ClassVar[List[str]] = []